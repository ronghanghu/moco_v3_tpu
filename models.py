from itertools import chain

import torch
from torch import nn

from config import cfg
from distributed import is_xla
import vision_transformer
from xla_sync_bn import XLASyncBNTrainModeOnly


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=4096):
        super().__init__()
        # a 3-layer projection head based on MoCo V3 paper in
        # https://arxiv.org/abs/2104.02057
        bn_class = XLASyncBNTrainModeOnly if is_xla() else nn.SyncBatchNorm
        layers = [
            nn.Linear(input_dim, hidden_dim),
            bn_class(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            bn_class(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            bn_class(output_dim),
        ]
        # use only gamma but not beta in the last BN layer
        nn.init.zeros_(layers[-1].bias)
        layers[-1].bias.requires_grad = False
        self.clf = nn.Sequential(*layers)

    def forward(self, batch):
        out = self.clf(batch)
        return out


class PredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=4096):
        super().__init__()
        # a 2-layer prediction head based on MoCo V3 paper in
        # https://arxiv.org/abs/2104.02057
        bn_class = XLASyncBNTrainModeOnly if is_xla() else nn.SyncBatchNorm
        layers = [
            nn.Linear(input_dim, hidden_dim),
            bn_class(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            bn_class(output_dim),
        ]
        # use only gamma but not beta in the last BN layer
        nn.init.zeros_(layers[-1].bias)
        layers[-1].bias.requires_grad = False
        self.clf = nn.Sequential(*layers)

    def forward(self, batch):
        out = self.clf(batch)
        return out


class SimCLRViTModel(nn.Module):
    def __init__(
        self, vit_model_class, vit_pos_embed_type, freeze_patch_embed, simclr_embed_dim
    ):
        super().__init__()
        vit_trunk = getattr(vision_transformer, vit_model_class)(**cfg.vit)
        vit_trunk.head = nn.Identity()  # remove the classifier layer
        init_vit_and_pos_embedding(vit_trunk, vit_pos_embed_type)
        if freeze_patch_embed:
            # freezing ViT patch embedding as in MoCo V3 to improve stability
            for p in vit_trunk.patch_embed.parameters():
                p.requires_grad = False
        vit_hidden_dim = vit_trunk.cls_token.size(-1)

        self.trunk = vit_trunk
        self.ssl_head = ProjectionHead(vit_hidden_dim, simclr_embed_dim)

    def forward(self, images):
        features = self.trunk.forward_features(images)
        simclr_embeddings = self.ssl_head(features)
        return simclr_embeddings


class MoCoV3ViTModel(nn.Module):
    def __init__(
        self, vit_model_class, vit_pos_embed_type, freeze_patch_embed, mocov3_embed_dim
    ):
        super().__init__()
        vit_trunk = getattr(vision_transformer, vit_model_class)(**cfg.vit)
        vit_trunk.head = nn.Identity()  # remove the classifier layer
        init_vit_and_pos_embedding(vit_trunk, vit_pos_embed_type)
        if freeze_patch_embed:
            # freezing ViT patch embedding as in MoCo V3 to improve stability
            for p in vit_trunk.patch_embed.parameters():
                p.requires_grad = False
        vit_hidden_dim = vit_trunk.cls_token.size(-1)

        self.trunk = vit_trunk
        self.proj_head = ProjectionHead(vit_hidden_dim, mocov3_embed_dim)
        self.pred_head = PredictionHead(mocov3_embed_dim, mocov3_embed_dim)

        # a momentum copy of the trunk and the projection head
        vit_trunk_m = getattr(vision_transformer, vit_model_class)(**cfg.vit)
        vit_trunk_m.head = nn.Identity()  # remove the classifier layer
        init_vit_and_pos_embedding(vit_trunk_m, vit_pos_embed_type)
        self.trunk_m = vit_trunk_m
        self.proj_head_m = ProjectionHead(vit_hidden_dim, mocov3_embed_dim)
        # initialize the momentum copies from the same parameter
        self.trunk_m.load_state_dict(self.trunk.state_dict())
        self.proj_head_m.load_state_dict(self.proj_head.state_dict())
        for p in chain(self.trunk_m.parameters(), self.proj_head_m.parameters()):
            p.requires_grad = False

    def _f_q_trunk(self, images):
        return self.trunk.forward_features(images)

    def _f_k_trunk(self, images):
        with torch.no_grad():
            return self.trunk_m.forward_features(images)

    def _f_q_head(self, features):
        return self.pred_head(self.proj_head(features))

    def _f_k_head(self, features):
        with torch.no_grad():
            return self.proj_head_m(features)

    def forward(self, images):
        q_feat = self._f_q_trunk(images)
        k_feat = self._f_k_trunk(images)

        # separately forward the two augmentations through the heads so that BNs are
        # applied separately (see appendix of https://arxiv.org/abs/2104.02057)
        q_feat = q_feat.view(2, q_feat.shape[0] // 2, q_feat.shape[1])
        q1, q2 = self._f_q_head(q_feat[0]), self._f_q_head(q_feat[1])
        k_feat = k_feat.view(2, k_feat.shape[0] // 2, k_feat.shape[1])
        k1, k2 = self._f_k_head(k_feat[0]), self._f_k_head(k_feat[1])
        return q1, q2, k1, k2

    def update_momentum_params(self, momentum):
        for p, p_m in zip(self.trunk.parameters(), self.trunk_m.parameters()):
            p_m.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)
        for p, p_m in zip(self.proj_head.parameters(), self.proj_head_m.parameters()):
            p_m.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)


class LinearEvalViTModel(nn.Module):
    def __init__(self, vit_model_class, vit_pos_embed_type, num_classes):
        super().__init__()
        vit_trunk = getattr(vision_transformer, vit_model_class)(**cfg.vit)
        vit_trunk.head = nn.Identity()  # remove the classifier layer
        init_vit_and_pos_embedding(vit_trunk, vit_pos_embed_type)
        # freezing the trunk for linear evaluation
        for p in vit_trunk.parameters():
            p.requires_grad = False
        vit_hidden_dim = vit_trunk.cls_token.size(-1)

        self.trunk = vit_trunk
        self.classifier_head = nn.Linear(vit_hidden_dim, num_classes)

    def load_from_pretrained_checkpoint(self, pretrained_ckpt_path, reset_last_ln):
        pretrained_ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
        if "model" in pretrained_ckpt:
            pretrained_ckpt = pretrained_ckpt["model"]
        if not cfg.linear_eval.load_deit_ckpt:
            # convert our MoCo checkpoint by stripping "module.trunk." or "trunk." keys
            keys = list(pretrained_ckpt)
            for k in keys:
                param = pretrained_ckpt.pop(k)
                # keep all the trunk parameters (remove module. for DDP checkpoints)
                if k.startswith("module.trunk.") or k.startswith("trunk."):
                    new_key = k.replace("module.trunk.", "").replace("trunk.", "")
                    pretrained_ckpt[new_key] = param
        else:
            # for DeiT-format checkpoint, remove the "head.*" params
            keys = list(pretrained_ckpt)
            for k in keys:
                if k.startswith("head."):
                    param = pretrained_ckpt.pop(k)
        if reset_last_ln:
            # reset the last ViT LN layer's weight and bias to ones and zeros
            # otherwise their scales could be too large if they are learned with
            # vision SSL heads that involve BN in them.
            if "norm.weight" in pretrained_ckpt and "norm.bias" in pretrained_ckpt:
                pretrained_ckpt["norm.weight"][...] = 1
                pretrained_ckpt["norm.bias"][...] = 0

        self.trunk.load_state_dict(pretrained_ckpt)

    def forward(self, images):
        with torch.no_grad():
            features = self.trunk.forward_features(images)
        logits = self.classifier_head(features)
        return logits


# adapted from https://github.com/facebookresearch/moco-v3/blob/main/vits.py
def init_vit_and_pos_embedding(trunk, vit_pos_embed_type, temperature=10000.0):
    import math
    from functools import reduce
    from operator import mul

    for name, m in trunk.named_modules():
        if isinstance(m, nn.Linear):
            if "qkv" in name:
                # Treat the weights of Q, K, V separately
                val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                nn.init.uniform_(m.weight, -val, val)
            else:
                nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    nn.init.normal_(trunk.cls_token, std=1e-6)

    assert vit_pos_embed_type in ["sin-cos", "learned"]
    if vit_pos_embed_type == "learned":
        # the timm ViT is already built with learnable position embedding
        return

    # initialize the path embedding scale according to patch size
    val = math.sqrt(
        6.0 / float(3 * reduce(mul, trunk.patch_embed.patch_size, 1) + trunk.embed_dim)
    )
    nn.init.uniform_(trunk.patch_embed.proj.weight, -val, val)
    nn.init.zeros_(trunk.patch_embed.proj.bias)

    h, w = trunk.patch_embed.grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert (
        trunk.embed_dim % 4 == 0
    ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    pos_dim = trunk.embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature ** omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
    )[None, :, :]

    assert trunk.num_tokens == 1, "Assuming one and only one token, [CLS]"
    pe_token = torch.zeros([1, 1, trunk.embed_dim], dtype=torch.float32)
    trunk.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    trunk.pos_embed.requires_grad = False
