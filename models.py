import timm
import torch
from torch import nn

from distributed import is_xla
from xla_sync_bn import XLASyncBNTrainModeOnly


class SimCLRProjectionHead(nn.Module):
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
        ]
        self.clf = nn.Sequential(*layers)

    def forward(self, batch):
        out = self.clf(batch)
        return out


class SimCLRViTModel(nn.Module):
    def __init__(self, vit_model_class, freeze_patch_embed, simclr_embed_dim):
        super().__init__()
        vit_trunk = getattr(timm.models.vision_transformer, vit_model_class)()
        vit_trunk.head = nn.Identity()  # remove the classifier layer
        if freeze_patch_embed:
            # freezing ViT patch embedding as in MoCo V3 to improve stability
            for p in vit_trunk.patch_embed.parameters():
                p.requires_grad = False
        vit_hidden_dim = vit_trunk.cls_token.size(-1)

        self.trunk = vit_trunk
        self.ssl_head = SimCLRProjectionHead(vit_hidden_dim, simclr_embed_dim)

    def forward(self, images):
        features = self.trunk.forward_features(images)
        simclr_embeddings = self.ssl_head(features)
        return simclr_embeddings


class LinearEvalViTModel(nn.Module):
    def __init__(self, vit_model_class, num_classes):
        super().__init__()
        vit_trunk = getattr(timm.models.vision_transformer, vit_model_class)()
        vit_trunk.head = nn.Identity()  # remove the classifier layer
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
        keys = list(pretrained_ckpt)
        for k in keys:
            param = pretrained_ckpt.pop(k)
            # keep all the trunk parameters (remove module. for DDP checkpoints)
            if k.startswith("module.trunk.") or k.startswith("trunk."):
                new_key = k.replace("module.trunk.", "").replace("trunk.", "")
                pretrained_ckpt[new_key] = param
        if reset_last_ln:
            # reset the last ViT LN layer's weight and bias to ones and zeros
            # otherwise their scales could be too large if they are learned with
            # vision SSL heads that involve BN in them.
            pretrained_ckpt["norm.weight"][...] = 1
            pretrained_ckpt["norm.bias"][...] = 0

        self.trunk.load_state_dict(pretrained_ckpt)

    def forward(self, images):
        with torch.no_grad():
            features = self.trunk.forward_features(images)
        logits = self.classifier_head(features)
        return logits
