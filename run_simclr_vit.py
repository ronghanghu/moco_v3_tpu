import os
import pprint
import time

import torch
import torchvision
import torchvision.transforms as T

import config
from losses import SimCLRLoss
from models import SimCLRViTModel
from distributed import (
    get_world_size,
    get_rank,
    is_master,
    is_xla,
    broadcast_xla_master_model_param,
    infer_init_method,
    distributed_init,
    master_print,
    synchronize,
    reduce_tensor,
    save_ckpt,
    load_ckpt,
)
from schedulers import get_warmup_cosine_scheduler
from transforms import ImgPilColorDistortion, ImgPilGaussianBlur, MultiViewGenerator
from utils import SmoothedValue


try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
except ImportError:
    xm = xmp = pl = xu = None


def load_training_data():
    world_size = get_world_size()
    local_batch_size = cfg.batch_size // world_size
    if cfg.fake_data:
        train_dataset_len = 1281167  # Exactly the size of Imagenet dataset.
        train_loader = xu.SampleGenerator(
            data=(
                torch.zeros(2 * local_batch_size, 3, 224, 224),
                torch.zeros(local_batch_size, dtype=torch.int64),
            ),
            sample_count=train_dataset_len // local_batch_size // world_size,
        )
        train_sampler = None
        return [None] * train_dataset_len, train_loader, train_sampler

    master_print(f"loading images from disk folder: {cfg.data_dir}")
    simclr_transform = MultiViewGenerator(
        T.Compose(
            [
                T.RandomResizedCrop(size=224),
                T.RandomHorizontalFlip(p=0.5),
                ImgPilColorDistortion(strength=0.5),
                ImgPilGaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        n_views=2,
    )
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data_dir, "train"), simclr_transform
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=get_rank(),
        drop_last=cfg.drop_last,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        drop_last=cfg.drop_last,
        collate_fn=collate_fn,
        shuffle=False if train_sampler else True,
        num_workers=cfg.num_workers,
    )

    synchronize()
    master_print("data loading done!")

    return train_dataset, train_loader, train_sampler


def collate_fn(multi_view_img_list):
    """
    For N images with 2 views, it returns (2*N, C, H, W) shape, arranged as
    [img_1_view_1, ..., img_N_view_1, img_1_view_1, ..., img_N_view_1]
    and can be reshaped to (2, N, C, H, W) for loss computation
    """
    img_list = []
    for n_view in range(2):
        img_list.extend(views[n_view] for views, _ in multi_view_img_list)
    label_list = [label for _, label in multi_view_img_list]
    return torch.stack(img_list), torch.tensor(label_list, dtype=torch.long)


def train():
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs
    assert batch_size % get_world_size() == 0
    train_dataset, train_loader, train_sampler = load_training_data()
    model = SimCLRViTModel(
        cfg.vit_model_class, cfg.freeze_patch_embed, cfg.simclr_embed_dim
    )
    if is_xla():
        device = xm.xla_device()
        train_loader = pl.MpDeviceLoader(train_loader, device)
        model = model.to(device)
        broadcast_xla_master_model_param(model)
    else:
        device = torch.device(f"cuda:{cfg.device_id}")
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.device_id], output_device=cfg.device_id
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    iters_per_epoch = len(train_dataset) / batch_size
    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_iteration=int(iters_per_epoch * cfg.warmup_epochs),
        max_iteration=int(iters_per_epoch * num_epochs),
    )
    scaler = None
    if cfg.use_pytorch_amp:
        scaler = torch.cuda.amp.GradScaler()
    loss_fn = SimCLRLoss(temperature=cfg.simclr_loss_temperature)
    if is_master():
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
    master_print("\nmodel:")
    master_print(model, end="\n\n")

    resume_ckpt_path = None
    if cfg.resume_training:
        if cfg.resume_ckpt_path == "<auto-resume-latest>":
            # find the lastest checkpoint file
            for e in range(1, num_epochs + 1):
                try_path = os.path.join(
                    cfg.ckpt_dir, f"{cfg.ckpt_prefix}_epoch_{e}.ckpt"
                )
                if os.path.exists(try_path):
                    resume_ckpt_path = try_path
        else:
            assert os.path.exists(cfg.resume_ckpt_file)
            resume_ckpt_path = cfg.resume_ckpt_file
    if resume_ckpt_path is not None:
        meta_data = load_ckpt(resume_ckpt_path, model, optimizer, lr_scheduler, scaler)
        last_ckpt_epoch = meta_data["epoch"]
    else:
        last_ckpt_epoch = 0

    synchronize()
    smoothed_loss = SmoothedValue(window_size=20)
    model.train()

    master_print(
        "training begins (note that the first few XLA iterations "
        "are very slow due to compilation)"
    )
    for epoch in range(last_ckpt_epoch + 1, num_epochs + 1):
        master_print(f"starting epoch {epoch}")
        time_b = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for step, (data, target) in enumerate(train_loader):
            # forward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(data)
                loss = loss_fn(output)

            # backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if is_xla():
                # PyTorch XLA requires manually reducing gradients
                xm.reduce_gradients(optimizer)

            # param update
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step()

            if (step + 1) % cfg.log_step_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                reduced_loss = reduce_tensor(loss, average=True).item()
                smoothed_loss.update(reduced_loss, batch_size=target.size(0))
                master_print(
                    f"epoch {epoch} step {(step + 1)}, lr: {lr:.4f}, "
                    f"loss: {reduced_loss:.4f}, "
                    f"loss (avg): {smoothed_loss.avg:.4f}, "
                    f"loss (median): {smoothed_loss.median:.4f}"
                )

        time_elapsed = time.time() - time_b
        master_print(f"epoch {epoch} done ({time_elapsed:.2f} sec)")

        if epoch % cfg.ckpt_epoch_interval == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(
                cfg.ckpt_dir, f"{cfg.ckpt_prefix}_epoch_{epoch}.ckpt"
            )
            meta_data = {"cfg": cfg, "epoch": epoch}
            save_ckpt(ckpt_path, model, optimizer, lr_scheduler, scaler, meta_data)

    master_print("training completed")


def main(device_id, configuration):
    config.cfg = configuration
    distributed_init(configuration, device_id)
    global cfg
    cfg = configuration

    synchronize()
    master_print("\nconfig:")
    master_print(pprint.pformat(cfg), end="\n\n")
    train()


if __name__ == "__main__":
    config.cfg = config.build_cfg_from_argparse()

    if is_xla():
        tpu_cores_per_node = 8
        xmp.spawn(main, args=(config.cfg,), nprocs=tpu_cores_per_node)
    else:
        infer_init_method(config.cfg)
        if config.cfg.no_spawn:
            assert config.cfg.device_id >= 0
            main(config.cfg.device_id, config.cfg)
        else:
            torch.multiprocessing.spawn(
                main, nprocs=torch.cuda.device_count(), args=(config.cfg,)
            )
