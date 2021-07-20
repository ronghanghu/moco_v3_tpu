import os
import pprint
import time

import torch
import torchvision
import torchvision.transforms as T

import config
from models import LinearEvalViTModel
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
    local_batch_size = cfg.linear_eval.batch_size // world_size

    master_print(f"loading images from disk folder: {cfg.data_dir}")
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(size=224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = T.Compose(
        [
            T.Resize(size=256),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data_dir, "train"), train_transform
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
        shuffle=False if train_sampler else True,
        num_workers=cfg.num_workers,
    )

    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg.data_dir, "val"), val_transform
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=get_rank(),
        drop_last=cfg.drop_last,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        sampler=val_sampler,
        drop_last=cfg.drop_last,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    synchronize()
    master_print("data loading done!")

    return (
        train_dataset,
        train_loader,
        train_sampler,
        val_dataset,
        val_loader,
        val_sampler,
    )


def train():
    batch_size = cfg.linear_eval.batch_size
    num_epochs = cfg.linear_eval.num_epochs
    assert batch_size % get_world_size() == 0
    train_dataset, train_loader, train_sampler, _, val_loader, _ = load_training_data()
    model = LinearEvalViTModel(cfg.vit_model_class, cfg.linear_eval.num_classes)
    master_print(f"linear evaluation for: {cfg.linear_eval.pretrained_ckpt_path}")
    model.load_from_pretrained_checkpoint(
        cfg.linear_eval.pretrained_ckpt_path,
        reset_last_ln=cfg.linear_eval.reset_last_ln,
    )
    if is_xla():
        device = xm.xla_device()
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)
        model = model.to(device)
        broadcast_xla_master_model_param(model)
    else:
        device = torch.device(f"cuda:{cfg.device_id}")
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.device_id], output_device=cfg.device_id
        )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.linear_eval.lr,
        momentum=cfg.linear_eval.momentum,
        weight_decay=cfg.linear_eval.weight_decay,
    )
    lr_scheduler = get_warmup_cosine_scheduler(
        optimizer, 100, len(train_dataset) * num_epochs // batch_size
    )
    scaler = None
    if cfg.use_pytorch_amp:
        scaler = torch.cuda.amp.GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss()
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
                    cfg.ckpt_dir, f"{cfg.ckpt_prefix}_LEepoch_{e}.ckpt"
                )
                if os.path.exists(try_path):
                    resume_ckpt_path = try_path
        else:
            assert os.path.exists(cfg.resume_ckpt_file)
            resume_ckpt_path = cfg.resume_ckpt_file
    if resume_ckpt_path is not None:
        meta_data = load_ckpt(resume_ckpt_path, model, optimizer, lr_scheduler, scaler)
        last_ckpt_epoch = meta_data["epoch"]
        best_accuracy = meta_data["best_accuracy"]
        best_epoch = meta_data["best_epoch"]
    else:
        last_ckpt_epoch = 0
        best_accuracy = 0.0
        best_epoch = 0

    synchronize()
    smoothed_loss = SmoothedValue(window_size=20)
    model.eval()  # use eval mode for linear evaluation training

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
                loss = loss_fn(output, target.to(device) if not is_xla() else target)

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

        if epoch % cfg.linear_eval.test_epoch_interval == 0 or epoch == num_epochs:
            accuracy, _, _ = eval_on_val(val_loader, model, scaler, device)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
            master_print(
                f"accuracy on val: {accuracy:.4f} "
                f"(best accuracy {best_accuracy:.4f} at epoch {best_epoch})"
            )
        if epoch % cfg.linear_eval.ckpt_epoch_interval == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(
                cfg.ckpt_dir, f"{cfg.ckpt_prefix}_LEepoch_{epoch}.ckpt"
            )
            meta_data = {
                "cfg": cfg,
                "epoch": epoch,
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
            }
            save_ckpt(ckpt_path, model, optimizer, lr_scheduler, scaler, meta_data)

    master_print("training completed")


def eval_on_val(val_loader, model, scaler, device):
    local_correct = torch.tensor(0, device=device)
    local_total = torch.tensor(0, device=device)
    for data, target in val_loader:
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(data)
            pred = output.argmax(dim=-1)
        target = target.to(device) if not is_xla() else target
        local_correct += pred.eq(target.view_as(pred)).sum()
        local_total += torch.tensor(target.size(0), device=device)
    correct = reduce_tensor(local_correct.float(), average=False).item()
    total = reduce_tensor(local_total.float(), average=False).item()
    accuracy = correct / total
    return accuracy, correct, total


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
