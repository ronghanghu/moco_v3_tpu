import os
import logging
import subprocess
import socket
from itertools import chain

import torch
from torch import distributed as dist


try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


logger = None


class XLAGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all TPU workers with support for backward propagation.
    """

    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        tensor_list = xm.all_gather(x.unsqueeze(dim), dim=dim)
        return tensor_list

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        all_grad_output = xm.all_reduce(xm.REDUCE_SUM, grad_output)
        return all_grad_output.select(dim, xm.get_ordinal()), None


class XLAReduceSumLayer(torch.autograd.Function):
    """
    Reduce tensor on TPUs with support for backward propagation.
    Fixing https://github.com/pytorch/xla/issues/2989
    """

    @staticmethod
    def forward(ctx, x):
        return xm.all_reduce(xm.REDUCE_SUM, x)

    @staticmethod
    def backward(ctx, grad_output):
        return xm.all_reduce(xm.REDUCE_SUM, grad_output)


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_tensor_with_backward(tensor, dim=0):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    if is_xla():
        tensor_list = XLAGatherLayer.apply(tensor, dim)
        tensor_list = tensor_list.flatten(start_dim=dim, end_dim=dim + 1)
    else:
        tensor_list = GatherLayer.apply(tensor)
        tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list


def xla_all_reduce_sum_with_backward(tensor):
    return XLAReduceSumLayer.apply(tensor)


def broadcast_xla_master_model_param(model):
    parameters_and_buffers = []
    for p in chain(model.parameters(), model.buffers()):
        # Set all params in non-master devices to zero so that all_reduce is
        # equivalent to broadcasting parameters from master to other devices.
        if not is_master():
            zero = torch.tensor(0, dtype=p.data.dtype, device=p.data.device)
            p.data.mul_(zero)
        parameters_and_buffers.append(p.data)
    xm.wait_device_ops()
    xm.all_reduce(xm.REDUCE_SUM, parameters_and_buffers)
    xm.mark_step()
    xm.rendezvous("broadcast_xla_master_model_param")


def is_xla():
    from config import cfg

    return cfg.device == "xla"


def master_print(message):
    if is_master():
        if logger is not None:
            logger.info(message)
        else:
            print(message, flush=True)


def reduce_tensor(t, average=False):
    world_size = get_world_size()
    if world_size < 2:
        return t

    with torch.no_grad():
        if is_xla():
            scale = 1.0 / world_size if average else 1.0
            t = xm.all_reduce(xm.REDUCE_SUM, t, scale=scale)
        else:
            dist.reduce(t, dst=0)
            if average:
                t /= world_size
    return t


def get_world_size():
    if is_xla():
        return xm.xrt_world_size()
    return dist.get_world_size()


def get_rank():
    if is_xla():
        return xm.get_ordinal()
    return dist.get_rank()


def is_master():
    return get_rank() == 0


def synchronize(message="sync-workers"):
    if is_xla():
        xm.rendezvous(message)
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        return

    dist.barrier()


# adapted from
# https://github.com/facebookresearch/mmf/blob/master/mmf/utils/distributed.py
def infer_init_method(cfg):
    if cfg.init_method != "":
        return

    # if cfg.rank < 0 (default) after spawning,
    # cfg.rank will be filled as cfg.rank_offset + cfg.device_id
    cfg.rank_offset = 0

    # support torch.distributed.launch
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        cfg.init_method = "env://"
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.rank = int(os.environ["RANK"])
        cfg.device_id = int(os.environ["LOCAL_RANK"])
        cfg.no_spawn = True

    # we can determine the init method automatically for Slurm
    else:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            assert cfg.world_size > 0, "world size must be specified for slurm"
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                cfg.init_method = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"), port=cfg.port
                )
                nnodes = int(os.environ.get("SLURM_NNODES"))
                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get("SLURM_NTASKS"))
                    nnodes = int(os.environ.get("SLURM_NNODES"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)

                if ntasks_per_node == 1:
                    assert cfg.world_size % nnodes == 0
                    gpus_per_node = cfg.world_size // nnodes
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    cfg.rank_offset = node_id * gpus_per_node
                    # cfg.rank and cfg.device_id will be filled after spawning
                    cfg.no_spawn = False
                else:
                    assert ntasks_per_node == cfg.world_size // nnodes
                    cfg.rank = int(os.environ.get("SLURM_PROCID"))
                    cfg.device_id = int(os.environ.get("SLURM_LOCALID"))
                    cfg.no_spawn = True
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass
        else:
            # launched locally with `python main_simclr_vit.py`
            cfg.world_size = torch.cuda.device_count()
            # cfg.rank and cfg.device_id will be filled after spawning
            cfg.no_spawn = False
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = cfg.port


def setup_logging(cfg, logging_name):
    import sys

    global logger
    if is_master():
        logger = logging.getLogger(logging_name)
        logger.setLevel(logging.INFO)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(cfg.ckpt_dir, f"{logging_name}.log"))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        sh = logging.StreamHandler(stream=sys.stdout)
        fh.setLevel(logging.INFO)
        logger.addHandler(sh)


def distributed_init(cfg, device_id):
    cfg.device_id = device_id
    if is_xla():
        cfg.world_size = xm.xrt_world_size()
        cfg.rank = xm.get_ordinal()
        return
    if dist.is_initialized():
        cfg.world_size = dist.get_world_size()
        cfg.rank = dist.get_rank()
        return

    if cfg.rank < 0:
        cfg.rank = cfg.rank_offset + device_id

    print(f"Distributed Init (Rank {cfg.rank}): {cfg.init_method}\n", end="")
    dist.init_process_group(
        backend="nccl",
        init_method=cfg.init_method,
        world_size=cfg.world_size,
        rank=cfg.rank,
    )
    print(f"Initialized Host {socket.gethostname()} as rank {cfg.rank}\n", end="")

    torch.cuda.set_device(cfg.device_id)
    # perform a dummy all-reduce to initialize the NCCL communicator
    dist.all_reduce(torch.zeros(1).cuda())
    cfg.world_size = dist.get_world_size()
    cfg.rank = dist.get_rank()


def save_ckpt(ckpt_path, model, optimizer, lr_scheduler, scaler, meta_data):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "meta_data": meta_data,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if is_xla():
        xm.save(ckpt, ckpt_path, global_master=True)
    else:
        if is_master():
            torch.save(ckpt, ckpt_path)

    master_print(f"checkpoint saved to {ckpt_path}")


def load_ckpt(ckpt_path, model, optimizer, lr_scheduler, scaler):
    from config import cfg

    if is_xla():
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{cfg.device_id}")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    meta_data = ckpt["meta_data"]

    master_print(f"resumed from checkpoint {ckpt_path}")
    return meta_data
