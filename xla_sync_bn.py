import torch
from torch import nn

from distributed import get_world_size, xla_all_reduce_sum_with_backward

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


class XLASyncBNTrainModeOnly(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, batch):
        assert isinstance(batch, torch.Tensor) and batch.ndim == 2

        local_mean = torch.mean(batch, dim=0)
        local_sqr_mean = torch.mean(batch * batch, dim=0)

        scale = 1.0 / get_world_size()
        mean = xla_all_reduce_sum_with_backward(local_mean) * scale
        sqr_mean = xla_all_reduce_sum_with_backward(local_sqr_mean) * scale
        var = sqr_mean - mean.pow(2)

        batch = (batch - mean) / torch.sqrt(var + self.eps)
        batch = batch * self.weight + self.bias
        return batch

    def extra_repr(self) -> str:
        return "dim={}, eps={}".format(self.dim, self.eps)
