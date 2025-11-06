import torch
from torch import Tensor


def sample_t(batch_size: int, min_t=1e-6) -> Tensor:
    return torch.clamp(torch.FloatTensor(batch_size).uniform_(0, 1), min=min_t)
