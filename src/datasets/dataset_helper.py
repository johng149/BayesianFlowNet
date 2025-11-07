import torch
from torch import Tensor
from torch.nn import functional as F


def sample_t(batch_size: int, min_t=1e-6) -> Tensor:
    return torch.clamp(torch.FloatTensor(batch_size).uniform_(0, 1), min=min_t)


def theta(y: Tensor):
    """
    Args:
        y: Tensor of shape (batch_size, seq_len, K) representing the noisy version of kron_x.
    Returns:
        Tensor representing the scaled softmax of y, which is the input to the model.
    """
    assert y.ndim == 3, "y should be a 3D tensor of shape (batch_size, seq_len, K)"
    theta = F.softmax(y, dim=-1)
    theta = 2 * theta - 1  # scale to [-1, 1]
    return theta
