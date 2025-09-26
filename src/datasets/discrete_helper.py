import torch
from torch import Tensor
from torch.nn import functional as F


def y_distribution(
    beta: Tensor, K: int, kron_x: Tensor, deterministic: bool = False
) -> tuple[Tensor, Tensor | float]:
    """
    Args:
        beta: Tensor of accuracy values for each batch of shape (batch_size,).
        K: Number of classes (usually vocabulary size etc.)
        kron_x: One-hot encoded input tensor of shape (batch_size, seq_len, K).
    Returns:
        Noisy version of kron_x with the amount of noise controlled
        by beta. The shape of the output tensor is the same as kron_x, i.e., (batch_size, seq_len, K).
    """
    beta = beta.view(
        -1, 1, 1
    )  # allows for broadcasting with reach appropriate batch in kron_x
    mean = beta * (K * kron_x - 1)
    variance = beta * K
    assert torch.all(variance >= 0), f"Variance has negative values with beta: {beta}"
    epsilon = (
        torch.normal(0, 1, kron_x.shape, device=kron_x.device)
        if not deterministic
        else 0.0
    )
    return mean + (variance**0.5) * epsilon, epsilon


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


def sample_t(batch_size: int, min_t: float = 1e-6) -> Tensor:
    return torch.clamp(torch.FloatTensor(batch_size).uniform_(0, 1), min=min_t)
