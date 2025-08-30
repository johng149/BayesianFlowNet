from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import functional as F


def beta_t(beta_1: Tensor, t: Tensor, K: int) -> Tensor:
    """
    Args:
        beta_1: Maximum possible accuracy (reached when t=1) of shape (batch_size,).
        t: A tensor representing the time step, where 1 corresponds to maximum accuracy of shape (batch_size,).
        K: Number of classes (usually vocabulary size etc.)
    Returns:
        Beta value at given time step t
    """
    assert beta_1.ndim == 1, "beta_1 should be a 1D tensor"
    assert t.ndim == 1, "t should be a 1D tensor"
    assert beta_1.shape == t.shape, "beta_1 and t should have the same shape"
    assert torch.all(t >= 0), "t must be at least 0"
    assert torch.all(t <= 1), "t must be at most 1"
    return (-4 / K) * torch.log(1 - t + t * torch.exp(-K * beta_1 / 4))


def alpha(beta_1: Tensor, t: Tensor, K: int) -> Tensor:
    """
    Args:
        beta_1: Maximum possible accuracy (reached when t=1) of shape (batch_size,).
        t: A tensor representing the time step, where 1 corresponds to maximum accuracy of shape (batch_size,).
        K: Number of classes (usually vocabulary size etc.)
    Returns:
        Alpha value (derivative of beta_t) at given time step t
    """
    assert beta_1.ndim == 1, "beta_1 should be a 1D tensor"
    assert t.ndim == 1, "t should be a 1D tensor"
    assert beta_1.shape == t.shape, "beta_1 and t should have the same shape"
    assert torch.all(t >= 0), "t must be at least 0"
    assert torch.all(t <= 1), "t must be at most 1"
    return (
        (4 / K)
        * (1 - torch.exp(-K * beta_1 / 4))
        / (1 - t + t * torch.exp(-K * beta_1 / 4))
    )


def y_distribution(beta: Tensor, K: int, kron_x: Tensor) -> Tensor:
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
    epsilon = torch.normal(0, 1, kron_x.shape, device=kron_x.device)
    return mean + (variance**0.5) * epsilon


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


def sample_t(batch_size, min_t=1e-6):
    return torch.clamp(torch.FloatTensor(batch_size).uniform_(0, 1), min=min_t)


def collate_fn(
    batch: List[Dict[str, Tensor]], vocab_size: int, beta_1: float | None = None
):
    """
    This collate function will truncate all sequences to the minimum length of
    the sequences in the batch

    Args:
        batch: List of dictionaries, each containing 'x', 't'
        vocab_size: Size of the vocabulary (K)
        beta_1: Optional beta_1 value for the batch, if not set, will use 20.4054 / vocab_size
    Returns:
        A dictionary with keys 'x', 't' where 'x' is a tensor of shape
        (batch_size, seq_len, K), 't' is a tensor of shape (batch_size,)
    """
    x = [item["x"] for item in batch]
    min_length = min(seq.shape[0] for seq in x)
    x = [tensor[:min_length] for tensor in x]
    x = [F.one_hot(tensor, num_classes=vocab_size).float() for tensor in x]

    t = torch.cat([item["t"] for item in batch], dim=0)  # Shape: (batch_size * folds,)
    folds = batch[0]["t"].shape[0]  # all items should have the same number of folds

    x = torch.stack(
        [tensor.unsqueeze(0).expand(folds, -1, -1) for tensor in x], dim=0
    )  # Shape: (batch_size, folds, seq_len, K)

    x = x.view(
        -1, x.shape[-2], x.shape[-1]
    )  # Reshape to (batch_size * folds, seq_len, K)

    beta_1 = beta_1 if beta_1 is not None else 20.4054 / vocab_size

    b1 = beta_1 * torch.ones(t.shape, device=t.device)
    bt = beta_t(b1, t, vocab_size)
    a = alpha(b1, t, vocab_size)

    y = y_distribution(bt, vocab_size, x)

    model_input = theta(y)

    return {"x": x, "t": t, "theta": model_input, "alpha": a}
