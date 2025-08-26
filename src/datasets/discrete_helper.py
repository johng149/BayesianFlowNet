from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import functional as F


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


def sample_t(batch_size: int, min_t: float = 1e-6) -> Tensor:
    return torch.clamp(torch.FloatTensor(batch_size).uniform_(0, 1), min=min_t)


def collate_fn(batch: List[Dict[str, Tensor]], vocab_size: int):
    """
    This collate function will truncate all sequences to the minimum length of
    the sequences in the batch

    Args:
        batch: List of dictionaries, each containing 'x', 't'
        vocab_size: Size of the vocabulary (K)
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

    return {"x": x, "t": t}
