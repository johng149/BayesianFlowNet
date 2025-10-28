import math

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F


def beta_t(beta_1: Tensor, t: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Args:
        beta_1: Maximum possible accuracy (reached when t=1) of shape (batch_size,).
        t: A tensor representing the time step, where 1 corresponds to maximum accuracy of shape (batch_size,).
        eps: Small value to prevent beta at t=0 from being exactly 0.
    Returns:
        Beta value at given time step t
    """
    assert beta_1.ndim == 1, "beta_1 should be a 1D tensor"
    assert t.ndim == 1, "t should be a 1D tensor"
    assert beta_1.shape == t.shape, "beta_1 and t should have the same shape"
    assert torch.all(t >= 0), "t must be at least 0"
    assert torch.all(t <= 1), "t must be at most 1"
    return beta_1 * (t**2) + eps


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
    return (
        mean + variance * epsilon
    )  # I know the name `variance` suggests it should be squared, but this works just fine


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


def base_dims(base: int, K: int) -> int:
    """
    Calculate the number of digits in the `base` number system required to represent
    at least K distinct values.

    Args:
        base: The base of the number system (e.g., 2 for binary, 10 for decimal).
        K: The number of distinct values to represent.
    Returns:
        The minimum number of digits needed in the specified base to represent at least K values.
    """
    return math.ceil(math.log((-(1 - base) * K - 1), base))


def base_encode(values: Tensor, base: int, K: int) -> Tensor:
    """
    Given a tensor of shape (batch_size, seq_len) containing integer values in the range [0, K-1],
    encode each integer into its vector representation in the specified base.

    Args:
        values: Tensor of shape (batch_size, seq_len) with integer values in [0, K-1].
        base: The base for encoding (e.g., 2 for binary).
        K: The number of distinct values (vocabulary size).
    Returns:
        A tensor of shape (batch_size, seq_len, base_dims) where each integer is represented
        by its digits in the specified base.

    For example, if base is 3 and values is:
    tensor([[11,  6, 12,  0],
            [11,  3,  9, 12]])

    the flat_rep will be:
    tensor([[0, 1, 0, 2],
            [0, 0, 2, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 2],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0]])

    and the output will be the one-hot encoding of flat_rep with shape
    (batch_size, seq_len * base_dims, base):
    tensor([[
            [1, 0, 0], # 0
            [0, 1, 0], # 1
            [1, 0, 0], # 0
            [0, 0, 1], # 2
            ...
    ])
    """
    assert (
        values.ndim == 2
    ), "values should be a 2D tensor of shape (batch_size, seq_len)"
    batch_size, seq_len = values.shape
    dims = base_dims(base, K)
    powers = base ** torch.arange(dims - 1, -1, -1, device=values.device)

    values = values.view(-1, 1)
    flat_rep = (values // powers) % base
    vector_rep = F.one_hot(flat_rep, num_classes=base)
    vector_rep = rearrange(
        vector_rep,
        "(batch seq_len) dims base -> batch (seq_len dims) base",
        batch=batch_size,
    )
    return vector_rep


def base_decode(vector_rep: Tensor, base: int, K: int) -> Tensor:
    """
    Decode a tensor from its base representation back to integer values.

    Args:
        vector_rep: Tensor of shape (batch_size, seq_len * base_dims, base) representing
                     the one-hot encoded digits in the specified base.
        base: The base used for encoding.
        K: The number of distinct values (vocabulary size).
    Returns:
        A tensor of shape (batch_size, seq_len) containing the decoded integer values.
    """
    assert (
        vector_rep.ndim == 3
    ), "vector_rep should be a 3D tensor of shape (batch_size, seq_len * base_dims, base)"
    batch_size, total_seq_len, _ = vector_rep.shape
    dims = base_dims(base, K)
    recombined = rearrange(
        vector_rep, "batch (seq_len dims) base -> (batch seq_len) dims base", dims=dims
    )
    flat_rep = torch.argmax(recombined, dim=-1)
    powers = base ** torch.arange(dims - 1, -1, -1, device=vector_rep.device)
    values = (flat_rep * powers).sum(dim=-1)
    values = values.view(batch_size, -1)
    return values


def collate_fn(batch):
    """
    This collate function will truncate all sequences to the minimum length of
    the sequences in the batch

    Args:
        batch: List of dictionaries, each containing 'x', 't', 'beta', 'beta_1', and 'K'
    Returns:
        A dictionary with keys 'x', 't', 'beta_1' and 'theta', where 'x' is a tensor of shape
        (batch_size, seq_len, K), 't' is a tensor of shape (batch_size,), 'beta_1'
        is a tensor of shape (batch_size,), and 'theta' is the transformed version of 'x'.
    """
    K = batch[0]["K"]
    base = batch[0]["base"]
    x = [item["x"] for item in batch]
    min_length = min(seq.shape[0] for seq in x)
    x = [tensor[:min_length] for tensor in x]

    x = torch.stack(x, dim=0)  # Shape: (batch_size, seq_len)
    x = base_encode(x, base, K)  # Shape: (batch_size, seq_len * base_dims, base)
    t = torch.cat([item["t"] for item in batch], dim=0)  # Shape: (batch_size,)
    beta = torch.cat([item["beta"] for item in batch], dim=0)
    beta_1 = torch.cat(
        [item["beta_1"] for item in batch], dim=0
    )  # Shape: (batch_size,)

    y = y_distribution(
        beta, x.shape[-1], x
    )  # Shape: (batch_size, seq_len * base_dims, K)
    theta_tensor = theta(y)  # Shape: (batch_size, seq_len * base_dims, K)

    return {
        "x": x,
        "t": t,
        "beta_1": beta_1,
        "theta": theta_tensor,
        "base": base,
        "K": K,
    }
