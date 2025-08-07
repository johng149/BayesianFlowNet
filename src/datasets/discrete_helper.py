import torch
from torch import Tensor
from torch.nn import functional as F

def beta_t(beta_1: Tensor, t: Tensor) -> Tensor:
    """
    Args:
        beta_1: Maximum possible accuracy (reached when t=1) of shape (batch_size,).
        t: A tensor representing the time step, where 1 corresponds to maximum accuracy of shape (batch_size,).
    Returns:
        Beta value at given time step t
    """
    assert beta_1.ndim == 1, "beta_1 should be a 1D tensor"
    assert t.ndim == 1, "t should be a 1D tensor"
    assert beta_1.shape == t.shape, "beta_1 and t should have the same shape"
    assert torch.all(t >= 0), "t must be at least 0"
    assert torch.all(t <= 1), "t must be at most 1"
    return beta_1 * (t ** 2)

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
    beta = beta.view(-1, 1, 1) # allows for broadcasting with reach appropriate batch in kron_x
    mean = beta * (K * kron_x - 1)
    variance = beta * K
    epsilon = torch.normal(0, 1, kron_x.shape, device=kron_x.device)
    return mean + (variance ** 0.5) * epsilon

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
   return torch.clamp(torch.FloatTensor(batch_size).uniform_(0,1), min=min_t)

def collate_fn(batch):
    """
    This collate function will truncate all sequences to the minimum length of
    the sequences in the batch

    Args:
        batch: List of dictionaries, each containing 'x', 't', and 'beta'.
    Returns:
        A dictionary with keys 'x', 't', 'beta_1' and 'theta', where 'x' is a tensor of shape
        (batch_size, seq_len, K), 't' is a tensor of shape (batch_size,), 'beta_1'
        is a tensor of shape (batch_size,), and 'theta' is the transformed version of 'x'.
    """
    x = [item['x'] for item in batch]
    min_length = min(seq.shape[0] for seq in x)
    x = [tensor[:min_length] for tensor in x]

    x = torch.stack(x, dim=0)  # Shape: (batch_size, seq_len, K)
    t = torch.cat([item['t'] for item in batch], dim=0)  # Shape: (batch_size,)
    beta = torch.cat([item['beta'] for item in batch], dim=0)
    beta_1 = torch.cat([item['beta_1'] for item in batch], dim=0)  # Shape: (batch_size,)

    y = y_distribution(beta, x.shape[-1], x)  # Shape: (batch_size, seq_len, K)
    theta_tensor = theta(y)  # Shape: (batch_size, seq_len, K)

    return {
        'x': x,
        't': t,
        'beta_1': beta_1,
        'theta': theta_tensor
    }