import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F

def accuracy(i: Tensor, n: Tensor, beta_1: Tensor) -> Tensor:
    """
    Args:
        i: Current iteration number of shape (batch_size,).
        n: Total number of iterations of shape (batch_size,).
        beta_1: Maximum possible accuracy (reached when t=1) of shape (batch_size,).
    Returns:
        Accuracy at the current iteration i.
    """
    assert torch.all(n > 0), "Must have at least one inference step in total"
    assert torch.all(i > 0), "Must be on at least first inference step"
    assert torch.all(i <= n), "Current iteration must be less than or equal to total iterations"
    
    return beta_1 * (2 * i - 1) / (n ** 2)

def sample_model_output(model_output_logits: Tensor) -> Tensor:
    """
    Args:
        model_output_logits: Model output logits of shape (batch_size, seq_len, K).
    Returns:
        Sampled model output based on the logits.
    """
    batch_size, seq_len, K = model_output_logits.shape
    dist = Categorical(logits=model_output_logits)
    samples = dist.sample()
    return F.one_hot(samples, K)

def y(sampled_one_hot: Tensor, accuracy: Tensor) -> Tensor:
    """
    Args:
        sampled_one_hot: Sampled output described by model logits that has been one-hot encoded, 
                        of shape (batch_size, seq_len, K).
        accuracy: Accuracy at the current iteration of shape (batch_size,).
    Returns:
        Noisy version of the sampled one-hot tensor with the amount of noise controlled by accuracy.
        The shape of the output tensor is the same as sampled_one_hot, i.e., (batch_size, seq_len, K).
    """
    batch_size, seq_len, K = sampled_one_hot.shape
    accuracy = accuracy.view(-1, 1, 1)  # allows for broadcasting over batches
    mean = accuracy * (K * sampled_one_hot - 1)
    variance = accuracy * K
    epsilon = torch.normal(0, 1, sampled_one_hot.shape, device=sampled_one_hot.device)
    return mean + variance * epsilon  # I know the name `variance` suggests it should be squared, but this works just fine

def bayesian_update(y: Tensor, model_input: Tensor) -> Tensor:
    """
    Args:
        y: Noisy version of the sampled one-hot tensor of shape (batch_size, seq_len, K).
        model_input: Input to the model of shape (batch_size, seq_len, K).
    Returns:
        Resulting tensor after applying Bayesian update to the model input based on the noisy output y.
    """
    res = torch.exp(y) * model_input
    return res / torch.sum(res, dim=-1, keepdim=True)  # normalize to ensure it sums to 1 across the last dimension

def bayesian_inference(model_input: Tensor, model_output_logits: Tensor, i: Tensor, n: Tensor, beta_1: Tensor) -> Tensor:
    """
    Args:
        model_input: Input to the model of shape (batch_size, seq_len, K).
        model_output_logits: Model output logits of shape (batch_size, seq_len, K).
        i: Current iteration number of shape (batch_size,).
        n: Total number of iterations of shape (batch_size,).
        beta_1: Maximum possible accuracy (reached when t=1) of shape (batch_size,).
    Returns:
        Resulting tensor after performing Bayesian inference.
    """
    acc = accuracy(i, n, beta_1)
    sampled = sample_model_output(model_output_logits)
    noisy_y = y(sampled, acc)
    return bayesian_update(noisy_y, model_input)