import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F

def dis_t(i: Tensor, n: Tensor, minimum: float=1e-6):
  assert torch.all(i <= n), "i must be less than or equal to n"
  assert torch.all(n > 0), "n must be at least 1"
  assert torch.all(i > 0), "i must be at least 1"
  return torch.clamp((i - 1) / n, min=minimum)

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

def bayesian_update(y: Tensor, model_input: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Args:
        y: Noisy version of the sampled one-hot tensor of shape (batch_size, seq_len, K).
        model_input: Input to the model of shape (batch_size, seq_len, K).
    Returns:
        Resulting tensor after applying Bayesian update to the model input based on the noisy output y.
    """
    log_model_input = torch.log(model_input + eps) # add eps to avoid log(0)
    z = y + log_model_input
    log_new_probs = F.log_softmax(z, dim=-1)
    res = torch.exp(log_new_probs)
    return res

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

    # we need to do `(model_input + 1) / 2` to convert the input from [-1, 1] to [0, 1]
    # if we did not, the parameters of the distribution wouldn't produce a valid probability distribution
    # and so the `bayesian_update` may end up with NaN values
    # however, upon returning, we need to convert it back to [-1, 1] as that is what the model is trained on
    return bayesian_update(noisy_y, (model_input + 1) / 2) * 2 - 1