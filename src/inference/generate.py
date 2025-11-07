from typing import Callable, NewType

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module
from torch.nn import functional as F

from src.datasets.dataset_helper import theta
from src.schedule.base import Scheduler

Accuracy = NewType("Accuracy", Tensor)


def accuracy(
    i: Tensor, n: Tensor, scheduler: Scheduler, min_t: float = 1e-6
) -> Accuracy:
    """
    Compute accuracy for inference step `i` out of `n` total steps using the provided scheduler.
    The accuracy is defined as the difference between the beta value at step `i` and step `i-1`.

    Args:
        - i (Tensor) Current inference step of shape (batch_size,)
        - n (Tensor) Total number of inference steps of shape (batch_size,)
        - scheduler (Scheduler) Scheduler to compute the accuracy.
        - min_t (float) Minimum timestep value to avoid numerical issues.
    Returns:
        Tensor: Computed accuracy.
    """
    assert i.shape == n.shape, "Shapes of i and n must match."
    assert i.ndim == 1, "Input tensors must be 1D."
    assert torch.all(i > 0) and torch.all(
        n > 0
    ), "Input tensors must contain positive values."
    assert torch.all(
        i <= n
    ), "Current step i must be less than or equal to total steps n."

    t_i = torch.clamp(i / n, min=min_t, max=1.0)
    t_im1 = torch.clamp((i - 1) / n, min=min_t, max=1.0)
    beta_i = scheduler(t_i)["beta"]
    beta_im1 = scheduler(t_im1)["beta"]
    accuracy = beta_i - beta_im1
    return Accuracy(accuracy)


def dis_t(i: Tensor, n: Tensor, min_t: float = 1e-6) -> Tensor:
    """
    The current timestep `t` to be passed into the model during inference given the current step `i`
    and total steps `n`.

    Args:
        - i (Tensor) Current inference step of shape (batch_size,)
        - n (Tensor) Total number of inference steps of shape (batch_size,)
        - min_t (float) Minimum timestep value to avoid numerical issues.
    Returns:
        Tensor: Current timestep `t` of shape (batch_size,).
    """
    assert i.shape == n.shape, "Shapes of i and n must match."
    assert i.ndim == 1, "Input tensors must be 1D."
    assert torch.all(i > 0) and torch.all(
        n > 0
    ), "Input tensors must contain positive values."
    assert torch.all(
        i <= n
    ), "Current step i must be less than or equal to total steps n."

    return torch.clamp((i - 1) / n, min=min_t, max=1.0)


def sample_model_output(model_output_logits: Tensor) -> Tensor:
    """
    Args:
        - model_output_logits (Tensor) Model output logits of shape (batch_size, seq_len, K)
    Returns:
        Tensor: Sampled model output based on the logits.
    """
    batch_size, seq_len, K = model_output_logits.shape
    dist = Categorical(logits=model_output_logits)
    samples = dist.sample()
    return F.one_hot(samples, K)


def y(sampled_one_hot: Tensor, accuracy_t: Accuracy) -> Tensor:
    """
    Args:
        - sampled_one_hot (Tensor) Sampled output described by model logits that has been one-hot encoded,
                        of shape (batch_size, seq_len, K).
        - accuracy (Accuracy) Accuracy at the current iteration of shape (batch_size,).
    Returns:
        Tensor: Noisy version of the sampled one-hot tensor with the amount of noise controlled by accuracy.
        The shape of the output tensor is the same as sampled_one_hot, i.e., (batch_size, seq_len, K).
    """
    batch_size, seq_len, K = sampled_one_hot.shape
    accuracy = accuracy_t.view(-1, 1, 1)  # allows for broadcasting over batches
    mean = accuracy * (K * sampled_one_hot - 1)
    variance = accuracy * K
    standard_deviation = torch.sqrt(variance)
    epsilon = torch.normal(0, 1, sampled_one_hot.shape, device=sampled_one_hot.device)
    return mean + standard_deviation * epsilon


def bayesian_update(y: Tensor, model_input: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Args:
        - y (Tensor) Noisy version of the sampled one-hot tensor of shape (batch_size, seq_len, K).
        - model_input (Tensor) Input to the model of shape (batch_size, seq_len, K).
    Returns:
        Tensor: Resulting tensor after applying Bayesian update to the model input based on the noisy output y.
    """
    log_model_input = torch.log(model_input + eps)  # add eps to avoid log(0)
    z = y + log_model_input
    log_new_probs = F.log_softmax(z, dim=-1)
    res = torch.exp(log_new_probs)
    return res


def bayesian_inference(
    model_input: Tensor,
    model_output_logits: Tensor,
    i: Tensor,
    n: Tensor,
    scheduler: Scheduler,
) -> Tensor:
    """
    Args:
        - model_input (Tensor) Input to the model of shape (batch_size, seq_len, K).
        - model_output_logits (Tensor) Model output logits of shape (batch_size, seq_len, K).
        - i (Tensor) Current iteration number of shape (batch_size,).
        - n (Tensor) Total number of iterations of shape (batch_size,).
        - beta_1 (Tensor) Maximum possible accuracy (reached when t=1) of shape (batch_size,).
    Returns:
        Tensor: Resulting tensor after performing Bayesian inference.
    """
    acc = accuracy(i, n, scheduler)
    sampled = sample_model_output(model_output_logits)
    noisy_y = y(sampled, acc)

    # we need to do `(model_input + 1) / 2` to convert the input from [-1, 1] to [0, 1]
    # if we did not, the parameters of the distribution wouldn't produce a valid probability distribution
    # and so the `bayesian_update` may end up with NaN values
    # however, upon returning, we need to convert it back to [-1, 1] as that is what the model is trained on
    return bayesian_update(noisy_y, (model_input + 1) / 2) * 2 - 1


def generative_prior(
    batch_size: int,
    seq_len: int,
    K: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Create an uninformative prior tensor used to start the inference process.
    Args:
        - batch_size (int) Batch size.
        - seq_len (int) Sequence length.
        - K (int) Number of categories.
        - device (torch.device) Device to create the tensor on.
        - dtype (torch.dtype) Data type of the tensor.
    Returns:
        Tensor: Prior tensor of shape (batch_size, seq_len, K) with uniform probabilities.
    """
    uniform = torch.full(
        (batch_size, seq_len, K),
        fill_value=1.0 / K,
        device=device,
        dtype=dtype,
    )
    return theta(uniform)


def inference(
    model: Module,
    scheduler: Scheduler,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    K: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    conditioning_callback: Callable[[Tensor], Tensor] | None = None,
):
    total_iterations = torch.ones(batch_size, device=device) * num_steps
    current = generative_prior(
        batch_size=batch_size,
        seq_len=seq_len,
        K=K,
        device=device,
        dtype=dtype,
    )

    for i in range(1, num_steps + 1):
        current_iteration = torch.ones_like(total_iterations) * i
        curr_t = dis_t(current_iteration, total_iterations)
        output = model(current, curr_t)
        if conditioning_callback is not None:
            output = conditioning_callback(output)
        current = bayesian_inference(
            model_input=current,
            model_output_logits=output,
            i=current_iteration,
            n=total_iterations,
            scheduler=scheduler,
        )

    return current
