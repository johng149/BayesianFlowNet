from typing import Callable, NewType

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module
from torch.nn import functional as F

from src.common.data_prep_types import Accuracy, Beta
from src.schedule.base import Scheduler


def sample_t(batch_size: int, min_t=1e-6) -> Tensor:
    return torch.clamp(torch.FloatTensor(batch_size).uniform_(0, 1), min=min_t)


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


def beta(t: Tensor, scheduler: Scheduler) -> Beta:
    """
    Compute beta value at timestep `t` using the provided scheduler.

    Args:
        - t (Tensor) Timestep tensor of shape (batch_size,).
        - scheduler (Scheduler) Scheduler to compute the beta value.
    Returns:
        Tensor: Computed beta value.
    """
    beta_t = scheduler(t)["beta"]
    return Beta(beta_t)


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


def y(sampled_one_hot: Tensor, accuracy_t: Accuracy | Beta) -> Tensor:
    """
    Args:
        - sampled_one_hot (Tensor) Sampled output described by model logits that has been one-hot encoded,
                        of shape (batch_size, seq_len, K).
        - accuracy (Accuracy | Beta) Accuracy or beta value at the current timestep of shape (batch_size,).
            In practice, the two are used in the exact same way, but the types are different to differentiate
            between their use cases. Accuracy is used during Bayesian inference, while Beta is used during
            training, as the `y` output will be used as the model input once it is passed through `theta(y)`
    Returns:
        Tensor: Noisy version of the sampled one-hot tensor with the amount of noise controlled by accuracy.
        The shape of the output tensor is the same as sampled_one_hot, i.e., (batch_size, seq_len, K).
    """
    batch_size, seq_len, K = sampled_one_hot.shape
    if accuracy_t.ndim == 1:
        accuracy = accuracy_t.view(-1, 1, 1)  # allows for broadcasting over batches
    elif accuracy_t.ndim == 2:
        accuracy = accuracy_t.unsqueeze(-1)
    else:
        accuracy = accuracy_t
    mean = accuracy * (K * sampled_one_hot - 1)
    variance = accuracy * K
    standard_deviation = torch.sqrt(variance)
    epsilon = torch.normal(0, 1, sampled_one_hot.shape, device=sampled_one_hot.device)
    return mean + standard_deviation * epsilon


def theta(y: Tensor):
    """
    Applies softmax to input tensor y and scales the result to the range [-1, 1], as recommended by the
    paper when feeding inputs to the model.

    Args:
        y: Tensor of shape (batch_size, seq_len, K) representing the noisy version of kron_x.
    Returns:
        Tensor representing the scaled softmax of y, which is the input to the model.
    """
    assert y.ndim == 3, "y should be a 3D tensor of shape (batch_size, seq_len, K)"
    theta = F.softmax(y, dim=-1)
    # theta = 2 * theta - 1  # scale to [-1, 1]
    return theta
