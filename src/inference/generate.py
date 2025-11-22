from typing import Callable, NewType

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module
from torch.nn import functional as F

from src.common.data_prep import accuracy, dis_t, sample_model_output, theta, y
from src.schedule.base import Scheduler
from src.tokenizers.base import TokenizerBase


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
    return bayesian_update(noisy_y, model_input)


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
    encoder_model_input: Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    conditioning_callback: Callable[[Tensor], Tensor] | None = None,
    tk: TokenizerBase | None = None,
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
        if tk is not None and (i % (num_steps // 10) == 0 or i == num_steps):
            print(f"Step {i}: {tk.decode(torch.argmax(current, dim=-1)[0].cpu())}")
        current_iteration = torch.ones_like(total_iterations) * i
        curr_t = dis_t(current_iteration, total_iterations)
        output = model(current, curr_t, encoder_model_input)
        if conditioning_callback is not None:
            output = conditioning_callback(output)
        current = bayesian_inference(
            model_input=current,
            model_output_logits=output,
            i=current_iteration,
            n=total_iterations,
            scheduler=scheduler,
        )
    final_t = torch.ones_like(total_iterations)
    final_output_logits = model(current, final_t)
    return final_output_logits
