import torch
from torch import Tensor

from src.schedule.base import ScheduleOutput


def loss(
    scheduler_output: ScheduleOutput,
    target: Tensor,
    model_output_logits: Tensor,
    mask: Tensor,
    auxiliary_loss: Tensor,
    aux_weight: float = 0.03,
) -> Tensor:
    """
    Compute the loss given the scheduler output, target tensor, and model output logits.

    Args:
        - scheduler_output (ScheduleOutput) Output from the scheduler containing beta and alpha tensors
            for the current timestep. The alpha term should be of shape (batch_size,).
        - target (Tensor) Target tensor of shape (batch_size, seq_len, K)
        - model_output_logits (Tensor) Model output logits of shape (batch_size, seq_len, K)
        - mask (Tensor) Mask tensor of shape (batch_size, seq_len). For every sequence position where mask is False,
          the loss will not be computed
        - auxiliary_loss (Tensor) Should be a scalar tensor given by the chunker
    Returns:
        Tensor: Computed loss.
    """
    batch_size, seq_len, K = target.shape
    alpha = scheduler_output["alpha"]
    assert (
        alpha.ndim == 1 or alpha.ndim == 2
    ), "Alpha tensor must be 1D or 2D."  # 1d is for normal, 2d is for packed
    assert alpha.shape[0] == batch_size or (
        alpha.ndim == 2 and alpha.shape[0] == batch_size and alpha.shape[1] == seq_len
    ), "Alpha tensor batch size must match target batch size."
    assert model_output_logits.shape == target.shape, (
        "Model output logits shape must match target shape."
        f" Got {model_output_logits.shape} and {target.shape}."
    )

    model_output = torch.softmax(model_output_logits, dim=-1)
    mse = (target - model_output) ** 2  # shape is (batch_size, seq_len, K)
    mse = mse * mask.unsqueeze(-1)  # apply mask
    mse = mse * alpha.unsqueeze(
        -1
    )  # scale by time dependent accuracy (penalize more loss when t -> 1)
    seq_normalized_loss = torch.sum(mse, dim=(-2, -1)) * 0.5 * K / seq_len
    l_infty_loss = torch.mean(seq_normalized_loss)

    return l_infty_loss + aux_weight * auxiliary_loss
