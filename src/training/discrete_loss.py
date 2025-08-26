import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.common.types import DiscreteFormattedLoss
from src.datasets.discrete_helper import y_distribution
from src.nn.layers.learnable_schedule import LearnableBetaScheduleNI


def format_loss(
    alpha: Tensor,
    target: Tensor,
    model_output_probs: Tensor | None = None,
    model_output_logits: Tensor | None = None,
    folds: int = 1,
) -> DiscreteFormattedLoss:
    """
    Args:
        alpha: Tensor of shape (batch_size * folds,), it is the derivative of the beta scheduling function at time `t`
            for each sample in the batch
        target: Target tensor of shape (batch_size * folds, seq_len, K).
        model_output_probs: Model output probabilities of shape (batch_size * folds, seq_len, K). If None, model_output_logits must be provided.
        model_output_logits: Model output logits of shape (batch_size * folds, seq_len, K). If None, model_output_probs must be provided.
        folds: Number of folds used for loss variance estimation.
    Returns:
        Formatted loss tensor of shape (batch_size, folds)
    """

    batch_folds, seq_len, K = target.shape
    model_output: Tensor
    if model_output_probs is not None:
        model_output = model_output_probs
    else:
        assert (
            model_output_logits is not None
        ), "Must provide either model_output_probs or model_output_logits"
        model_output = torch.softmax(model_output_logits, dim=-1)

    # seq_norm_loss should have shape (batch_size * folds)
    seq_norm_loss = (
        torch.sum((target - model_output) ** 2, dim=(-2, -1))
        * 0.5
        * K
        * alpha
        / seq_len
    )

    return DiscreteFormattedLoss(seq_norm_loss.view(-1, folds))


def loss(
    formatted_loss: DiscreteFormattedLoss,
) -> Tensor:
    return torch.mean(formatted_loss)


def variance_loss(
    formatted_loss: DiscreteFormattedLoss,
) -> Tensor:
    """
    Variance loss is a regularization term that is intended to help train the
    learnable schedule. As noted in the paper, the purpose of the beta schedule
    is to make the denoising problem equally difficult at all timesteps.

    One way to describe that is to guide the model towards having the same loss
    (high or low) at various sampled timesteps for the same input samples.
    This means, to minimize the variance of the loss across timesteps per
    given sample.

    For more information see https://arxiv.org/pdf/2308.07037 Figure 14
    """
    # note that while I call this `variance_loss` it actually calculates the
    # standard deviation. I've tried both, and both seem to work. Ultimately
    # settled on standard deviation though
    return torch.mean(torch.std(formatted_loss, dim=-1))


def divergence_loss(x: Tensor, schedule: LearnableBetaScheduleNI) -> Tensor:
    """
    Divergence loss is a regularization term that is intended to help train
    the learnable schedule. As noted in the paper, at timestep `t = 1.0`, the
    input distribution should approach the ground truth `x`. Minimizing the loss
    tends to drive the model towards a degenerate solution.

    As such, we must penalize the schedule to ensure that its `beta` output at
    `t = 1.0` is large enough such that, when the input distribution is perturbed
    as a function of `beta`, it is approximately equal to the ground truth

    For more information, see https://arxiv.org/pdf/2308.07037 equation 69

    Args:
        x: Ground truth tensor of shape (batch_folds, seq_len, K).
        schedule: Learnable beta schedule.
    Returns:
        Divergence loss tensor.
    """
    batch_folds, seq_len, K = x.shape  # batch_folds is batch_size * folds

    # first, we find what the schedule thinks beta_1 should be
    t_1 = torch.ones(batch_folds, device=x.device)
    beta_1 = schedule(t_1, K)

    # next, we use that to create the perturbed input distribution
    beta_1_dist = y_distribution(beta_1, K, x)  # should be logits
    beta_1_probs = F.log_softmax(beta_1_dist, dim=-1)

    # now we can compute the divergence loss
    div_loss = F.kl_div(beta_1_probs, x, reduction="batchmean", log_target=False)

    return div_loss
