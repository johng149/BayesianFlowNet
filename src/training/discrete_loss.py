from math import log

import einops
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
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


def divergence_loss(
    x: Tensor, schedule: LearnableBetaScheduleNI, folds: int, samples: int = 64
) -> Tensor:
    """
    Divergence loss is a regularization term that is intended to help train
    the learnable schedule. As noted in the paper, at timestep `t = 1.0`, the
    input distribution should approach the ground truth `x`. Minimizing the loss
    tends to drive the model towards a degenerate solution.

    As such, we must penalize the schedule to ensure that its `beta` output at
    `t = 1.0` is large enough such that, when the input distribution is perturbed
    as a function of `beta`, it is approximately equal to the ground truth

    For more information, see https://arxiv.org/pdf/2308.07037 equation 69

    Looking at section 6.8, we see that we actually want the entropy to decrease
    linearly with time. At `t=0.0` we have maximum entropy (ln(K)) while at
    `t=1.0` we have minimum entropy (0). Note that due to Jensen's inequality,
    we will need to use Monte Carlo sampling to estimate the entropy of the
    distribution

    Args:
        x: Ground truth tensor of shape (batch_folds, seq_len, K).
        schedule: Learnable beta schedule.
        folds: Number of folds used for loss variance estimation.
        samples: Number of Monte Carlo samples to use for entropy estimation.
    Returns:
        Divergence loss tensor.
    """

    # batch_folds is batch_size * folds, `batch_size` referes to the unique
    # x samples while `folds` refers to the different times that were sampled
    # for each unique `x`. To get an entropy estimate, we'll need to repeat
    # each of the folds time samples
    batch_folds, seq_len, K = x.shape
    assert (
        batch_folds % folds == 0
    ), f"batch_folds {batch_folds} must be divisible by folds {folds}"
    batch_size = batch_folds // folds

    # for each `x`, sample `folds` timesteps
    t = torch.rand(batch_folds, device=x.device)

    beta_t = schedule.forward(t, K)

    beta_t = einops.repeat(
        beta_t,
        "(batch_size folds) -> (batch_size folds samples)",
        batch_size=batch_size,
        folds=folds,
        samples=samples,
    )

    x = einops.repeat(
        x,
        "(batch_size folds) seq_len K -> (batch_size folds samples) seq_len K",
        batch_size=batch_size,
        folds=folds,
        samples=samples,
    )

    logits, eps = y_distribution(beta_t, K, x)
    torch.save(eps, "debug_div_eps.pt")
    cat = Categorical(logits=logits)

    entropy = cat.entropy()  # should be shape (batch_size * folds * samples, seq_len)

    expected_entropy = einops.reduce(
        entropy,
        "(batch_size folds samples) seq_len -> (batch_size folds)",
        "mean",
        batch_size=batch_size,
        folds=folds,
        samples=samples,
    )

    target_entropy = log(K) * (1 - t)

    div = F.mse_loss(expected_entropy, target_entropy, reduction="mean")
    return div
