from math import log

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


# note to self:
# do NOT use target beta_1 value as a loss function for divergence such as
# torch.mean((learned_beta_1 - 20.4054 / K)**2)
# it does NOT work. The reason is because model is greatly incentivized to make
# derivative of beta schedule (alpha) zero since the main loss is scaled by alpha
# thus making learned beta approach 0. Only KL divergence (as programmed below)
# helps mitigate this issue
def divergence_loss(
    x: Tensor, schedule: LearnableBetaScheduleNI, target_kl: float = 1.68
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

    However, excessive penalties can cause the schedule to prefer ever increasing
    beta values, which, in practice, appears to prevent convergence. As such,
    we need a target KL divergence value to discourage the optimizer from
    exploring excessively high beta values.

    Args:
        x: Ground truth tensor of shape (batch_folds, seq_len, K).
        schedule: Learnable beta schedule.
        target_kl: Target KL divergence value.
    Returns:
        Divergence loss tensor.
    """
    batch_folds, seq_len, K = x.shape  # batch_folds is batch_size * folds

    # trying a new version of this based on https://arxiv.org/html/2308.07037v6 section 6.8
    t = torch.rand(batch_folds, device=x.device)
    beta_t = schedule(t, K)
    try:
        beta_t_dist = y_distribution(
            beta_t, K, x, deterministic=True
        )  # should be logits
    except AssertionError as e:
        raise RuntimeError(
            f"Error in divergence_loss with beta_t: {beta_t} at time {t}"
        ) from e
    beta_cat = Categorical(logits=beta_t_dist)

    entropy = beta_cat.entropy()  # should be shape (batch_folds, seq_len)
    assert entropy.shape == (
        batch_folds,
        seq_len,
    ), f"Expected entropy shape {(batch_folds, seq_len)}, got {entropy.shape}"
    seq_expected_entropy = torch.mean(entropy, dim=-1)  # shape (batch_folds,)

    # the entropy of a uniform categorical distribution with K classes is ln(K)
    # the paper says we want the entropy to linearly decrease from ln(K) to 0
    # as t goes from 0 to 1. This means the slope of the decrease is -ln(K)
    # thus the expected entropy of the distribution at time t is:
    # ln(K) - ln(K) * t = ln(K) * (1 - t)
    target_entropy = log(K) * (1 - t)  # shape (batch_folds,)

    divergence = torch.mean((seq_expected_entropy - target_entropy) ** 2)
    return divergence

    # # first, we find what the schedule thinks beta_1 should be
    # t_1 = torch.ones(batch_folds, device=x.device)
    # beta_1 = schedule(t_1, K)

    # # next, we use that to create the perturbed input distribution
    # beta_1_dist = y_distribution(beta_1, K, x, deterministic=True)  # should be logits
    # beta_1_probs = F.log_softmax(beta_1_dist, dim=-1)

    # # now we can compute the divergence loss
    # kl = F.kl_div(beta_1_probs, x, reduction="batchmean", log_target=False)

    # # div_loss = (kl - target_kl) ** 2

    # # # pseudo-huber loss for when kl is greater than or equal to target_kl
    # # # torch.sqrt(1 + (div_loss**2)) - 1

    # # # pseudo-huber-loss for when kl is less than target_kl
    # # # δ = 1.6, δ**2 * (torch.sqrt(1 + (div_loss / δ)**2) - 1)

    # # delta = 1.6 if kl < target_kl else 1.0
    # # return (delta**2) * (torch.sqrt(1 + (div_loss / delta) ** 2) - 1)
    # return kl


def alpha_variance_loss(
    alpha: Tensor, target_variance: float = 1.0, delta: float = 1.0
) -> Tensor:
    alpha_std = torch.std(alpha)
    alpha_std_loss = (alpha_std - target_variance) ** 2

    return (delta**2) * (torch.sqrt(1 + (alpha_std_loss / delta) ** 2) - 1)


def alpha_below_linear_loss(x: Tensor, schedule: LearnableBetaScheduleNI) -> Tensor:
    t_sample = torch.rand(x.shape[0], device=x.device)
    _, alpha_sample = schedule.get_alpha(t_sample, x.shape[-1])
    t_sample, t_sample_i = torch.sort(t_sample)
    alpha_sample = alpha_sample[t_sample_i]

    t_boundaries = torch.tensor([0.0, 1.0], device=x.device)
    _, alpha_boundaries = schedule.get_alpha(t_boundaries, x.shape[-1])

    # create linear best fit line using alpha boundaries
    slope = alpha_boundaries[1] - alpha_boundaries[0]
    linear_alpha = slope * t_sample + alpha_boundaries[0]

    above_line = alpha_sample - linear_alpha
    above_line = F.relu(above_line)

    return torch.mean(above_line)
