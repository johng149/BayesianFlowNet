from typing import Callable, List, NewType, Tuple

import torch
from einops import rearrange
from torch import Tensor
from torch.distributions import Categorical as TorchCategorical
from torch.nn import Module
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.common.data_prep import accuracy, dis_t, sample_model_output, theta, y
from src.schedule.base import Scheduler
from src.tokenizers.base import TokenizerBase


def generative_prior(
    batch_size: int,
    seq_len: int,
    K: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    stochastic: float = 1e-3,
) -> Tensor:
    """
    Create an uninformative prior tensor used to start the inference process.
    Args:
        - batch_size (int) Batch size.
        - seq_len (int) Sequence length.
        - K (int) Number of categories.
        - device (torch.device) Device to create the tensor on.
        - dtype (torch.dtype) Data type of the tensor.
        - stochastic (float) Magnitude of noise to add to the uniform distribution. 0 or negative means no noise.
    Returns:
        Tensor: Prior tensor of shape (batch_size, seq_len, K) with uniform probabilities.
    """
    uniform = torch.full(
        (batch_size, seq_len, K),
        fill_value=1.0 / K,
        device=device,
        dtype=dtype,
    )
    if stochastic and stochastic > 0.0:
        noise = (
            torch.rand((batch_size, seq_len, K), device=device, dtype=dtype)
            * stochastic
        )
        uniform = uniform + noise
        uniform = uniform / uniform.sum(dim=-1, keepdim=True)
    return uniform


def stratified(logits, num_samples, is_probs: bool = False):
    """
    We take in logits or probabilities and perform stratified sampling on
    each batch dimension. By stratified sampling, we ensure that we cover the
    distribution more evenly compared to naive sampling.

    Args:
        logits: (Batch, Seq, Vocab) - Logits or probabilities over vocabulary.
        num_samples: Number of samples to draw per position.
        is_probs: If True, logits are treated as probabilities.
    Returns:
        samples: (Batch x num_samples, Seq) - Sampled token indices.
    """
    if not is_probs:
        probs = torch.softmax(logits, dim=-1)
    else:
        probs = logits

    # cdf shape: (Batch, Seq, Vocab)
    cdf = probs.cumsum(dim=-1)

    # Get dynamic shape
    *batch_dims, seq_len, vocab_size = cdf.shape

    # Generate stratified noise with matching batch dimensions
    # Shape: (Batch, Seq, num_samples)
    u = torch.rand(*batch_dims, seq_len, num_samples, device=logits.device)
    strata = (torch.arange(num_samples, device=logits.device) + u) / num_samples

    # searchsorted requires input (strata) to match the prefix dimensions of boundaries (cdf)
    # cdf: (Batch, Seq, Vocab)
    # strata: (Batch, Seq, num_samples)
    samples = torch.searchsorted(cdf, strata)

    # Clamp to ensure indices are within valid range
    samples = samples.clamp(max=vocab_size - 1)
    return rearrange(samples, "batch seq num_samples -> (batch num_samples) seq")


def select_top_k_branches(
    input_tensor: Tensor, scores: Tensor, doc_ids: Tensor, k: int
) -> Tensor:
    """
    Args:
        input_tensor: (N, total_seq, classes)
        scores: (N, num_docs) - Lower is better
        doc_ids: (1, total_seq)
        k: Number of top branches to select

    Returns:
        refined: (K, total_seq, classes)
    """
    N, total_seq, classes = input_tensor.shape

    # 1. Get indices of the best K branches for each document.
    # We want smallest scores, so we use largest=False.
    # topk_indices shape: (K, num_docs)
    _, topk_indices = torch.topk(scores, k, dim=0, largest=False)

    # 2. Expand doc_ids to match the sequence length.
    # doc_ids is (1, total_seq), containing values 0 to num_docs-1.
    # We use these values to index into topk_indices.

    # topk_indices is (K, num_docs). We want to select columns based on doc_ids.
    # Result shape: (K, total_seq)
    # We squeeze doc_ids to (total_seq,) for indexing.
    expanded_indices = topk_indices[:, doc_ids.squeeze(0)]

    # 3. Prepare indices for gathering.
    # input_tensor is (N, total_seq, classes).
    # We need to gather along dim 0 (the N dimension).
    # expanded_indices is (K, total_seq). We need to expand it to (K, total_seq, classes).
    gather_indices = expanded_indices.unsqueeze(-1).expand(-1, -1, classes)

    # 4. Gather the data.
    # torch.gather requires the input and index to have the same number of dimensions.
    # We gather from input_tensor along dim 0.
    refined = torch.gather(input_tensor, 0, gather_indices)

    return refined


def ode_euler(
    model: Module,
    t: Tensor,
    x: Tensor,
    mask: Tensor,
    doc_ids: Tensor,
    model_prompt: Tensor,
) -> Tuple[Tensor, Tensor]:
    x = F.softmax(x, -1)
    x = torch.where(mask.unsqueeze(-1), x, model_prompt)
    logits, _, _, energy = model(x, t, mask, doc_ids)
    return logits, energy


def inference(
    model: Module,
    scheduler: Scheduler,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    K: int,
    mask: Tensor,
    masked_input: Tensor,
    doc_ids: Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    tk: TokenizerBase | None = None,
    algorithm="sde_euler",
    energy_tracker: List[float] | None = None,
    k_search: int = 1,  # if >1, use verifier search scaling
):
    with torch.no_grad():
        xt = generative_prior(batch_size, seq_len, K, device, dtype)
        total_iterations = torch.ones(batch_size, seq_len, device=device) * num_steps
        energy = None
        initial_ebm_energy = None
        for step in range(1, num_steps + 1):
            current_iteration = torch.ones_like(total_iterations) * step
            curr_t = dis_t(current_iteration, total_iterations)
            xt, energy = ode_euler(model, curr_t, xt, mask, doc_ids, masked_input)
            if initial_ebm_energy is None:
                initial_ebm_energy = energy
            if energy_tracker is not None:
                energy_tracker.append(energy.mean().item())
        assert energy is not None and initial_ebm_energy is not None
        return xt, energy, initial_ebm_energy
