from typing import Callable, List, NewType, Tuple

import torch
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
    return uniform


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
