import random
from typing import Callable, List, TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from src.common.data_prep import theta, y
from src.schedule.base import ScheduleOutput, Scheduler

DatasetOutput = TypedDict("DatasetOutput", {"x": Tensor, "t": Tensor})
CollateOutput = TypedDict(
    "CollateOutput",
    {
        "ground_truth": Tensor,
        "t": Tensor,
        "model_input": Tensor,
        "mask": Tensor,
        "scheduler_output": ScheduleOutput,
        "document_id": Tensor,
        "contrastive_input": Tensor,
    },
)

from abc import ABC


class BFNDataset(ABC):
    def __getitem__(self, index: int) -> DatasetOutput:
        raise NotImplementedError


import torch
from einops import rearrange


def stratified_window(
    logits,
    num_samples: int,
    left: float = 0.0,
    right: float = 1.0,
    is_probs: bool = False,
    temperature: float = 1.0,
):
    """
    Samples from a specified window [left, right] of stratified probabilities.
    If left=0.0 and right=1.0, this is equal to normal sampling with given
    temperature. While left=0.0 and right=0.0 is greedy sampling.

    The temperature parameter will only be used if the input is logits.

    Args:
        logits: (Batch, Seq, Vocab)
        num_samples: How many samples to draw from the specified window.
        left: The start of the probability range (0.0 = most likely).
        right: The end of the probability range (1.0 = least likely).
    """
    left = min(max(left, 0.0), 1.0)
    right = min(max(right, 0.0), 1.0)
    assert left <= right, "Left boundary must be less than or equal to right boundary."

    if not is_probs:
        probs = torch.softmax(logits / temperature, dim=-1)
    else:
        probs = logits

    # 1. Sort probabilities descending so that 'left=0' is the most likely token
    probs_sorted, indices_sorted = torch.sort(probs, dim=-1, descending=True)

    # 2. Compute CDF on the sorted tokens
    cdf = probs_sorted.cumsum(dim=-1)

    # 3. Define the window width
    window_width = right - left

    # 4. Generate stratified noise within the [left, right] range
    # If num_samples=1, left=0.1, right=0.2:
    # u will be in [0, 1], then scaled to [0, 0.1], then shifted to [0.1, 0.2]
    *batch_dims, seq_len, vocab_size = cdf.shape
    u = torch.rand(*batch_dims, seq_len, num_samples, device=logits.device)

    # This formula splits the window [left, right] into 'num_samples' equal strata
    # and picks one random point from each.
    strata_steps = torch.arange(num_samples, device=logits.device)
    strata = left + (strata_steps + u) / num_samples * window_width

    # 5. Find the indices in the sorted distribution
    # searchsorted finds the first index where cdf >= strata
    sampled_sorted_indices = torch.searchsorted(cdf, strata)
    sampled_sorted_indices = sampled_sorted_indices.clamp(max=vocab_size - 1)

    # 6. Map back to original vocabulary indices
    # indices_sorted is (Batch, Seq, Vocab), we need to gather from it
    # We need to expand indices_sorted to match the num_samples dimension
    # or use gather carefully.

    # Flatten batch/seq for easier gathering
    flat_indices_sorted = rearrange(indices_sorted, "b s v -> (b s) v")
    flat_sampled_indices = rearrange(sampled_sorted_indices, "b s n -> (b s) n")

    # Gather the actual token IDs
    samples = torch.gather(flat_indices_sorted, dim=1, index=flat_sampled_indices)

    return rearrange(samples, "(b s) n -> (b n) s", b=batch_dims[0])


def generate_span_mask(
    seq_len: int, mask_ratio: float, mean_span_length: float
) -> Tensor:
    """
    Generates a boolean mask using Poisson span corruption logic.
    True (1) = Masked/Target (Noised)
    False (0) = Visible/Context (Clean)
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    num_tokens_to_mask = int(seq_len * mask_ratio)

    if num_tokens_to_mask == 0:
        return mask

    num_masked = 0
    while num_masked < num_tokens_to_mask:
        # Sample span length from Poisson (clamped to at least 1)
        # We use numpy for poisson as torch.poisson implies floats
        span_len = max(1, np.random.poisson(mean_span_length))

        # Don't exceed remaining budget significantly
        # (Optional: remove this clamp if you prefer adhering strictly to span dynamics over ratio)
        span_len = min(span_len, num_tokens_to_mask - num_masked + 2)

        # Pick a random start index
        # We ensure the span fits within the sequence
        if seq_len - span_len <= 0:
            start_index = 0
            span_len = seq_len  # Mask whole sequence if span is too big
        else:
            assert isinstance(span_len, int)
            start_index = np.random.randint(low=0, high=seq_len - span_len + 1)

        # Apply mask
        # Note: This might overlap with existing masks, which is generally acceptable
        # in span corruption literature (effectively merges spans)
        mask[start_index : start_index + span_len] = True

        # Recalculate count
        num_masked = mask.sum().item()

    return mask


def make_collate_fn(
    scheduler: Scheduler,
    vocab_size: int,
    min_mask_ratio: float = 0.0,
    max_mask_ratio: float = 0.95,
    mean_span_length: float = 3.0,
    contrastive_corruption_prob_base: float = 0.3,
    contrastive_corruption_prob_max: float = 0.9,
    stratified_sampling_prob: float = 0.5,
    stratify_width: float = 0.1,
) -> Callable[[List[DatasetOutput]], CollateOutput]:
    """
    Resulting collate function encodes input into one-hot vectors assuming classes equal to vocab_size,
    and then adds noise according to the scheduler before transforming the noisy vectors using theta function.

    For the stratification sampling process, if it is triggered, we pick a random `left` stratification boundary
    in [0, 1) and then set `right = left + stratify_width`, which will be clamped by the straitifcation sampling
    function to ensure both `left` and `right` are in [0, 1] and `left <= right`.

    Args:
        - scheduler (Scheduler): Scheduler used to determine the amount of noise to add.
        - vocab_size (int): Number of classes for one-hot encoding.
        - min_mask_ratio (float): Minimum masking ratio for input sequences.
        - max_mask_ratio (float): Maximum masking ratio for input sequences.
        - mean_span_length (float): Mean span length for masking.
        - contrastive_corruption_prob_base (float): Base probability for corruption in contrastive input.
            The probability increases with time linearly up to contrastive_corruption_prob_max.
        - contrastive_corruption_prob_max (float): Maximum probability for corruption in contrastive input
        - stratified_sampling_prob (float): Probability of using stratified sampling for normal and contrastive inputs.
        - stratify_width (float): Width of the stratified sampling window.
    Returns:
        Collate function that can be used in a DataLoader.
    """

    def collate_fn(batch: List[DatasetOutput]) -> CollateOutput:
        xs = [item["x"] for item in batch]
        ts = [item["t"] for item in batch]

        # Create document_ids and pack sequences
        doc_ids = []
        packed_x_list = []
        packed_t_list = []
        masks = []

        for i, (x, t) in enumerate(zip(xs, ts)):
            seq_len = x.shape[0]

            # Document ID
            doc_ids.append(torch.full((seq_len,), i, dtype=torch.long))

            # X
            packed_x_list.append(x)

            # T (expand scalar t to seq_len)
            packed_t_list.append(t.repeat(seq_len))

            # Mask
            r = np.random.uniform(min_mask_ratio, max_mask_ratio)
            m = generate_span_mask(seq_len, r, mean_span_length)
            masks.append(m)

        # Concatenate everything
        packed_x_indices = torch.cat(packed_x_list, dim=0)
        packed_doc_ids = torch.cat(doc_ids, dim=0)
        packed_t = torch.cat(packed_t_list, dim=0)
        packed_mask = torch.cat(masks, dim=0)

        # for each position, determine contrastive corruption probability based on time t
        slope = (
            contrastive_corruption_prob_max - contrastive_corruption_prob_base
        ) / 1.0  # recall that t in [0, 1]
        contrastive_corruption_probs = torch.clamp(
            contrastive_corruption_prob_base + slope * packed_t, 0.0, 1.0
        )
        should_corrupt = packed_mask & (
            torch.rand_like(packed_t) < contrastive_corruption_probs
        )  # to corrupt, it needs to be masked and pass the prob check

        random_indices = torch.randint(
            0, vocab_size, packed_x_indices.shape, device=packed_x_indices.device
        )
        collision = random_indices == packed_x_indices
        random_indices[collision] = (random_indices[collision] + 1) % vocab_size
        contrastive_indices = torch.where(
            should_corrupt, random_indices, packed_x_indices
        )

        # One-hot encode x
        packed_x = F.one_hot(packed_x_indices, num_classes=vocab_size)
        contrastive_x = F.one_hot(contrastive_indices, num_classes=vocab_size)

        # Add batch dimension (1, total_len, ...)
        packed_x = packed_x.unsqueeze(0)  # (1, total_len, vocab_size)
        contrastive_x = contrastive_x.unsqueeze(0)  # (1, total_len, vocab_size)
        packed_doc_ids = packed_doc_ids.unsqueeze(0)  # (1, total_len)
        packed_t = packed_t.unsqueeze(0)  # (1, total_len)
        packed_mask = packed_mask.unsqueeze(0)  # (1, total_len)

        # Scheduler and Noise
        scheduler_output = scheduler(packed_t)
        beta = scheduler_output["beta"]  # (1, total_len)

        y_dist = y(packed_x, beta)

        contrastive_y_dist = y(contrastive_x, beta)

        use_stratified = random.random() < stratified_sampling_prob
        # if we are using stratified sampling, we resample `model_input` and `contrastive_input`
        # and then pass it through the y and theta processes again
        if use_stratified:
            stratification_level_left = random.random()  # in [0, 1)
            right = stratification_level_left + stratify_width
            # Resample model_input
            sampled_packed_x = stratified_window(
                logits=y_dist,
                num_samples=1,
                left=stratification_level_left,
                right=right,
                is_probs=False,
            )  # (1, total_len)
            sampled_packed_x = F.one_hot(sampled_packed_x, num_classes=vocab_size)
            y_dist = y(sampled_packed_x, beta)

            # sampled_constrastive_x = stratified_window(
            #     logits=contrastive_y_dist,
            #     num_samples=1,
            #     left=stratification_level_left,
            #     right=right,
            #     is_probs=False,
            # )  # (1, total_len)
            # sampled_constrastive_x = F.one_hot(
            #     sampled_constrastive_x, num_classes=vocab_size
            # )
            # contrastive_y_dist = y(sampled_constrastive_x, beta)

        model_input = theta(y_dist)
        contrastive_input = theta(contrastive_y_dist)

        # for each batch, for each sequence position, use `model_input` if mask is True else use `x`
        model_input = torch.where(packed_mask.unsqueeze(-1), model_input, packed_x)
        contrastive_input = torch.where(
            packed_mask.unsqueeze(-1), contrastive_input, packed_x
        )

        return {
            "ground_truth": packed_x,
            "t": packed_t,
            "model_input": model_input,
            "mask": packed_mask,
            "scheduler_output": scheduler_output,
            "document_id": packed_doc_ids,
            "contrastive_input": contrastive_input,
        }

    return collate_fn
