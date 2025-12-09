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
) -> Callable[[List[DatasetOutput]], CollateOutput]:
    """
    Resulting collate function encodes input into one-hot vectors assuming classes equal to vocab_size,
    and then adds noise according to the scheduler before transforming the noisy vectors using theta function

    Args:
        - scheduler (Scheduler): Scheduler used to determine the amount of noise to add.
        - vocab_size (int): Number of classes for one-hot encoding.
        - min_mask_ratio (float): Minimum masking ratio for input sequences.
        - max_mask_ratio (float): Maximum masking ratio for input sequences.
        - mean_span_length (float): Mean span length for masking.
        - contrastive_corruption_prob_base (float): Base probability for corruption in contrastive input.
            The probability increases with time linearly up to contrastive_corruption_prob_max.
        - contrastive_corruption_prob_max (float): Maximum probability for corruption in contrastive input
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
        model_input = theta(y_dist)

        contrastive_y_dist = y(contrastive_x, beta)
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
