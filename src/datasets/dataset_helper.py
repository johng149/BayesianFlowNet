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
    Returns:
        Collate function that can be used in a DataLoader.
    """

    def collate_fn(batch: List[DatasetOutput]) -> CollateOutput:
        x = [item["x"] for item in batch]
        min_length = min(seq.shape[0] for seq in x)
        x = [F.one_hot(seq[:min_length], num_classes=vocab_size) for seq in x]
        x = torch.stack(x, dim=0)  # batch_size x seq_len x K

        masks = []
        for _ in range(x.shape[0]):
            r = np.random.uniform(min_mask_ratio, max_mask_ratio)
            m = generate_span_mask(min_length, r, mean_span_length)
            masks.append(m)
        mask = torch.stack(masks, dim=0)  # batch_size x seq_len

        t = torch.cat([item["t"] for item in batch], dim=0)  # batch_size,
        scheduler_output = scheduler(t)
        beta = scheduler_output["beta"]  # batch_size,
        y_dist = y(x, beta)
        model_input = theta(y_dist)

        # for each batch, for each sequence position, use `model_input` if mask is True else use `x`
        model_input = torch.where(mask.unsqueeze(-1), model_input, x)

        return {
            "ground_truth": x,
            "t": t,
            "model_input": model_input,
            "mask": mask,
            "scheduler_output": scheduler_output,
        }

    return collate_fn
