from typing import Callable, List, TypedDict

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
        "decoder_model_input": Tensor,
        "encoder_model_input": Tensor,
        "scheduler_output": ScheduleOutput,
    },
)

from abc import ABC


class BFNDataset(ABC):
    def __getitem__(self, index: int) -> DatasetOutput:
        raise NotImplementedError


def make_collate_fn(
    scheduler: Scheduler, vocab_size: int, mask_idx: int
) -> Callable[[List[DatasetOutput]], CollateOutput]:
    """
    Resulting collate function encodes input into one-hot vectors assuming classes equal to vocab_size,
    and then adds noise according to the scheduler before transforming the noisy vectors using theta function

    Args:
        - scheduler (Scheduler): Scheduler used to determine the amount of noise to add.
        - vocab_size (int): Number of classes for one-hot encoding.
        - mask_idx (int): Index of the mask token in the vocabulary.

    Returns:
        Collate function that can be used in a DataLoader.
    """

    def collate_fn(batch: List[DatasetOutput]) -> CollateOutput:
        x = [item["x"] for item in batch]
        min_length = min(seq.shape[0] for seq in x)
        assert min_length > 1, "Sequences must have length greater than 1."
        if min_length % 2 != 0:
            min_length -= 1  # make even length
        x = [seq[:min_length] for seq in x]
        x = torch.stack(x, dim=0)  # batch_size x seq_len

        first_half_x = x[:, : min_length // 2]  # batch_size x (min_length/2)
        second_half_x = x[:, min_length // 2 :]  # batch_size x (min_length/2)

        second_half_x = F.one_hot(
            second_half_x, num_classes=vocab_size
        )  # batch_size x (min_length/2) x K

        t = torch.cat([item["t"] for item in batch], dim=0)  # batch_size,
        scheduler_output = scheduler(t)
        beta = scheduler_output["beta"]  # batch_size,
        y_dist = y(second_half_x, beta)
        model_input = theta(y_dist)

        return {
            "ground_truth": second_half_x,
            "t": t,
            "decoder_model_input": model_input,
            "encoder_model_input": first_half_x,
            "scheduler_output": scheduler_output,
        }

    return collate_fn
