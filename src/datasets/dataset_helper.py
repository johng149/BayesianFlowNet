from typing import Callable, List, TypedDict

import torch
from torch import Tensor
from torch.nn import functional as F

from src.common.data_prep import theta, y
from src.schedule.base import Scheduler

DatasetOutput = TypedDict("DatasetOutput", {"x": Tensor, "t": Tensor})
CollateOutput = TypedDict(
    "CollateOutput", {"ground_truth": Tensor, "t": Tensor, "model_input": Tensor}
)

from abc import ABC


class BFNDataset(ABC):
    def __getitem__(self, index: int) -> DatasetOutput:
        raise NotImplementedError


def make_collate_fn(
    scheduler: Scheduler, vocab_size: int
) -> Callable[[List[DatasetOutput]], CollateOutput]:
    """
    Resulting collate function encodes input into one-hot vectors assuming classes equal to vocab_size,
    and then adds noise according to the scheduler before transforming the noisy vectors using theta function

    Args:
        - scheduler (Scheduler): Scheduler used to determine the amount of noise to add.
        - vocab_size (int): Number of classes for one-hot encoding.
    Returns:
        Collate function that can be used in a DataLoader.
    """

    def collate_fn(batch: List[DatasetOutput]) -> CollateOutput:
        x = [item["x"] for item in batch]
        min_length = min(seq.shape[0] for seq in x)
        x = [F.one_hot(seq[:min_length], num_classes=vocab_size) for seq in x]
        x = torch.stack(x, dim=0)  # batch_size x seq_len x K

        t = torch.cat([item["t"] for item in batch], dim=0)  # batch_size,
        scheduler_output = scheduler(t)
        beta = scheduler_output["beta"]  # batch_size,
        y_dist = y(x, beta)
        model_input = theta(y_dist)

        return {
            "ground_truth": x,
            "t": t,
            "model_input": model_input,
        }

    return collate_fn
