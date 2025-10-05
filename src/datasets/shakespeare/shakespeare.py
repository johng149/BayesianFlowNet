import random
from typing import Dict, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from datasets import load_dataset
from src.datasets.discrete_helper import beta_t, sample_t
from src.tokenizers.base import TokenizerBase


class ShakespeareDataset(Dataset):
    def __init__(
        self,
        tokenizer: TokenizerBase,
        max_length: int = 100,
        min_t: float = 1e-6,
        train: bool = True,
        beta_1: float | None = None,
    ):
        beta_1 = beta_1 if beta_1 is not None else 20.4054 / tokenizer.vocab_size()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_t = min_t
        self.beta_1 = torch.tensor([beta_1])

        data = load_dataset(
            "karpathy/tiny_shakespeare", split="train" if train else "test"
        )
        text = data["text"][0]  # pyright: ignore[reportIndexIssue]
        assert isinstance(text, str)
        self.data = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, Union[Tensor, int]]:
        # start = random.randint(0, len(self.data) - self.max_length)
        start = random.randint(0, 0)
        end = start + self.max_length
        seq = self.data[start:end]
        t = sample_t(1, self.min_t)
        beta = beta_t(self.beta_1, t)
        return {
            "x": seq,
            "t": t,
            "beta": beta,
            "beta_1": self.beta_1,
            "K": self.tokenizer.vocab_size(),
        }
