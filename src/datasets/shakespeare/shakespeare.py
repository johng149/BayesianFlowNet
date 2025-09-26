import random

import torch
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
        beta_1: float = 4.0,
        min_t: float = 1e-6,
        train: bool = True,
    ):
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

    def __getitem__(self, index):
        start = random.randint(0, len(self.data) - self.max_length)
        end = start + self.max_length
        seq = self.data[start:end]
        seq = F.one_hot(seq, num_classes=self.tokenizer.vocab_size())
        t = sample_t(1, self.min_t)
        beta = beta_t(self.beta_1, t)
        return {"x": seq, "t": t, "beta": beta, "beta_1": self.beta_1}
