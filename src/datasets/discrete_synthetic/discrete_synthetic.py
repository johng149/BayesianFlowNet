import random

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.datasets.discrete_helper import sample_t
from src.tokenizers.discrete_synthetic.discrete_synthetic_tokenizer import (
    DiscreteSyntheticTokenizer,
)


class DiscreteSyntheticDataset(Dataset):
    def __init__(
        self,
        tokenizer: DiscreteSyntheticTokenizer,
        length: int = 32,
        tokenized_length: int = 32,
        mini: int = 0,
        maxi: int = 100,
        min_t: float = 1e-6,
        folds: int = 2,
    ):
        assert (
            folds >= 2
        ), "loss variance estimation needs at least two folds to sample from"
        self.length = length
        self.tokenized_length = tokenized_length
        self.tokenizer = tokenizer
        self.mini = mini
        self.maxi = maxi
        self.min_t = min_t
        self.folds = folds

    def generate_sequence(self):
        start = random.randint(self.mini, self.maxi - self.length)
        end = start + self.length
        acc = ""
        for i in range(start, end + 1):
            for c in str(i):
                acc += " " + c
            acc += " ,"
        tokenized = self.tokenizer.encode(acc)
        return tokenized[: self.tokenized_length]

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        seq = self.generate_sequence()
        t = sample_t(self.folds - 1, self.min_t)
        t = torch.cat((torch.tensor([self.min_t]), t), dim=0)
        return {"x": seq, "t": t}
