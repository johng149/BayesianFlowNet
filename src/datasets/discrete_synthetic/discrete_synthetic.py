import random
from typing import Dict, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.datasets.discrete_helper import beta_t, sample_t
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
        beta_1: float = 4.0,
        min_t: float = 1e-6,
    ):
        self.length = length
        self.tokenized_length = tokenized_length
        self.tokenizer = tokenizer
        self.mini = mini
        self.maxi = maxi
        self.min_t = min_t
        self.beta_1 = torch.tensor([beta_1])

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

    def __getitem__(self, idx) -> Dict[str, Union[Tensor, int]]:
        seq = self.generate_sequence()
        t = sample_t(1, self.min_t)
        beta = beta_t(self.beta_1, t)
        return {
            "x": seq,
            "t": t,
            "beta": beta,
            "beta_1": self.beta_1,
            "K": self.tokenizer.vocab_size(),
        }
