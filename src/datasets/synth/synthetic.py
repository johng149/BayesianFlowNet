import random

from torch.utils.data import Dataset

from src.common.data_prep import sample_t
from src.datasets.dataset_helper import BFNDataset, DatasetOutput
from src.tokenizers.base import TokenizerBase


class DiscreteSyntheticDataset(Dataset, BFNDataset):
    def __init__(
        self,
        tokenizer: TokenizerBase,
        length: int = 32,
        mini: int = 0,
        maxi: int = 100,
        min_t: float = 1e-6,
        train: bool = True,  # for consistency with other datasets, this synthetic dataset does not use this parameter
    ):
        assert mini + length < maxi, "Invalid mini, maxi, length configuration."
        self.length = length
        self.tokenized_length = length // 2
        self.tokenizer = tokenizer
        self.mini = mini
        self.maxi = maxi
        self.min_t = min_t

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

    def __getitem__(self, idx) -> DatasetOutput:
        seq = self.generate_sequence()
        t = sample_t(1, self.min_t)
        return {"x": seq, "t": t}
