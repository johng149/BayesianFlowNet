import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from datasets import load_dataset
from src.datasets.discrete_helper import sample_t
from src.tokenizers.base import TokenizerBase


class ShakespeareDataset(Dataset):
    def __init__(
        self,
        tokenizer: TokenizerBase,
        max_length: int = 100,
        min_t: float = 1e-6,
        folds: int = 2,
        train: bool = True,
    ):
        assert (
            folds >= 2
        ), "loss variance estimation needs at least two folds to sample from"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_t = min_t
        self.folds = folds

        data = load_dataset(
            "karpathy/tiny_shakespeare", split="train" if train else "test"
        )
        self.data = self.tokenizer.encode(data["text"][0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        start = index
        end = start + self.max_length
        seq = self.data[start:end]
        t = sample_t(self.folds - 1, self.min_t)
        t = torch.cat((torch.tensor([self.min_t]), t), dim=0)
        return {"x": seq, "t": t}
