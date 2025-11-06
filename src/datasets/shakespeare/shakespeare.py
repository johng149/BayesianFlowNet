import random
from typing import Dict, Union

from torch import Tensor
from torch.utils.data import Dataset

from datasets import load_dataset
from src.datasets.dataset_helper import sample_t
from src.tokenizers.base import TokenizerBase


class ShakespeareDataset(Dataset):
    def __init__(
        self,
        tokenizer: TokenizerBase,
        max_length: int = 100,
        min_t: float = 1e-6,
        train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_t = min_t

        data = load_dataset(
            "karpathy/tiny_shakespeare", split="train" if train else "test"
        )
        text = data["text"][0]  # pyright: ignore[reportIndexIssue]
        assert isinstance(text, str)
        self.data = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, Union[Tensor, int]]:
        start = random.randint(0, len(self.data) - self.max_length)
        end = start + self.max_length
        seq = self.data[start:end]
        t = sample_t(1, self.min_t)
        return {
            "x": seq,
            "t": t,
        }
