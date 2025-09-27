import random

import pytest
import torch

from src.datasets.discrete_helper import collate_fn
from src.datasets.discrete_synthetic.discrete_synthetic import DiscreteSyntheticDataset
from src.tokenizers.discrete_synthetic.discrete_synthetic_tokenizer import (
    DiscreteSyntheticTokenizer,
)


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(420)


def test_init():
    tokenizer = DiscreteSyntheticTokenizer()
    dataset = DiscreteSyntheticDataset(tokenizer)
    assert dataset.length == 32
    assert dataset.tokenized_length == 32
    assert dataset.mini == 0
    assert dataset.maxi == 100


def test_getitem():
    tokenizer = DiscreteSyntheticTokenizer()
    dataset = DiscreteSyntheticDataset(tokenizer)
    sample = next(iter(dataset))
    sample = collate_fn([sample])["x"][0]
    expected = torch.tensor(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 3
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 7
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 8
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 9
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 3
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )  # 10
    # we know this since we seeded the random generator
    assert torch.equal(sample, expected)


def test_decode():
    tokenizer = DiscreteSyntheticTokenizer()
    dataset = DiscreteSyntheticDataset(tokenizer)
    sample = next(iter(dataset))
    sample = collate_fn([sample])["x"][0]
    decoded = tokenizer.decode(sample)
    expected = "3 , 4 , 5 , 6 , 7 , 8 , 9 , 1 0 , 1 1 , 1 2 , 1 3 , 1 4 , 1 5 ,"
    assert decoded == expected
