import pytest
import torch
from torch import Tensor
from unittest.mock import patch, MagicMock

from src.datasets.shakespeare.shakespeare import ShakespeareDataset
from src.tokenizers.base import TokenizerBase


class MockTokenizer(TokenizerBase):
    def vocab_size(self) -> int:
        return 10

    def encode(self, text: str) -> Tensor:
        return torch.tensor([i for i, char in enumerate(text)], dtype=torch.long)

    def _decode(self, tokens: Tensor) -> str:
        return "".join([str(token.item()) for token in tokens])


@pytest.fixture
def mock_load_dataset():
    with patch("src.datasets.shakespeare.shakespeare.load_dataset") as mock_load:
        # Create a mock dataset object that returns a sample text
        mock_dataset = {"text": ["This is a test text for Shakespeare dataset."]}
        mock_load.return_value = mock_dataset
        yield mock_load


def test_shakespeare_dataset_init(mock_load_dataset):
    tokenizer = MockTokenizer()
    dataset = ShakespeareDataset(tokenizer)

    mock_load_dataset.assert_called_once_with(
        "karpathy/tiny_shakespeare", split="train"
    )
    assert dataset.tokenizer is tokenizer
    assert dataset.max_length == 100
    assert dataset.min_t == 1e-6
    assert isinstance(dataset.data, Tensor)
    assert len(dataset.data) > 0


def test_shakespeare_dataset_init_test_split(mock_load_dataset):
    tokenizer = MockTokenizer()
    dataset = ShakespeareDataset(tokenizer, train=False)
    mock_load_dataset.assert_called_once_with(
        "karpathy/tiny_shakespeare", split="test"
    )


def test_shakespeare_dataset_len(mock_load_dataset):
    tokenizer = MockTokenizer()
    dataset = ShakespeareDataset(tokenizer)
    # The length of the dataset should be the length of the encoded text
    text = "This is a test text for Shakespeare dataset."
    encoded_text = tokenizer.encode(text)
    assert len(dataset) == len(encoded_text)


def test_shakespeare_dataset_getitem(mock_load_dataset):
    tokenizer = MockTokenizer()
    max_length = 10
    dataset = ShakespeareDataset(tokenizer, max_length=max_length)

    # Since __getitem__ has random slicing, we'll patch random.randint
    with patch("src.datasets.shakespeare.shakespeare.random.randint") as mock_randint:
        mock_randint.return_value = 5  # Let's always start at index 5
        item = dataset[0]  # Index doesn't matter for this implementation

        assert "x" in item
        assert "t" in item

        seq = item["x"]
        t = item["t"]

        assert isinstance(seq, Tensor)
        assert seq.dtype == torch.long
        assert len(seq) == max_length

        # Check if the sequence is correct based on the mocked start
        expected_seq = dataset.data[5 : 5 + max_length]
        assert torch.equal(seq, expected_seq)

        assert isinstance(t, Tensor)
        assert t.ndim == 1
        assert t.shape[0] == 1
        assert dataset.min_t <= t.item() <= 1.0
