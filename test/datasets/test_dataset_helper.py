import torch

from src.datasets.dataset_helper import sample_t


def test_sample_t_shape():
    t = sample_t(10, 1e-5)
    assert t.shape == (10,)


def test_sample_t_range():
    min_t = 1e-5
    t = sample_t(1000, min_t)
    assert torch.all(t >= min_t)
    assert torch.all(t <= 1.0)
