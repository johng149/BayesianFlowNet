from typing import Callable, Tuple

import torch
from torch import Tensor


def half_callback_maker(
    ground_truth: Tensor,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """
    Create a callback that takes as input the model's prediction and makes the first half of the sequence
    match the ground truth.

    Args:
        - ground_truth (Tensor) Ground truth tensor of shape (batch_size, seq_len, K)
    Returns:
        Callable[[Tensor], Tensor]: A callback function that modifies the model's prediction.
    """
    batch_size, seq_len, K = ground_truth.shape
    indices = torch.arange(seq_len, device=ground_truth.device)
    lower_half = indices < (seq_len // 2)
    mask = lower_half.unsqueeze(0).unsqueeze(-1)

    # currently, the ground_truth is assumed to be one-hot encoded, which does not work well as logits
    # for categorical distribution, so we need to make all zeros into -inf
    x_zero = ground_truth.clone().float()
    x_zero[x_zero == 0] = -float("inf")

    def callback(model_prediction: Tensor) -> Tensor:
        """
        Modify the model's prediction to match the ground truth in the first half of the sequence.

        Args:
            - model_prediction (Tensor) Model's prediction tensor of shape (batch_size, seq_len, K)

        Returns:
            Tensor: Modified model prediction tensor.
        """
        return torch.where(mask, x_zero, model_prediction)

    expanded_mask = mask.expand(batch_size, -1, K)

    def masker(prediction: Tensor) -> Tensor:
        # works for both ground truth and model prediction, it will keep only the second half, as
        # that is the part that is relevant to calculating model accuracy
        assert prediction.shape == (batch_size, seq_len, K), (
            "Prediction shape must match ground truth shape."
            f" Got {prediction.shape} and {ground_truth.shape}."
        )
        return prediction[~expanded_mask].view(batch_size, -1, K)

    return callback, masker
