import random
from typing import Callable, Dict, List, Tuple

import einops
import torch
from torch import Tensor
from torch.nn import functional as F

from src.tokenizers.base import TokenizerBase


def span_masking(original: Tensor, mask: Tensor, span_idx: int) -> Tensor:
    """
    Mask the spans of `original` tensor, where spans are defined as
    any position where `mask` is True. Each span will be collapsed
    into a single token, which will be represented by `span_idx`.

    The `original` tensor and `mask` tensor must have the same shape
    and must be 1D tensors.

    The `span_idx` must be different from any normal token value, which
    means it cannot appear in `original` that is passed in

    @param original: The original tensor to mask
    @param mask: The mask tensor, where True values represent spans
    @param span_idx: The index to represent masked spans
    @return: The masked tensor

    For example:
    original = torch.tensor([1, 2, 3, 4, 5])
    mask     = torch.tensor([0, 1, 1, 0, 1]).bool()
    span_idx = -1
    masked = span_masking(original, mask, span_idx)

    The `masked` tensor will be:
    torch.tensor([1, -1, 4, -1])
    """
    assert not (original == span_idx).any(), "span_idx must not be in original tensor"
    original[mask] = span_idx

    # this will collapse all consecutive span_idx values into a single value
    # the logit is as follows:
    # if the current value is the same as the previous value, we are either
    # in a contiguous span or in a series of duplicate but non-span values
    # if the current value is not the span_idx, then we are in a series of
    # duplicate but non-span values. Since it is non-span values, we want to
    # keep it.
    # if the current value is not the same as the previous value, then even
    # if current value is span_idx, we want to keep it since it is the start
    # of a new span.
    # however, this also means that the span_idx must be different from
    # any normal token value, otherwise we would not be able to distinguish
    # between a span and a series of duplicate but non-span values.
    prev = None
    elements = [prev := x for x in original if x != prev or x != span_idx]
    return torch.stack(elements)


def flow(
    original: Tensor, mask: Tensor, enc_span_idx: int, targ_span_idx: int
) -> Tuple[Tensor, Tensor]:
    """
    Given an `original` tensor and a `mask` tensor, this function
    will return two tensors: one tensor with the encoder spans
    masked and another tensor with the target spans masked.

    It is assumed that the given mask will be a 1D boolean tensor with the
    same shape as the `original` tensor. Also, the mask tensor should be
    True where the spans in the encoder should be masked and False otherwise.

    The target span tensor will be the inverse of the encoder span tensor.
    Any elements that are not masked in the encoder will be masked in the
    target tensor and vice versa.

    Also consecutive masked elements will be collapsed into a single token
    represented by `enc_span_idx` and `targ_span_idx` respectively.

    @param original: The original tensor to mask
    @param mask: The mask tensor, where True values represent spans
    @param enc_span_idx: The index to represent encoder masked spans
    @param targ_span_idx: The index to represent target masked spans
    @return: A tuple of two tensors, the encoder and target masked tensors

    For example:
    original = torch.tensor([1, 2, 3, 4, 5])
    mask     = torch.tensor([0, 1, 1, 0, 1]).bool()
    enc_span_idx = -1
    targ_span_idx = -2
    enc, targ = flow(original, mask, enc_span_idx, targ_span_idx)

    The `enc` tensor will be:
    torch.tensor([1, -1, 4, -1])

    The `targ` tensor will be:
    torch.tensor([-2, 2, 3, -2, 5])
    """
    target_mask = ~mask
    enc = span_masking(original.clone(), mask, enc_span_idx)
    targ = span_masking(original.clone(), target_mask, targ_span_idx)
    return enc, targ


def limit_percentage(mask, max_percentage):
    """
    Ensures that no more than `max_percentage` of the 1-D mask is True

    Args:
        - mask (torch.Tensor): 1-D tensor of booleans.
        - max_percentage (float): The maximum proportion of the mask that should be True.

    Returns:
        - mask (torch.Tensor): The mask with no more than `max_percentage` True values.
    """
    assert mask.dim() == 1, "Mask must be a 1-D tensor."
    assert 0 <= max_percentage <= 1, "Invalid max percentage."

    max_true = int(len(mask) * max_percentage)
    true_indices = torch.nonzero(mask).flatten()

    if len(true_indices) > max_true:
        excess_indices = true_indices[torch.randperm(len(true_indices))][max_true:]
        mask[excess_indices] = False

    return mask


def mask_span(
    tokens,
    max_num_spans: int = 6,
    max_span_fill: float = 0.8,
    min_num_spans: int = 0,
    min_span_fill: float = 0,
    hard_fill=True,
):
    """
    Creates a mask for the tokens, where spans of tokens are masked out. Elements in mask that
    are True indicate that the token should be masked out.

    Args:
        - tokens (torch.Tensor): 1-D tensor of tokens to mask out.
        - max_num_spans (int): The maximum number of spans to mask out.
        - max_span_fill (float): The maximum proportion of tokens to mask out in a span.
        - min_num_spans (int): The minimum number of spans to mask out.
        - min_span_fill (float): The minimum proportion of tokens to mask out in a span.
        - hard_fill (bool): If True, will ensure that no more than `max_span_fill` percent of the
            tokens are masked out. If False, the percentage of masked tokens may be higher.

    Returns:
        - mask (torch.Tensor): A mask of the same shape as `tokens` where True indicates that the
            token should be masked out.
    """
    assert tokens.dim() == 1, "Tokens must be a 1-D tensor."
    assert 0 <= min_span_fill <= max_span_fill <= 1, "Invalid span fill percentages."
    assert min_num_spans > 0, "min_num_spans must be positive."
    assert max_num_spans >= min_num_spans, "max_num_spans must be >= min_num_spans."

    fill_percent = random.uniform(min_span_fill, max_span_fill)
    max_mask_len = int(len(tokens) * fill_percent)
    num_spans = random.randint(min_num_spans, max_num_spans)
    uniform_amount = max_mask_len // num_spans
    start_indices = torch.randint(0, max(1, len(tokens) - uniform_amount), (num_spans,))
    # span_lengths = torch.randint(1, max(2, max_mask_len + 1), (num_spans,))
    # paper says that instead of random span lengths, we use about the same
    # amount for each span
    span_lengths = torch.ones((num_spans,), dtype=torch.int64) * uniform_amount
    span_lengths = torch.min(span_lengths, len(tokens) - start_indices)
    indices = torch.arange(len(tokens))
    mask = torch.any(
        (indices >= start_indices[:, None])
        & (indices < start_indices[:, None] + span_lengths[:, None]),
        dim=0,
    )

    if hard_fill:
        mask = limit_percentage(mask, fill_percent)

    return mask


def collate_fn(
    batch: List[Dict[str, Tensor]],
    vocab_size: int,
    enc_span_mask_token_id: int,
    dec_span_mask_token_id: int,
    max_masks: int = 3,
    min_masks: int = 1,
    max_fill: float = 0.95,
    min_fill: float = 0.0,
) -> Dict[str, Tensor]:
    """
    This collate function will truncate all sequences to the minimum length of
    the sequences in the batch

    Args:
        batch: List of dictionaries, each containing 'x', 't'
        vocab_size: Size of the vocabulary (K)
    Returns:
        A dictionary with keys 'x', 't' where 'x' is a tensor of shape
        (batch_size, seq_len, K), 't' is a tensor of shape (batch_size,)
    """
    # x = [item["x"] for item in batch]
    # min_length = min(seq.shape[0] for seq in x)
    # x = [tensor[:min_length] for tensor in x]
    # x = [F.one_hot(tensor, num_classes=vocab_size).float() for tensor in x]

    # t = torch.cat([item["t"] for item in batch], dim=0)  # Shape: (batch_size * folds,)
    # folds = batch[0]["t"].shape[0]  # all items should have the same number of folds

    # x = torch.stack(
    #     [tensor.unsqueeze(0).expand(folds, -1, -1) for tensor in x], dim=0
    # )  # Shape: (batch_size, folds, seq_len, K)

    # x = x.view(
    #     -1, x.shape[-2], x.shape[-1]
    # )  # Reshape to (batch_size * folds, seq_len, K)

    # return {"x": x, "t": t}
    x = [item["x"] for item in batch]
    min_length = min(seq.shape[0] for seq in x)
    x = [tensor[:min_length] for tensor in x]

    encs, targs = [], []
    mask = mask_span(
        tokens=x[0],
        max_num_spans=max_masks,
        max_span_fill=max_fill,
        min_num_spans=min_masks,
        min_span_fill=min_fill,
        hard_fill=True,
    )
    for sample in x:
        enc, targ = flow(sample, mask, enc_span_mask_token_id, dec_span_mask_token_id)
        encs.append(enc)
        targs.append(targ)

    try:
        encs = torch.stack(encs, dim=0)  # Shape: (batch_size, seq_len)
    except RuntimeError as e:
        print(f"Error stacking encoder inputs: {e}")
        torch.save(
            {
                "x": x,
                "mask": mask,
                "encs": encs,
                "targs": targs,
            },
            "collate_error_debug.pt",
        )
        raise e

    targs = [F.one_hot(tensor, num_classes=vocab_size).float() for tensor in targs]
    targs = torch.stack(targs, dim=0)  # Shape: (batch_size, seq_len, K)

    t = torch.cat([item["t"] for item in batch], dim=0)  # Shape: (batch_size * folds,)
    folds = batch[0]["t"].shape[0]  # all items should have the same number of folds

    encs = einops.repeat(
        encs, "batch_size seq_len -> (batch_size folds) seq_len", folds=folds
    )
    targs = einops.repeat(
        targs, "batch_size seq_len K -> (batch_size folds) seq_len K", folds=folds
    )

    return {"encoder_input": encs, "target": targs, "t": t}


def collate_fn_maker(
    tokenizer: TokenizerBase,
    max_masks: int = 3,
    min_masks: int = 1,
    max_fill: float = 0.95,
    min_fill: float = 0.0,
) -> Callable[[List[Dict[str, Tensor]]], Dict[str, Tensor]]:
    """
    This function returns a collate function that will truncate all sequences to the minimum length of
    the sequences in the batch and apply Double-Source Text Infilling span masking to the sequences.

    See https://arxiv.org/pdf/2304.11791 section 4.1

    Args:
        tokenizer: An instance of TokenizerBase to get special token ids.
        max_masks: Maximum number of spans to mask out.
        min_masks: Minimum number of spans to mask out.
        max_fill: Maximum proportion of tokens to mask out in a span.
        min_fill: Minimum proportion of tokens to mask out in a span.
    Returns:
        A collate function that can be used in a DataLoader.
    """

    def collate(
        batch: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        return collate_fn(
            batch,
            vocab_size=tokenizer.vocab_size(),
            enc_span_mask_token_id=tokenizer.enc_mask_token_id(),
            dec_span_mask_token_id=tokenizer.dec_mask_token_id(),
            max_masks=max_masks,
            min_masks=min_masks,
            max_fill=max_fill,
            min_fill=min_fill,
        )

    return collate
