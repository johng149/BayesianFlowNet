import torch
from torch import Tensor


def doc_ids_to_cu_seqlen(doc_ids: Tensor) -> Tensor:
    """
    Convert document IDs to cumulative sequence lengths for use in attention mechanisms.

    Args:
        doc_ids (Tensor): A tensor of shape (batch_size, seq_len) containing document IDs.

    Returns:
        Tensor: A 1D tensor of cumulative sequence lengths.
    """
    batch_size, seq_len = doc_ids.shape
    # Find where groups change
    is_new_group = torch.cat(
        [
            torch.ones_like(doc_ids[:, :1], dtype=torch.bool),
            doc_ids[:, 1:] != doc_ids[:, :-1],
        ],
        dim=1,
    )

    # Get the indices where new groups start
    group_start_indices = torch.where(is_new_group)[1]
    return torch.cat(
        [group_start_indices, torch.tensor([seq_len], device=doc_ids.device)]
    )
