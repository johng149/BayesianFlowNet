import torch
from torch import Tensor

def loss(beta_1: Tensor, t: Tensor, target: Tensor, model_output_probs: Tensor | None = None, model_output_logits: Tensor | None = None) -> Tensor:
    """
    Args:
        beta_1: Maximum possible accuracy (reached when t=1) of shape (batch_size,).
        t: A tensor representing the time step (batch_size,).
        target: Target tensor of shape (batch_size, seq_len, K).
        model_output_probs: Model output probabilities of shape (batch_size, seq_len, K). If None, model_output_logits must be provided.
        model_output_logits: Model output logits of shape (batch_size, seq_len, K). If None, model_output_probs must be provided.
    Returns:
        Loss value

    Must provide either model_output_probs or model_output_logits, but not both.
    """
    assert model_output_probs is None or model_output_logits is None, "Must provide either model_output_probs or model_output_logits, but not both"
    assert (model_output_probs is None or model_output_probs.shape == target.shape), "model_output_probs must have the same shape as target if provided"
    assert (model_output_logits is None or model_output_logits.shape == target.shape), "model_output_logits must have the same shape as target if provided"

    batch_size, seq_len, K = target.shape
    model_output = model_output_probs if model_output_probs is not None else torch.softmax(model_output_logits, dim=-1)
    result = torch.sum(K * beta_1 * t * torch.sum((target - model_output) ** 2) / (batch_size**2))
    return result / seq_len