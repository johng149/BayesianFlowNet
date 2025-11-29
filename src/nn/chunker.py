from collections import namedtuple

import torch
from assoc_scan import AssocScan
from einops import rearrange, repeat
from einx import multiply
from torch import Tensor, arange, cat
from torch.nested import nested_tensor
from torch.nn import Linear, Module, Parameter
from torch.nn.functional import cosine_similarity, pad

Outputs = namedtuple(
    "Outputs", ["downsampled", "upsample_fn", "weighted_aux_ratio_loss"]
)

Intermediates = namedtuple(
    "Intermediates",
    [
        "mask",
        "probs",
        "chunk_lens",
        "boundary_mask",
        "residual",
        "gates",
        "upsampler_output_scale",
        "aux_ratio_loss",
        "new_seq_lens",
    ],
)


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def straight_through(t, value):
    return t + (value - t).detach()


def frac_gradient(t: Tensor, frac=1.0) -> Tensor:
    if frac == 1:
        return t

    t_grad = t * frac
    return straight_through(t_grad, t)


class PackDynamicSequenceChunker(Module):
    def __init__(
        self,
        dim: int,
        dim_queries_keys: int | None = None,
        boundary_threshold=0.5,
        target_avg_token_length=6.0,  # N in eq(10)
        ratio_loss_weight=3e-2,
        handle_residual_proj=False,  # turning this on will automatically handle a projection of the residual and its application in the inverse upsample function
        assoc_scan_use_accelerated=False,
        learning_rate_difference=0.75,  # in the paper, they report that as one moves up a hierarchy, the learning rate needs to decrease. we'll default to 0.75 for the rough 2.0 -> 1.5 somewhere in the appendix from level 0 -> 1
        straight_through_frac_vecs=True,  # improvisation where F receives gradients through straight-through with sigmoid
    ):
        super().__init__()
        dim_queries_keys = default(dim_queries_keys, dim)
        assert dim_queries_keys is not None

        # linear to queries and keys

        self.to_queries_keys = Linear(dim, dim_queries_keys * 2, bias=False)

        # start key token, so first token can be segmented / chunked out

        self.start_key_token = Parameter(
            torch.randn(dim_queries_keys) * 1e-2
        )  # presumably, need a start key token for the first token, open an issue if i got it wrong

        # threshold to determine boundary

        assert 0.0 < boundary_threshold < 1.0

        self.boundary_threshold = boundary_threshold

        # smoothing related

        self.smooth_assoc_scan = AssocScan(use_accelerated=assoc_scan_use_accelerated)

        # maybe residual proj

        self.handle_residual_proj = handle_residual_proj

        if handle_residual_proj:
            self.residual_proj = Linear(dim, dim)

        # learning rate modulation, appendix C
        # the multiplier on the learning rate as one goes from outer to inner of the h-net, and inverse of this value from inner to outer

        self.learning_rate_difference = learning_rate_difference

        # ratio aux loss related

        self.target_avg_token_length = target_avg_token_length

        self.straight_through_frac_vecs = straight_through_frac_vecs

        self.ratio_loss_weight = ratio_loss_weight

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    def upsample(
        self, downsampled: Tensor, intermediates: Intermediates, apply_scale=True
    ) -> Tensor:
        batch, needs_grad, device = (
            downsampled.shape[0],
            downsampled.requires_grad,
            downsampled.device,
        )

        mask = intermediates.mask
        gates = intermediates.gates
        residual = intermediates.residual

        # smoothing module for improved gradients eq(5)

        downsampled = self.smooth_assoc_scan(gates, downsampled)

        # upsample

        downsampled_without_padding = downsampled[mask]
        chunk_lens_without_padding = intermediates.chunk_lens[mask]

        seq = arange(downsampled_without_padding.shape[0], device=device)

        repeated_indices = torch.repeat_interleave(
            seq, chunk_lens_without_padding, dim=0
        )
        upsampled = downsampled_without_padding[repeated_indices]

        upsampled = rearrange(upsampled, "(b n) d -> b n d", b=batch)

        scale = intermediates.upsampler_output_scale

        if needs_grad and apply_scale and exists(scale):
            upsampled = multiply("b n d, b n", upsampled, scale)

        if self.handle_residual_proj:
            upsampled = upsampled + self.residual_proj(residual)

        upsampled = frac_gradient(upsampled, self.learning_rate_difference)

        return upsampled

    def forward(
        self,
        tokens,  # float[b n d] or float[total_n d] if seq_lens is specified,
        seq_lens: Tensor | None = None,
        return_intermediates=False,
        return_only_chunk_lens=False,
    ):
        with torch.no_grad():
            if seq_lens is not None:
                total_lens = seq_lens.sum().item()
                document_ids = torch.repeat_interleave(
                    torch.arange(len(seq_lens), device=seq_lens.device), seq_lens
                )

                # a sequence position with 1 in probs_mask is the position of the first
                # token of a new document, which means it must be a chunk start with
                # probability 1
                packed_probs_mask = torch.zeros_like(document_ids)
                packed_probs_mask[1:] = document_ids[:-1] != document_ids[1:]

                # however, since the sequence position is the start of a new document,
                # we must prevent the associative scan from reading from the token before
                # it. To do this, we reverse probs_mask, so the sequence position that used
                # to be 1 becomes 0 and the positions that used to be 0 become 1.
                # this means that at the start of each new document, the token cannot
                # read from the token before it
                packed_gate_mask = -1 * (packed_probs_mask - 1)
                tokens = tokens.unsqueeze(0)
            else:
                packed_probs_mask = None
                packed_gate_mask = None
                document_ids = None

        batch, length, device = *tokens.shape[:2], tokens.device

        residual = tokens

        queries, keys = self.to_queries_keys(tokens).chunk(2, dim=-1)

        start_keys = repeat(self.start_key_token, "d -> b 1 d", b=batch)

        keys = cat((start_keys, keys), dim=1)

        if packed_probs_mask is not None:
            # when packed, the keys end up being compared incorrectly at this current stage
            # for example, suppose we have two documents of lengths 2 and 2.
            # if passed individually, each document's first token will compare against the start key token
            # however, when packed, the 3rd token (first token of second document)
            # will compare against the key of the 2nd token, resulting in a wrong cosine_similarity
            # which later impacts the probability
            # at first I thought this would be fine because we hard set the probability, however
            # now I recall that in the associative scan smoothing, this probability term is involved
            # beyond the gate itself, which would result in an incorrect calculation, so
            # we need to make all those keys that are at the start of a new document
            # equal to the start key token

            # first, we start by adding a 1 to the right side of the packed_probs_mask, this is to account
            # for the fact that when calculating cosine similarity, we use `keys[:, :-1]`, so it is shifted
            # so the placement of the start key token needs to be shifted as well
            packed_probs_mask_with_start = pad(packed_probs_mask, (0, 1), value=0)

            # and now, for all sequence positions where packed_probs_mask_with_start is 1,
            # we set the corresponding keys to the start key token
            keys[:, packed_probs_mask_with_start == 1] = start_keys

        # each query looks at the previous key to determine if distance is greater than some threshold for determining a boundary exists (they use 0.5 as threshold)

        cosine_sim = cosine_similarity(queries, keys[:, :-1], dim=-1)

        probs = (
            1.0 - cosine_sim
        ) * 0.5  # cosine sim is -1. to 1., this transforms it to 0. to 1.

        boundary_mask = probs > self.boundary_threshold  # bool[b n]

        boundary_mask[:, 0] = True  # first token must always be boundary

        if packed_probs_mask is not None:
            # at all positions where the packed_probs_masking is 1, it means it is the start
            # of a new document. We must force these positions to be boundaries
            # previously I tried doing it by setting probs to 1, but that
            # will cause issues later down the line because downsampling tensor is multiplied
            # by the probs, so we must directly set the boundary mask instead
            boundary_mask = torch.where(packed_probs_mask == 1, True, boundary_mask)

        # compute some lengths, per chunk and number of chunks per batch

        num_chunks = boundary_mask.long().sum(dim=-1)

        boundary_mask_with_end = pad(boundary_mask, (0, 1), value=True)
        sel_indices = repeat(
            arange(boundary_mask_with_end.shape[-1], device=device), "n -> b n", b=batch
        )[boundary_mask_with_end]

        sel_indices = nested_tensor(
            sel_indices.split((num_chunks + 1).tolist()),
            layout=torch.jagged,
            device=device,
        )

        sel_indices = sel_indices.to_padded_tensor(padding=-1)

        mask = (sel_indices != -1)[:, 1:]

        chunk_lens = sel_indices[:, 1:] - sel_indices[:, :-1]
        chunk_lens.masked_fill_(~mask, 0)

        # early return chunk lens if using a trained module as a tokenizer

        if return_only_chunk_lens:
            return chunk_lens

        # downsampling - they show in their experiments that picking out the boundary tokens works just fine

        boundary_tokens = tokens[boundary_mask]  # pick out boundary tokens

        tokens_nt = nested_tensor(
            boundary_tokens.split(num_chunks.tolist()),
            layout=torch.jagged,
            device=device,
            requires_grad=True,
        )

        downsampled_tokens = tokens_nt.to_padded_tensor(padding=0.0)

        # smoothing module for improved gradients eq(5)

        probs_nt = nested_tensor(
            probs[boundary_mask].split(num_chunks.tolist()),
            layout=torch.jagged,
            device=device,
            requires_grad=True,
        )

        boundary_probs = probs_nt.to_padded_tensor(padding=0.0)

        gates = 1.0 - boundary_probs

        if packed_gate_mask is not None:
            # at all positions where the packed_gate_masking is 0, it means it is the start
            # of a new document. We must prevent associative scan from allowing
            # this starting token from reading into the past document
            # also, gradients cannot propagate through this to modify this gating, as it is
            # fixed by the document sequence
            packed_gate_mask_nt = nested_tensor(
                packed_gate_mask.unsqueeze(0)[boundary_mask].split(num_chunks.tolist()),
                layout=torch.jagged,
                device=device,
                requires_grad=False,
            )
            packed_gate_masking = packed_gate_mask_nt.to_padded_tensor(padding=1.0)
            gates = gates * packed_gate_masking

        downsampled_tokens = multiply("b n d, b n", downsampled_tokens, boundary_probs)

        # for the upsampler

        confidence = torch.where(boundary_mask, probs, 1.0 - probs)

        # defaults if not training

        upsampler_output_scale = None
        aux_loss = self.zero
        weighted_aux_loss = self.zero

        needs_grad = tokens.requires_grad

        if needs_grad:
            # straight through for 1. multiplier on the expanded processed boundary tokens

            upsampler_output_scale = straight_through(confidence, 1.0)

            # auxiliary ratio loss in section 2.3.2, eq (10)
            # lets follow their notation

            N = self.target_avg_token_length

            F = boundary_mask.float()
            G = probs.mean(dim=-1)

            # allow for a soft F to straight through - https://arxiv.org/abs/2505.22074

            if self.straight_through_frac_vecs:
                F_soft = (probs - self.boundary_threshold).sigmoid()
                F = straight_through(F_soft, F)

            F = F.mean(dim=-1)

            aux_ratio_loss = N / (N - 1) * ((N - 1) * F * G + (1.0 - F) * (1.0 - G))

            aux_loss = aux_ratio_loss.mean()
            weighted_aux_loss = aux_loss * self.ratio_loss_weight

        # intermediates
        if document_ids is not None:
            assert seq_lens is not None
            # this minlength should not be necessary as the boundaries should
            # guarantee that each document has at least one chunk
            new_seq_lens = torch.bincount(
                document_ids,
                weights=boundary_mask.squeeze(0).long(),
                minlength=len(seq_lens),
            ).long()
        else:
            new_seq_lens = num_chunks

        intermediates = Intermediates(
            mask,
            probs,
            chunk_lens,
            boundary_mask,
            residual,
            gates,
            upsampler_output_scale,
            aux_loss,
            new_seq_lens,
        )

        # return the upsample function

        def upsample(downsampled, apply_scale=True):
            downsampled_input = (
                downsampled.unsqueeze(0) if downsampled.ndim == 2 else downsampled
            )
            upsampled = self.upsample(
                downsampled_input, intermediates, apply_scale=apply_scale
            )
            return upsampled.squeeze(0) if downsampled.ndim == 2 else upsampled

        # adjust learning rate

        downsampled_tokens = frac_gradient(
            downsampled_tokens, self.learning_rate_difference**-1
        )

        if packed_probs_mask is not None:
            downsampled_tokens = downsampled_tokens.squeeze(0)

        # returning

        outputs = Outputs(downsampled_tokens, upsample, weighted_aux_loss)

        if not return_intermediates:
            return outputs

        return outputs, intermediates
        return outputs, intermediates
