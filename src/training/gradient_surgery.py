import copy
import random

import numpy as np
import torch
from accelerate import Accelerator
from torch import Size, Tensor, nn
from torch.optim import Optimizer


def retrieve_grads(
    optim: Optimizer,
) -> tuple[list[Tensor], list[Size], list[Tensor]]:
    grad, shape, has_grad = [], [], []
    for group in optim.param_groups:
        for p in group["params"]:
            shape.append(p.shape)
            if p.grad is None:
                grad.append(torch.zeros_like(p, device=p.device))
                has_grad.append(torch.zeros_like(p, device=p.device))
            else:
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p, device=p.device))
    return grad, shape, has_grad


def flatten(grads: list[Tensor]) -> Tensor:
    flatten_grad = torch.cat([g.flatten() for g in grads])
    return flatten_grad


def project_conflicting(grads: list[Tensor], has_grads: list[Tensor]) -> Tensor:
    shared = torch.stack(has_grads).prod(0).bool()
    pc_grad = copy.deepcopy(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
    merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)

    return merged_grad


def unflatten(grads: Tensor, shapes: list[Size]) -> list[Tensor]:
    unflatten_grad, idx = [], 0
    for shape in shapes:
        length = np.prod(shape)
        unflatten_grad.append(grads[idx : idx + length].view(shape).clone())
        idx += length
    return unflatten_grad


def set_grad(grads: list[Tensor], optim: Optimizer):
    idx = 0
    for group in optim.param_groups:
        for p in group["params"]:
            p.grad = grads[idx]
            idx += 1


def gradient_surgery(
    accelerator: Accelerator,
    body_optim: Optimizer,
    schedule_optim: Optimizer,
    loss: Tensor,
    var_loss: Tensor,
    div_loss: Tensor,
    alpha_var_loss: Tensor,
    body: nn.Module,
    schedule: nn.Module,
    grad_clip_norm: float | None = None,
):
    body_optim.zero_grad(set_to_none=True)
    schedule_optim.zero_grad(set_to_none=True)

    l = loss + var_loss + div_loss + alpha_var_loss
    accelerator.backward(l)
    body_optim.step()
    schedule_optim.step()

    # accelerator.backward(l, retain_graph=True)
    # if accelerator.sync_gradients and grad_clip_norm is not None:
    #     accelerator.clip_grad_norm_(body.parameters(), grad_clip_norm)
    # body_optim.step()

    # grads, shapes, has_grads = [], [], []

    # l_grad, l_shape, l_has_grad = retrieve_grads(schedule_optim)
    # grads.append(flatten(l_grad))
    # has_grads.append(flatten(l_has_grad))
    # shapes.append(l_shape)

    # schedule_optim.zero_grad(set_to_none=True)
    # accelerator.backward(div_loss, retain_graph=True)

    # div_grad, div_shape, div_has_grad = retrieve_grads(schedule_optim)
    # grads.append(flatten(div_grad))
    # has_grads.append(flatten(div_has_grad))
    # shapes.append(div_shape)

    # projected = project_conflicting(grads, has_grads)
    # unflattened = unflatten(projected, shapes[0])

    # set_grad(unflattened, schedule_optim)
    # if accelerator.sync_gradients and grad_clip_norm is not None:
    #     accelerator.clip_grad_norm_(schedule.parameters(), grad_clip_norm)
    # schedule_optim.step()
