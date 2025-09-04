import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# much of the following code is lifted straight from
# https://arxiv.org/abs/1908.05164 (Unconstrained Monotonic NN)
# with only a handful of changes for our specific use case:
# 1. Positional encoding
# 2. No offset


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def compute_cc_weights(nb_steps):
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = 0.5
    lam[:, -1] = 0.5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W**2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = torch.tensor(lam.T @ W).float()
    steps = torch.tensor(
        np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps)
    ).float()

    return cc_weights, steps


def integrate(
    x0, nb_steps, step_sizes, integrand, h, compute_grad=False, x_tot=None, inv_f=False
):
    # Clenshaw-Curtis Quadrature Method
    cc_weights, steps = compute_cc_weights(nb_steps)

    device = x0.get_device() if x0.is_cuda or x0.is_mps else "cpu"
    if x0.is_mps:
        device = "mps"

    cc_weights, steps = cc_weights.to(device), steps.to(device)

    xT = x0 + nb_steps * step_sizes
    if not compute_grad:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        if inv_f:
            dzs = 1 / integrand(X_steps, h_steps)
        else:
            dzs = integrand(X_steps, h_steps)
        dzs = dzs.view(xT_t.shape[0], nb_steps + 1, -1)
        dzs = dzs * cc_weights.unsqueeze(0).expand(dzs.shape)
        z_est = dzs.sum(1)
        return z_est * (xT - x0) / 2
    else:

        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        x_tot = x_tot * (xT - x0) / 2
        x_tot_steps = x_tot.unsqueeze(1).expand(
            -1, nb_steps + 1, -1
        ) * cc_weights.unsqueeze(0).expand(x_tot.shape[0], -1, x_tot.shape[1])
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        x_tot_steps = x_tot_steps.contiguous().view(-1, x_tot.shape[1])

        g_param, g_h = computeIntegrand(
            X_steps, h_steps, integrand, x_tot_steps, nb_steps + 1, inv_f=inv_f
        )
        return g_param, g_h


def computeIntegrand(x, h, integrand, x_tot, nb_steps, inv_f=False):
    h.requires_grad_(True)
    with torch.enable_grad():
        if inv_f:
            f = 1 / integrand.forward(x, h)
        else:
            f = integrand.forward(x, h)

        g_param = _flatten(
            torch.autograd.grad(
                f, integrand.parameters(), x_tot, create_graph=True, retain_graph=True
            )
        )
        g_h = _flatten(torch.autograd.grad(f, h, x_tot))

    return g_param, g_h.view(int(x.shape[0] / nb_steps), nb_steps, -1).sum(1)


class ParallelNeuralIntegral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, h, nb_steps=20, inv_f=False):
        with torch.no_grad():
            x_tot = integrate(
                x0, nb_steps, (x - x0) / nb_steps, integrand, h, False, inv_f=inv_f
            )
            # Save for backward
            ctx.integrand = integrand
            ctx.nb_steps = nb_steps
            ctx.inv_f = inv_f
            ctx.save_for_backward(x0.clone(), x.clone(), h)
        return x_tot

    @staticmethod
    def backward(ctx, grad_output):  # pyright: ignore[reportIncompatibleMethodOverride]
        x0, x, h = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps
        inv_f = ctx.inv_f
        integrand_grad, h_grad = integrate(
            x0, nb_steps, x / nb_steps, integrand, h, True, grad_output, inv_f
        )
        x_grad = integrand(x, h)
        x0_grad = integrand(x0, h)
        # Leibniz formula
        return (
            -x0_grad * grad_output,
            x_grad * grad_output,
            None,
            integrand_grad,
            h_grad.view(h.shape),
            None,
        )


# PositionalEncoding is LLM generated, but works well enough. Will look into more principaled
# https://arxiv.org/abs/2006.10739 (Fourier Features paper) later
# actually we are trying to use monotonic functions here for the
# Advancing Constrained Monotonic Neural Network paper, which needs monotonic inputs
class PositionalEncoding(nn.Module):
    def __init__(self, encoding_dim: int, scaling: float = 3.0, epsilon: float = 1e-8):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.scaling = scaling
        self.epsilon = epsilon
        self.register_buffer("centers", torch.randn(encoding_dim))
        self.register_buffer("exp_scales", torch.logspace(-2, 0, encoding_dim))
        self.register_buffer("log_offsets", torch.logspace(-1, 1, encoding_dim))

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: A tensor of shape (batch_size, 1) representing time.
        Returns:
            A tensor of shape (batch_size, encoding_dim + 1) with positional encodings.
        """
        centers = self.centers
        assert isinstance(centers, Tensor)
        diffs = t.unsqueeze(-1) - centers
        sigmoid_features = torch.sigmoid(self.scaling * diffs).squeeze(1)

        exp_scales = self.exp_scales
        assert isinstance(exp_scales, Tensor)
        exp_features = torch.exp(exp_scales * t)

        log_offsets = self.log_offsets
        assert isinstance(log_offsets, Tensor)
        log_features = torch.log(t + log_offsets + self.epsilon)

        pe = (sigmoid_features + exp_features + log_features) / 3

        # Concatenate the original time 't' with its encoding.
        # This gives the network access to both the raw time and its non-linear features.
        return torch.cat([t, pe], dim=1)


class MonotonicLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        pre_activation: nn.Module = nn.Identity(),
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.act = pre_activation

    def forward(self, input):
        w_pos = self.weight.clamp(min=0.0)
        w_neg = self.weight.clamp(max=0.0)
        x_pos = F.linear(self.act(input), w_pos, self.bias)
        x_neg = F.linear(self.act(-input), w_neg, self.bias)
        return x_pos + x_neg


class IntegrandNN_PE(nn.Module):
    def __init__(self, in_d, hidden_layers, encoding_dim=32):
        super().__init__()
        self.pos_encoder = PositionalEncoding(encoding_dim)

        # The input to our network is the encoded time + the conditioning variables.
        # Encoded time has dimension: encoding_dim + 1 (for the original t)
        # Conditioning variables have dimension: in_d - 1 (since in_d includes time)
        net_input_dim = (encoding_dim + 1) + (in_d - 1)

        self.net_layers = []
        hs = [net_input_dim] + hidden_layers + [1]
        i = 0
        for h0, h1 in zip(hs, hs[1:]):
            activation = nn.Identity() if i == 0 else nn.ReLU()
            self.net_layers.extend([MonotonicLinear(h0, h1, pre_activation=activation)])
            i += 1
        self.net = nn.Sequential(*self.net_layers)

    def forward(self, t, h):
        """
        Args:
            t (Tensor): The integration variable, shape (batch_size, 1).
            h (Tensor): The conditioning variables, shape (batch_size, 1).
        """
        # 1. Encode the time variable, output should be (batch_size, encoding_dim + 1)
        t_encoded = self.pos_encoder(t)

        # 2. Combine with conditioning variables, so now (batch_size, encoding_dim + 2)
        combined_input = torch.cat((t_encoded, h), 1)

        # 3. Pass through the network, take exp since want only positive outputs
        net_out = self.net(combined_input)

        return F.softplus(net_out)


class MonotonicNN(nn.Module):
    def __init__(
        self,
        in_d,
        hidden_layers,
        beta_1: float,
        nb_steps=50,
        encoding_dim=32,
        learner_weight=0.0,
    ):
        super().__init__()
        # The MonotonicNN takes the variable to be integrated over (dim 1)
        # and conditioning variables (dim in_d-1)
        # I have found that in practice, this MonotonicNN struggles to learn properly, as such,
        # will be trying two-stage learning. First, we start with learner_weight = 0.0, so
        # we skip the integrand network and directly use the approximation beta_1 * ( t ** 2)
        # as the schedule.
        # once the main model has been trained to an extent, we freeze the main model and then
        # train this schedule, with learner weight = 1.0
        self.learner_weight = learner_weight
        self.beta_1 = beta_1
        self.integrand = IntegrandNN_PE(in_d, hidden_layers, encoding_dim=encoding_dim)
        self.net_layers = []
        hs = [in_d - 1] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net_layers.extend(
                [
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net_layers.pop()  # pop the last ReLU for the output layer
        # It will output the scaling factor.
        self.net = nn.Sequential(*self.net_layers)
        self.nb_steps = nb_steps

    def forward(self, x, h):
        learner_beta = 0.0
        reference_beta = self.beta_1 * (x**2)
        if self.learner_weight > 0.0:
            # `x` is the variable of integration, `h` is conditioning
            x0 = torch.zeros_like(x)

            flat_params = _flatten(self.integrand.parameters())
            # calculate integral up to 1 for normalization
            # but seems like we don't actually need it?
            # x1 = torch.ones_like(x)
            # integral_1 = ParallelNeuralIntegral.apply(x0, x1, self.integrand, flat_params, h, self.nb_steps)
            integral_x = ParallelNeuralIntegral.apply(
                x0, x, self.integrand, flat_params, h, self.nb_steps
            )
            assert isinstance(integral_x, Tensor)

            integral_norm = (
                integral_x  # / integral_1, seems we don't need this norming?
                # it's actually even worse than that. If we try to hard-code a target beta_1
                # value by doing beta_1 * integral_norm at time t=1.0, we have a much higher chance of ending up
                # with NaN loss, for whatever reason
            )

            scaling = self.scaling(h)
            learner_beta = integral_norm * scaling
        return (
            1 - self.learner_weight
        ) * reference_beta + self.learner_weight * learner_beta

    def scaling(self, h):
        # This method returns the scaling factor for the input `h`, maybe useful for debugging
        out = self.net(h)
        scaling = F.elu(out[:, [0]]) + 1.001
        return scaling
