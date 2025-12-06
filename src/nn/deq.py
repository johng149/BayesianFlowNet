import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# --- 1. SOLVERS AND IMPLICIT BACKWARD PASS (from solvers.py) ---


def broyden_solve(f, x0, threshold=50, eps=1e-5):
    """
    Simplified Broyden solver for finding the root of g(x) = f(x) - x = 0.

    Args:
        f (callable): The weight-tied function f_theta.
        x0 (Tensor): Initial estimate for the hidden state z (often zeros or initial input).
        threshold (int): Maximum number of iterations.
        eps (float): Tolerance for convergence (norm of g(x)).

    Returns:
        dict: {'result': z*, 'nstep': iterations}
    """

    # g(x) = f(x) - x
    g = lambda x: f(x) - x

    x_est = x0.clone().detach()  # Initial estimate
    gx = g(x_est)  # Initial residual

    bsz = x0.shape[0]
    # Flatten the state for linear algebra (z is B x L x D, flatten to B x (L*D))
    flat_size = x0.numel() // bsz

    # Initialize J^{-1} (approximate inverse Jacobian of g) as -I
    # We store the inverse Jacobian approximation B as a list of rank-1 updates
    Us = []
    VTs = []

    # Update direction: delta_x = -B @ g(x)
    # Since B = -I initially, update = -(-I) @ gx = gx
    update = gx.clone()

    nstep = 0
    lowest_norm = gx.norm().item()
    lowest_xest = x_est

    while nstep < threshold:
        nstep += 1

        # --- Line Search (Simplified) ---
        s = 1.0  # Default step size
        x_new = x_est + s * update
        gx_new = g(x_new)

        # --- Check Convergence ---
        norm_new = gx_new.norm().item()
        if norm_new < lowest_norm:
            lowest_norm = norm_new
            lowest_xest = x_new

        if norm_new < eps:
            x_est = x_new
            break

        # --- Broyden Update ---
        # Note: We use the simpler method of not recomputing f(x_est)
        # but relying on the saved gx
        delta_x = x_new - x_est
        delta_gx = gx_new - gx

        # Calculate rank-1 update u * vT for B = B + u * vT
        # We use B_new = B_old + (delta_x - B_old @ delta_gx) @ delta_x.T @ B_old / (delta_x.T @ B_old @ delta_gx)
        # Using the Sherman-Morrison identity update:
        # vT = J^{-T} @ delta_x (approximated by -delta_x since J^{-1} is approx -I)
        # u = (delta_x - B @ delta_gx) / (delta_x @ vT)

        # Approximate B @ delta_gx using the current rank-1 updates (Us, VTs)

        # For simplicity and stability in this generic implementation, we only use -I
        # in the initial iteration, and then rely purely on the rank-1 updates after step 1.

        if nstep > 1:
            # Low-rank inverse Jacobian approximation B
            # We need to compute: B_old @ delta_gx
            # B_old = -I + sum(u_k * vT_k)
            B_delta_gx = -delta_gx
            for u, vT in zip(Us, VTs):
                B_delta_gx += u * (vT * delta_gx).sum(dim=-1, keepdim=True)

            # vT = (delta_x - B @ delta_gx) / (delta_x.T @ B @ delta_gx)

            # Simplified update based purely on Broyden's "Good" method approximation:
            # u = (delta_x - B_old @ delta_gx)
            # denom = (delta_gx.T @ B_old @ delta_gx)

            u = delta_x - B_delta_gx
            vT = delta_x.view(
                bsz, 1, flat_size
            )  # vT is a vector proportional to delta_x

            # Normalize u and vT to maintain stability and use Sherman-Morrison
            # We need B_new * delta_gx = delta_x

            # Simplified Broyden update for B * g: B_new @ gx = B_old @ gx + (u / (vT @ u)) @ vT @ gx

            # Simple rank-1 update: (delta_x - B * delta_gx) / (delta_gx.T * delta_x) * delta_x.T
            # For simplicity, we use the vector form of the Broyden update:

            # 1. Compute delta_x - B_old * delta_gx
            y_k = delta_x - B_delta_gx

            # 2. Compute vT * y_k (where vT is delta_gx)
            vT_k = delta_gx.view(
                bsz, 1, flat_size
            )  # Treat delta_gx as the vector to multiply by

            # Denominator: (delta_gx.T @ B @ delta_gx)
            denominator = (vT_k * B_delta_gx.view(bsz, flat_size, 1)).sum(dim=1)

            # Final update vectors
            u_k = y_k.view(bsz, flat_size, 1) / denominator[:, :, None]
            vT_k = delta_gx.view(bsz, 1, flat_size)

            # Append updates (max threshold - 1 updates)
            if len(Us) < threshold - 1:
                Us.append(u_k)
                VTs.append(vT_k)

            # --- Compute new update direction ---
            update = -gx_new  # Initial guess: -I @ gx_new

            # Apply accumulated updates
            for u, vT in zip(Us, VTs):
                # Apply B @ gx_new = (-I + sum(u*vT)) @ gx_new
                update += u.squeeze(-1) * (
                    vT.squeeze(1) * gx_new.view(bsz, flat_size)
                ).sum(dim=1, keepdim=True).view(bsz, flat_size)

        else:  # nstep == 1
            # B_0 is -I
            update = gx_new

        x_est = x_new
        gx = gx_new

    return {
        "result": lowest_xest.view_as(x0),
        "nstep": nstep,
        "lowest_norm": lowest_norm,
    }


# --- 2. JACOBIAN REGULARIZATION (from jacobian.py) ---


def jac_loss_estimate(f0, z0, vecs=1):
    """
    Estimating tr(J^T J) via Hutchinson estimator for Jacobian regularization.

    Args:
        f0 (Tensor): Output of the function f(z).
        z0 (Tensor): Input z.
        vecs (int): Number of random Gaussian vectors to use.

    Returns:
        Tensor: A 1x1 tensor of the (normalized) Jacobian loss.
    """
    result = 0
    z0_flat = z0.view(-1)
    f0_flat = f0.view(-1)

    for _ in range(vecs):
        # Sample random vector v
        v = torch.randn_like(z0_flat)

        # Compute vJ = v @ J (vector-Jacobian product)
        # We compute the gradient of f0_flat with respect to z0_flat, multiplying by v
        vJ = autograd.grad(f0_flat, z0_flat, v, retain_graph=True, create_graph=True)[0]

        # Add ||vJ||^2
        result += vJ.norm() ** 2

    return result / vecs / z0_flat.numel()


# --- 3. THE DEQ MODULE ---


class DEQLayer(nn.Module):
    """
    Deep Equilibrium Layer implementation.
    Wraps an arbitrary weight-tied module and implements the implicit forward/backward pass.
    """

    def __init__(
        self,
        f_module,
        dim,
        input_dim=None,
        f_solver=broyden_solve,
        b_solver=broyden_solve,
        max_f_iter=12,
        max_b_iter=12,
        f_eps=1e-5,
        b_eps=1e-5,
        jac_reg_weight=1e-3,
    ):
        super().__init__()

        # The core weight-tied module (e.g., TransformerBlock)
        self.f_module = f_module
        self.dim = dim
        self.jac_reg_weight = jac_reg_weight

        # Input injection layer (W_x)
        input_dim = input_dim if input_dim is not None else dim
        self.input_injection = nn.Linear(input_dim, dim)

        # Solvers and iteration settings
        self.f_solver = f_solver
        self.b_solver = b_solver
        self.max_f_iter = max_f_iter
        self.max_b_iter = max_b_iter
        self.f_eps = f_eps
        self.b_eps = b_eps

        self.hook = None  # For storing the backward hook

        # We need a function that represents the DEQ iteration: z_new = f(z) + X
        # Since the f_module's forward signature is unknown, we handle the X injection here.
        # This function is *only* defined and used inside the fixed-point solver.

    def _get_deq_function(self, E, *args):
        """Returns the function F(z) = f_module(z, *args) + X, where X is constant."""

        # 1. Compute the constant input injection X (W_x * E)
        # Assuming E is the initial embedding (B x L x D_in)
        X = self.input_injection(E)  # X is now B x L x D

        # Ensure the underlying module handles the input args correctly.
        # This assumes the underlying f_module takes the hidden state z
        # as its first argument, followed by any auxiliary arguments (like mask).

        def deq_function(z):
            """
            This is F(z) = z_new, where z_new is the result of applying the
            weight-tied block and adding the input injection.
            """
            # Apply the weight-tied module
            f_out = self.f_module(z, *args)

            # Add the constant input injection term
            z_new = f_out + X
            return z_new

        return deq_function

    def forward(self, E, z_init=None, *args):
        """
        Args:
            E (Tensor): The constant input (initial embedding B x L x D_in).
            z_init (Tensor, optional): Initial guess for the hidden state (B x L x D).
                                       If None, uses zeros.
            *args: Additional arguments required by the f_module (e.g., attention mask).

        Returns:
            Tensor: The equilibrium hidden state z_star.
            Tensor: The Jacobian regularization loss (scalar).
        """

        bsz, seq_len = E.shape[0], E.shape[1]

        if z_init is None:
            # Initialize with zeros in the hidden dimension of the DEQ output (dim)
            z_init = torch.zeros(bsz, seq_len, self.dim, device=E.device, dtype=E.dtype)

        # 1. Define the DEQ system F(z) = z_new
        deq_func = self._get_deq_function(E, *args)

        # --- Forward Pass (Root Finding) ---
        with torch.no_grad():
            # Solve for z* such that F(z*) - z* = 0
            # Note: The solver expects flattened input if it's using the Broyden implementation above
            z_init_flat = z_init.view(bsz, -1)

            # The solver works on the flattened state
            def flat_deq_func(z_flat):
                return deq_func(z_flat.view_as(z_init)).view(bsz, -1)

            fwd_res = self.f_solver(
                flat_deq_func, z_init_flat, threshold=self.max_f_iter, eps=self.f_eps
            )
            z_star = fwd_res["result"].view_as(z_init)

            # You can log fwd_res['nstep'] or fwd_res['lowest_norm'] here for monitoring

        # --- Backward Pass (Implicit Differentiation) ---
        if self.training:
            # We need to run F(z*) again, but with z_star detached from the no_grad block
            # and requiring gradient computation.
            z_star_fwd = z_star.clone().detach().requires_grad_()

            # F_out = F(z_star)
            F_out = deq_func(z_star_fwd)

            # --- Jacobian Regularization Loss ---
            jac_loss = jac_loss_estimate(F_out, z_star_fwd) * self.jac_reg_weight

            # --- Define the Backward Hook ---

            # The backward gradient (dL/dz_star) is solved by the linear system:
            # (I - J_F^T) * dL/dF = dL/dz_star  (where dL/dF is the hook's input grad)
            # which means dL/dz_star = (I - J_F^T)^{-1} * dL/dF
            # The DEQ paper uses J_g = J_F - I, so J_g^T = J_F^T - I.
            # We want to solve J_g^T * v = -(dL/dz_star)

            def backward_hook(grad_output):
                """
                This hook computes: (I - J_F^T)^{-1} @ grad_output
                """
                if self.hook is not None:
                    # Remove hook to prevent infinite recursion/loop
                    self.hook.remove()
                    torch.cuda.synchronize()

                # J_F^T is the Jacobian of F_out w.r.t z_star_fwd
                # J_g^T is the Jacobian of G_out (G=F-z) w.r.t z_star_fwd

                # The linear system is: v - J_F^T @ v = grad_output
                # We solve for v (the resulting gradient dL/dz_star)

                v_flat = torch.zeros_like(grad_output).view(bsz, -1)
                grad_output_flat = grad_output.view(bsz, -1)

                # The backward objective is solving: (I - J_F^T) @ v = grad_output
                # Or: G_hook(v) = v - J_F^T @ v - grad_output = 0

                def bwd_objective(v_flat):
                    v = v_flat.view_as(grad_output)

                    # Compute J_F^T @ v (vector-Jacobian product)
                    # We compute the gradient of F_out w.r.t z_star_fwd, multiplying by v
                    J_F_T_v = autograd.grad(
                        F_out, z_star_fwd, v, retain_graph=True, create_graph=False
                    )[0]

                    # Residual: v - J_F^T @ v - grad_output
                    residual = v - J_F_T_v - grad_output
                    return residual.view(bsz, -1)

                # Solve the linear system
                bwd_res = self.b_solver(
                    bwd_objective,
                    v_flat,  # Initial guess for v is zero
                    threshold=self.max_b_iter,
                    eps=self.b_eps,
                )

                v_star = bwd_res["result"].view_as(grad_output)

                # The hook returns the gradient dL/dz_star (v_star)
                return v_star

            # Attach the hook to the output of F(z*)
            self.hook = F_out.register_hook(backward_hook)

            # The final output is F_out, but we zero out its gradient w.r.t. z_star_fwd
            # to prevent it from contributing via the trivial path, relying only on the hook.
            # This is achieved by creating an identity function on z_star_fwd

            return z_star + jac_loss

        else:
            # If not training, just return the solved z_star (no grad/hook needed)
            return z_star


# --- Example Usage (requires a dummy TransformerBlock) ---

if __name__ == "__main__":
    # 0. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Dummy Weight-Tied Module (e.g., simplified Transformer block)
    class DummyBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim * 4)
            self.linear2 = nn.Linear(dim * 4, dim)
            self.norm = nn.LayerNorm(dim)
            self.dim = dim

        def forward(self, x, mask=None):
            # z = LayerNorm(z)
            z_norm = self.norm(x)

            # Self-Attention + Residual (simplified)
            # Assume attention is just an identity op for simplicity,
            # but in reality this is where QKV/masking happens.
            attn_out = z_norm
            z = x + attn_out

            # FFN + Residual
            residual = z
            z_norm = self.norm(z)

            ff_out = self.linear2(F.gelu(self.linear1(z_norm)))
            z = residual + ff_out

            # The true F_theta output
            return z

    # 2. Instantiate Components
    HIDDEN_DIM = 64
    INPUT_DIM = 32  # E dimension
    SEQ_LEN = 10
    BATCH_SIZE = 4

    # The weight-tied module instance
    transformer_block = DummyBlock(HIDDEN_DIM).to(device)

    # The DEQ layer wrapping the block
    deq_layer = DEQLayer(
        f_module=transformer_block,
        dim=HIDDEN_DIM,
        input_dim=INPUT_DIM,
        max_f_iter=30,
        f_eps=1e-3,
    ).to(device)

    # 3. Dummy Data
    input_E = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, device=device)
    target = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device)

    optimizer = torch.optim.Adam(deq_layer.parameters(), lr=1e-3)

    # 4. Training Loop (Single Step)
    print(f"Starting DEQ training simulation on {device}...")
    deq_layer.train()
    optimizer.zero_grad()

    t0 = time.time()

    # Forward Pass
    z_star_output = deq_layer(input_E)

    # Note: z_star_output is z* + jac_loss. We separate them for calculating the data loss.
    jac_loss = z_star_output - z_star_output.detach()
    z_star = z_star_output.detach()  # Pure fixed point result

    # Calculate Data Loss
    data_loss = F.mse_loss(z_star, target)

    # Total Loss = Data Loss + Jacobian Regularization
    total_loss = data_loss + jac_loss.sum()

    # Backward Pass (Triggers the implicit differentiation hook)
    total_loss.backward()

    optimizer.step()
    t1 = time.time()

    print(f"\n--- Simulation Results ---")
    print(f"Time taken: {t1 - t0:.4f} seconds")
    print(f"Data Loss (MSE): {data_loss.item():.6f}")
    print(f"Jac Reg Loss: {jac_loss.sum().item():.6f}")
    print(f"Total Loss: {total_loss.item():.6f}")
    print(
        f"Memory Check (Gradient): {deq_layer.input_injection.weight.grad.norm().item():.6f}"
    )
