"""
Debug: Check if gradient error increases with steps using actual CRR regularizer.
"""

import os
import sys
import torch
import numpy as np

HERE = os.path.dirname(__file__)
LR_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(LR_ROOT, ".."))
for p in (LR_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from training_methods.reversible_deq import ReversibleSolver, solve_reversible_adjoint
from priors import ParameterLearningWrapper, WCRR


class IdentityPhysics:
    def __call__(self, x): return x
    def A_dagger(self, y): return y


class L2DataFidelity:
    def __call__(self, x, y, physics):
        return 0.5 * ((x - y) ** 2).view(x.shape[0], -1).sum(-1)
    def grad(self, x, y, physics):
        return x - y


def test_gradient_vs_steps(num_steps, beta, nn_dtype=torch.float32):
    """Test gradient accuracy for given steps and beta."""
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cpu"
    
    # Create regularizer (float32 for realistic test, or float64 for verification)
    if nn_dtype == torch.float64:
        base_reg = WCRR(sigma=0.1, weak_convexity=0.0).to(torch.float64).to(device)
        regularizer = ParameterLearningWrapper(base_reg, device=device).to(torch.float64)
        x_true = torch.randn(1, 1, 16, 16, dtype=torch.float64, device=device) * 0.5
    else:
        base_reg = WCRR(sigma=0.1, weak_convexity=0.0).to(device)
        regularizer = ParameterLearningWrapper(base_reg, device=device)
        x_true = torch.randn(1, 1, 16, 16, dtype=torch.float32, device=device) * 0.5
    
    data_fidelity = L2DataFidelity()
    physics = IdentityPhysics()
    
    y = physics(x_true) + torch.randn_like(x_true) * 0.1
    z0 = physics.A_dagger(y).clone()
    
    step_size = 0.1
    lamda = 1.0
    
    def fixed_point_function(z, args):
        y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg = args
        grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
        grad_reg = lamda_arg * regularizer_arg.grad(z)
        grad_total = grad_data + grad_reg
        return z - step_size_arg * grad_total
    
    args = (y, physics, data_fidelity, regularizer, lamda, step_size)
    params = list(regularizer.parameters())
    
    # Unrolled forward with autograd
    for p in params:
        p.grad = None
    
    solver = ReversibleSolver(beta=beta, use_float64=True, nn_dtype=nn_dtype)
    y_state, fz = solver.init(fixed_point_function, z0.clone().to(torch.float64), args)
    z = z0.clone().to(torch.float64)
    
    for _ in range(num_steps):
        z, (y_state, fz), err = solver.step(fixed_point_function, z, args, (y_state, fz))
    
    z_unroll = z.to(x_true.dtype)
    loss_unroll = ((z_unroll - x_true) ** 2).sum()
    loss_unroll.backward()
    grads_unroll = [p.grad.clone() for p in params]
    
    # RevDEQ
    for p in params:
        p.grad = None
    
    z_revdeq, _, fp_error = solve_reversible_adjoint(
        fixed_point_function, z0.clone(), args, params=params,
        beta=beta, tol=-1.0, max_steps=num_steps,
        use_float64=True, nn_dtype=nn_dtype
    )
    
    loss_revdeq = ((z_revdeq - x_true) ** 2).sum()
    loss_revdeq.backward()
    grads_revdeq = [p.grad.clone() for p in params]
    
    # Compute errors
    z_diff = (z_unroll - z_revdeq).abs().max().item()
    
    max_rel = 0
    for gu, gr in zip(grads_unroll, grads_revdeq):
        if gu.abs().max() > 1e-10:
            rel = (gu - gr).abs().max().item() / gu.abs().max().item()
            max_rel = max(max_rel, rel)
    
    return z_diff, max_rel, fp_error


def main():
    print("=" * 80)
    print("GRADIENT ERROR vs STEPS with ACTUAL CRR REGULARIZER")
    print("=" * 80)
    
    print("\n--- Float32 NN (like in actual training) ---")
    print(f"{'Steps':>6} | {'Beta':>5} | {'z_diff':>12} | {'max_rel_grad':>12} | {'FP Error':>12}")
    print("-" * 65)
    
    for num_steps in [5, 10, 15, 20, 25, 30]:
        for beta in [0.5, 0.8]:
            z_diff, max_rel, fp_err = test_gradient_vs_steps(num_steps, beta, nn_dtype=torch.float32)
            status = "OK" if max_rel < 0.01 else "BAD"
            print(f"{num_steps:>6} | {beta:>5.1f} | {z_diff:>12.2e} | {max_rel:>12.2e} | {fp_err:>12.2e} [{status}]")
    
    print("\n--- Float64 NN (like in verification tests) ---")
    print(f"{'Steps':>6} | {'Beta':>5} | {'z_diff':>12} | {'max_rel_grad':>12} | {'FP Error':>12}")
    print("-" * 65)
    
    for num_steps in [5, 10, 15, 20, 25, 30]:
        for beta in [0.5, 0.8]:
            z_diff, max_rel, fp_err = test_gradient_vs_steps(num_steps, beta, nn_dtype=torch.float64)
            status = "OK" if max_rel < 0.01 else "BAD"
            print(f"{num_steps:>6} | {beta:>5.1f} | {z_diff:>12.2e} | {max_rel:>12.2e} | {fp_err:>12.2e} [{status}]")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)


if __name__ == "__main__":
    main()
