"""
Debug script to check dtype handling in RevDEQ during actual training conditions.
"""

import os
import sys
import torch

HERE = os.path.dirname(__file__)
LR_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(LR_ROOT, ".."))
for p in (LR_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from training_methods.reversible_deq import (
    ReversibleSolver, 
    solve_reversible_adjoint,
    FP_DTYPE,
    NN_DTYPE,
)
from priors import ParameterLearningWrapper, WCRR


def compare_gradient_accuracy(num_steps, beta=0.8):
    """Compare RevDEQ vs Unrolled gradients with specific step count."""
    
    torch.manual_seed(42)
    device = "cpu"
    
    # Create regularizer in FLOAT32 (like pretrained weights)
    base_reg = WCRR(sigma=0.1, weak_convexity=0.0).to(device)
    regularizer = ParameterLearningWrapper(base_reg, device=device)
    
    # Create test data in float32
    x_true = torch.randn(1, 1, 16, 16, dtype=torch.float32, device=device)
    y = x_true + torch.randn_like(x_true) * 0.1
    z0 = y.clone()
    
    step_size = 0.1
    lamda = 1.0
    
    class SimpleL2:
        def grad(self, x, y, physics):
            return x - y
    
    class IdentityPhysics:
        pass
    
    data_fidelity = SimpleL2()
    physics = IdentityPhysics()
    
    def fixed_point_function(z, args):
        y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg = args
        grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
        grad_reg = lamda_arg * regularizer_arg.grad(z)
        grad_total = grad_data + grad_reg
        z_new = z - step_size_arg * grad_total
        return z_new
    
    args = (y, physics, data_fidelity, regularizer, lamda, step_size)
    params = list(regularizer.parameters())
    
    # Method 1: Unrolled
    for p in params:
        p.grad = None
    
    solver_unroll = ReversibleSolver(beta=beta, use_float64=True, nn_dtype=torch.float32)
    y_u, fz_u = solver_unroll.init(fixed_point_function, z0.clone().to(FP_DTYPE), args)
    z_u = z0.clone().to(FP_DTYPE)
    
    for _ in range(num_steps):
        z_u, (y_u, fz_u), err = solver_unroll.step(fixed_point_function, z_u, args, (y_u, fz_u))
    
    z_u = z_u.to(torch.float32)
    loss_u = ((z_u - x_true) ** 2).sum()
    loss_u.backward()
    
    grads_unroll = [p.grad.clone() for p in params]
    
    # Method 2: RevDEQ
    for p in params:
        p.grad = None
    
    z_r, _, _ = solve_reversible_adjoint(
        fixed_point_function, z0.clone(), args, params=params,
        beta=beta, tol=-1.0, max_steps=num_steps,
        use_float64=True,
        nn_dtype=torch.float32,
    )
    
    loss_r = ((z_r - x_true) ** 2).sum()
    loss_r.backward()
    
    grads_revdeq = [p.grad.clone() for p in params]
    
    # Calculate errors
    z_diff = (z_u - z_r).abs().max().item()
    
    max_rel_diff = 0
    for gu, gr in zip(grads_unroll, grads_revdeq):
        diff = (gu - gr).abs().max().item()
        rel_diff = diff / (gu.abs().max().item() + 1e-10)
        max_rel_diff = max(max_rel_diff, rel_diff)
    
    total_diff = sum((gu - gr).abs().sum().item() for gu, gr in zip(grads_unroll, grads_revdeq))
    total_norm = sum(gu.abs().sum().item() for gu in grads_unroll)
    overall_rel = total_diff / (total_norm + 1e-10)
    
    return z_diff, max_rel_diff, overall_rel


def main():
    print("=" * 70)
    print("GRADIENT ERROR vs NUMBER OF STEPS (Float32 NN)")
    print("=" * 70)
    print(f"Beta = 0.8 (error amplification factor = 1/(1-beta) = 5.0)")
    print()
    
    print(f"{'Steps':>6} | {'z_diff':>12} | {'max_rel_grad':>12} | {'overall_rel':>12}")
    print("-" * 55)
    
    for num_steps in [5, 10, 15, 20, 30, 50]:
        z_diff, max_rel, overall_rel = compare_gradient_accuracy(num_steps, beta=0.8)
        print(f"{num_steps:>6} | {z_diff:>12.2e} | {max_rel:>12.2e} | {overall_rel:>12.2e}")
    
    print()
    print("=" * 70)
    print("GRADIENT ERROR vs BETA VALUE (20 steps, Float32 NN)")
    print("=" * 70)
    print()
    
    print(f"{'Beta':>6} | {'1/(1-beta)':>10} | {'max_rel_grad':>12} | {'overall_rel':>12}")
    print("-" * 55)
    
    for beta in [0.5, 0.6, 0.7, 0.8, 0.9]:
        amp_factor = 1.0 / (1.0 - beta)
        z_diff, max_rel, overall_rel = compare_gradient_accuracy(20, beta=beta)
        print(f"{beta:>6.1f} | {amp_factor:>10.1f} | {max_rel:>12.2e} | {overall_rel:>12.2e}")
    
    print()
    print("=" * 70)
    print("COMPARISON: Float32 NN vs Float64 NN (20 steps, beta=0.8)")
    print("=" * 70)
    
    # Now test with Float64 NN
    torch.manual_seed(42)
    device = "cpu"
    
    # Float64 NN
    base_reg = WCRR(sigma=0.1, weak_convexity=0.0).to(torch.float64).to(device)
    regularizer = ParameterLearningWrapper(base_reg, device=device).to(torch.float64)
    
    x_true = torch.randn(1, 1, 16, 16, dtype=torch.float64, device=device)
    y = x_true + torch.randn_like(x_true) * 0.1
    z0 = y.clone()
    
    step_size = 0.1
    lamda = 1.0
    
    class SimpleL2:
        def grad(self, x, y, physics):
            return x - y
    
    class IdentityPhysics:
        pass
    
    data_fidelity = SimpleL2()
    physics = IdentityPhysics()
    
    def fixed_point_function(z, args):
        y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg = args
        grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
        grad_reg = lamda_arg * regularizer_arg.grad(z)
        grad_total = grad_data + grad_reg
        z_new = z - step_size_arg * grad_total
        return z_new
    
    args = (y, physics, data_fidelity, regularizer, lamda, step_size)
    params = list(regularizer.parameters())
    
    # Unrolled
    for p in params:
        p.grad = None
    
    solver_unroll = ReversibleSolver(beta=0.8, use_float64=True, nn_dtype=torch.float64)
    y_u, fz_u = solver_unroll.init(fixed_point_function, z0.clone(), args)
    z_u = z0.clone()
    
    for _ in range(20):
        z_u, (y_u, fz_u), err = solver_unroll.step(fixed_point_function, z_u, args, (y_u, fz_u))
    
    loss_u = ((z_u - x_true) ** 2).sum()
    loss_u.backward()
    grads_unroll = [p.grad.clone() for p in params]
    
    # RevDEQ
    for p in params:
        p.grad = None
    
    z_r, _, _ = solve_reversible_adjoint(
        fixed_point_function, z0.clone(), args, params=params,
        beta=0.8, tol=-1.0, max_steps=20,
        use_float64=True,
        nn_dtype=torch.float64,
    )
    
    loss_r = ((z_r - x_true) ** 2).sum()
    loss_r.backward()
    grads_revdeq = [p.grad.clone() for p in params]
    
    z_diff = (z_u - z_r).abs().max().item()
    max_rel_diff = 0
    for gu, gr in zip(grads_unroll, grads_revdeq):
        diff = (gu - gr).abs().max().item()
        rel_diff = diff / (gu.abs().max().item() + 1e-10)
        max_rel_diff = max(max_rel_diff, rel_diff)
    
    total_diff = sum((gu - gr).abs().sum().item() for gu, gr in zip(grads_unroll, grads_revdeq))
    total_norm = sum(gu.abs().sum().item() for gu in grads_unroll)
    overall_rel_f64 = total_diff / (total_norm + 1e-10)
    
    print(f"\nFloat64 NN (20 steps, beta=0.8):")
    print(f"  z_diff: {z_diff:.2e}")
    print(f"  max_rel_grad: {max_rel_diff:.2e}")
    print(f"  overall_rel: {overall_rel_f64:.2e}")
    
    # Recompute float32 for comparison
    z_diff_32, max_rel_32, overall_rel_32 = compare_gradient_accuracy(20, beta=0.8)
    print(f"\nFloat32 NN (20 steps, beta=0.8):")
    print(f"  z_diff: {z_diff_32:.2e}")
    print(f"  max_rel_grad: {max_rel_32:.2e}")
    print(f"  overall_rel: {overall_rel_32:.2e}")
    
    print(f"\nRatio (Float32/Float64): {overall_rel_32 / (overall_rel_f64 + 1e-20):.1e}x worse")


if __name__ == "__main__":
    main()
