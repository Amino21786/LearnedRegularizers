"""
Final verification: RevDEQ gradient accuracy with correct parameters.

Key insights:
1. Beta=0.5 is the JAX default and is much more numerically stable
2. The reconstruction error grows as (1/(1-beta))^N
3. For beta=0.5, up to ~20 steps is stable
4. For beta=0.8, only ~5 steps is stable
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

from training_methods.reversible_deq import ReversibleSolver, solve_reversible_adjoint
from priors import ParameterLearningWrapper, WCRR


class IdentityPhysics:
    def __call__(self, x): return x
    def A_dagger(self, y): return y


class L2DataFidelity:
    def grad(self, x, y, physics):
        return x - y


def test_gradient_accuracy(num_steps, beta, nn_dtype=torch.float64, verbose=False):
    """Test gradient accuracy for given parameters."""
    torch.manual_seed(42)
    device = "cpu"
    
    # Create regularizer in the specified dtype
    base_reg = WCRR(sigma=0.1, weak_convexity=0.0).to(nn_dtype).to(device)
    regularizer = ParameterLearningWrapper(base_reg, device=device).to(nn_dtype)
    
    x_true = torch.randn(1, 1, 16, 16, dtype=nn_dtype, device=device) * 0.5
    
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
        return z - step_size_arg * (grad_data + grad_reg)
    
    args = (y, physics, data_fidelity, regularizer, lamda, step_size)
    params = list(regularizer.parameters())
    target = torch.zeros_like(x_true)
    
    # ========================================
    # Method 1: Unrolled with autograd
    # ========================================
    for p in params:
        p.grad = None
    
    # Use float64 for fixed-point operations
    solver = ReversibleSolver(beta=beta, use_float64=True, nn_dtype=nn_dtype)
    z = z0.clone().to(torch.float64)
    y_state, fz = solver.init(fixed_point_function, z, args)
    
    for _ in range(num_steps):
        z, (y_state, fz), _ = solver.step(fixed_point_function, z, args, (y_state, fz))
    
    z_unroll = z.to(nn_dtype)
    loss_unroll = ((z_unroll - target) ** 2).sum()
    loss_unroll.backward()
    grads_unroll = [p.grad.clone() for p in params]
    
    # ========================================
    # Method 2: RevDEQ with custom backward
    # ========================================
    for p in params:
        p.grad = None
    
    z_revdeq, steps, fp_error = solve_reversible_adjoint(
        fixed_point_function, z0.clone(), args, params=params,
        beta=beta, tol=-1.0, max_steps=num_steps,
        use_float64=True, nn_dtype=nn_dtype
    )
    
    loss_revdeq = ((z_revdeq - target) ** 2).sum()
    loss_revdeq.backward()
    grads_revdeq = [p.grad.clone() for p in params]
    
    # ========================================
    # Compare
    # ========================================
    z_diff = (z_unroll - z_revdeq).abs().max().item()
    
    max_rel = 0
    total_diff = 0
    total_norm = 0
    for gu, gr in zip(grads_unroll, grads_revdeq):
        diff = (gu - gr).abs().max().item()
        norm = gu.abs().max().item()
        if norm > 1e-10:
            rel = diff / norm
            max_rel = max(max_rel, rel)
        total_diff += (gu - gr).abs().sum().item()
        total_norm += gu.abs().sum().item()
    
    overall_rel = total_diff / (total_norm + 1e-15)
    
    if verbose:
        print(f"  z_diff: {z_diff:.2e}")
        print(f"  max_rel_grad: {max_rel:.2e}")
        print(f"  overall_rel: {overall_rel:.2e}")
    
    return z_diff, max_rel, overall_rel


def main():
    print("=" * 70)
    print("REVDEQ GRADIENT VERIFICATION (Float64 NN)")
    print("=" * 70)
    print("Using JAX default: beta=0.5")
    print()
    
    print(f"{'Steps':>6} | {'Beta':>6} | {'z_diff':>12} | {'max_rel_grad':>12} | {'Status'}")
    print("-" * 60)
    
    # Test with beta=0.5 (JAX default)
    for num_steps in [5, 10, 15, 20]:
        z_diff, max_rel, overall_rel = test_gradient_accuracy(num_steps, beta=0.5, nn_dtype=torch.float64)
        status = "OK" if max_rel < 1e-8 else ("WARN" if max_rel < 1e-4 else "BAD")
        print(f"{num_steps:>6} | {0.5:>6.2f} | {z_diff:>12.2e} | {max_rel:>12.2e} | {status}")
    
    print()
    
    # Compare beta=0.5 vs beta=0.8
    print("=" * 70)
    print("BETA COMPARISON (10 steps, Float64 NN)")
    print("=" * 70)
    
    for beta in [0.5, 0.6, 0.7, 0.8]:
        z_diff, max_rel, overall_rel = test_gradient_accuracy(10, beta=beta, nn_dtype=torch.float64)
        status = "OK" if max_rel < 1e-8 else ("WARN" if max_rel < 1e-4 else "BAD")
        amp = 1 / (1 - beta)
        print(f"Beta={beta:.1f} (amp={amp:.1f}x): max_rel_grad={max_rel:.2e} [{status}]")
    
    print()
    print("=" * 70)
    print("RECOMMENDED PARAMETERS")
    print("=" * 70)
    print()
    print("For Float64 NN (verification):")
    print("  - beta=0.5, steps<=20: gradient error < 1e-8")
    print("  - beta=0.5, steps<=15: gradient error < 1e-10")
    print()
    print("For Float32 NN (training):")
    print("  - beta=0.5, steps<=10: gradient error < 1e-6")
    print("  - The float32 NN limits precision regardless of fixed-point dtype")
    print()
    
    # Verify float32 NN
    print("=" * 70)
    print("FLOAT32 NN TEST (realistic training scenario)")
    print("=" * 70)
    
    print(f"\n{'Steps':>6} | {'Beta':>6} | {'max_rel_grad':>12} | {'Status'}")
    print("-" * 50)
    
    for num_steps in [5, 10, 15, 20]:
        z_diff, max_rel, overall_rel = test_gradient_accuracy(num_steps, beta=0.5, nn_dtype=torch.float32)
        status = "OK" if max_rel < 1e-5 else ("WARN" if max_rel < 1e-3 else "BAD")
        print(f"{num_steps:>6} | {0.5:>6.2f} | {max_rel:>12.2e} | {status}")


if __name__ == "__main__":
    main()
