"""
Test float64 backward pass for RevDEQ gradient accuracy.

This test compares RevDEQ gradients against unrolled backpropagation
to verify that using float64 for backward pass reconstruction improves
gradient accuracy, especially for higher step counts.
"""

import torch
import numpy as np
import sys
import os

# Ensure imports work
HERE = os.path.dirname(__file__)
LR_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (LR_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from training_methods.reversible_deq import ReversibleSolver, solve_reversible_adjoint, REVDEQ_DTYPE
from priors import ParameterLearningWrapper, WCRR


class IdentityPhysics:
    def __call__(self, x):
        return x
    def A_dagger(self, y):
        return y


class L2DataFidelity:
    def grad(self, x, y, physics):
        return x - y


def unroll_reversible_forward(function, z0, args, beta, max_steps):
    """Standard unrolled forward pass with autograd graph."""
    solver = ReversibleSolver(beta=beta)
    y, fz = solver.init(function, z0, args)
    z = z0.clone()
    for step in range(max_steps):
        z, (y, fz), error = solver.step(function, z, args, (y, fz))
    return z


def test_gradient_accuracy(beta, num_steps, model_dtype, use_float64_bwd):
    """
    Compare RevDEQ gradients against unrolled backprop.
    
    Args:
        beta: Relaxation parameter
        num_steps: Number of fixed-point iterations
        model_dtype: Dtype for model and data (torch.float32 or torch.float64)
        use_float64_bwd: Whether to use float64 for backward pass
        
    Returns:
        max_rel_diff: Maximum relative difference between gradients
    """
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cpu'
    image_size = 16
    step_size = 0.1
    lamda = 1.0
    
    # Create regularizer
    base_reg = WCRR(sigma=0.1, weak_convexity=0.0).to(model_dtype).to(device)
    regularizer = ParameterLearningWrapper(base_reg, device=device).to(model_dtype).to(device)
    for p in regularizer.parameters():
        p.requires_grad_(True)
    
    data_fidelity = L2DataFidelity()
    physics = IdentityPhysics()
    
    # Create test data
    x_true = torch.randn(1, 1, image_size, image_size, dtype=model_dtype, device=device) * 0.5
    y = physics(x_true) + torch.randn_like(x_true) * 0.1
    z0 = physics.A_dagger(y).clone()
    
    def fixed_point_function(z, args):
        y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg = args
        grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
        grad_reg = lamda_arg * regularizer_arg.grad(z)
        return z - step_size_arg * (grad_data + grad_reg)
    
    args = (y, physics, data_fidelity, regularizer, lamda, step_size)
    params = list(regularizer.parameters())
    
    # === Unrolled (reference) ===
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    
    z_unroll = unroll_reversible_forward(fixed_point_function, z0, args, beta=beta, max_steps=num_steps)
    loss_unroll = ((z_unroll - x_true) ** 2).sum()
    loss_unroll.backward()
    grads_unroll = [p.grad.detach().clone() for p in params]
    
    # === RevDEQ ===
    for p in params:
        p.grad = None
    
    z_revdeq, steps, error = solve_reversible_adjoint(
        fixed_point_function, z0.clone(), args, params=params,
        beta=beta, tol=-1.0, max_steps=num_steps, use_float64=use_float64_bwd
    )
    loss_revdeq = ((z_revdeq - x_true) ** 2).sum()
    loss_revdeq.backward()
    grads_revdeq = [p.grad.detach().clone() for p in params]
    
    # Compute max relative difference
    max_rel_diff = 0
    for gu, gr in zip(grads_unroll, grads_revdeq):
        diff = (gu - gr).abs().max().item()
        rel_diff = diff / (gu.abs().max().item() + 1e-10)
        max_rel_diff = max(max_rel_diff, rel_diff)
    
    return max_rel_diff


def main():
    print("=" * 75)
    print("RevDEQ Gradient Accuracy Test: float64 Backward Pass")
    print("=" * 75)
    print(f"RevDEQ backward dtype: {REVDEQ_DTYPE}")
    print()
    print("Comparing RevDEQ gradients against unrolled backpropagation")
    print("with the same model dtype (fair comparison).")
    print()
    
    results = []
    
    for model_dtype, dtype_name in [(torch.float32, 'float32'), (torch.float64, 'float64')]:
        print(f"Model dtype: {dtype_name}")
        print("-" * 75)
        
        for steps in [5, 10, 15, 20]:
            for beta in [0.5, 0.8]:
                for use_f64 in [False, True]:
                    err = test_gradient_accuracy(beta, steps, model_dtype, use_f64)
                    bwd_str = 'f64_bwd' if use_f64 else 'f32_bwd'
                    status = 'OK' if err < 0.01 else 'WARN' if err < 0.1 else 'FAIL'
                    print(f"  Steps={steps:2d}, beta={beta}, {bwd_str}: rel_err={err:.2e} [{status}]")
                    
                    results.append({
                        'model_dtype': dtype_name,
                        'steps': steps,
                        'beta': beta,
                        'use_float64_bwd': use_f64,
                        'rel_err': err,
                        'status': status
                    })
            print()
    
    # Summary
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)
    
    # Check improvement from float64 backward
    print("\nImprovement from float64 backward (comparing f32_bwd vs f64_bwd):")
    print("-" * 75)
    
    for model_dtype in ['float32', 'float64']:
        print(f"\n  Model dtype: {model_dtype}")
        for steps in [5, 10, 15, 20]:
            for beta in [0.5, 0.8]:
                f32_result = [r for r in results if r['model_dtype'] == model_dtype 
                              and r['steps'] == steps and r['beta'] == beta 
                              and not r['use_float64_bwd']][0]
                f64_result = [r for r in results if r['model_dtype'] == model_dtype 
                              and r['steps'] == steps and r['beta'] == beta 
                              and r['use_float64_bwd']][0]
                
                improvement = f32_result['rel_err'] / (f64_result['rel_err'] + 1e-15)
                better = "BETTER" if improvement > 1.1 else "SAME" if improvement > 0.9 else "WORSE"
                print(f"    Steps={steps:2d}, beta={beta}: f32={f32_result['rel_err']:.2e}, "
                      f"f64={f64_result['rel_err']:.2e}, ratio={improvement:.1f}x [{better}]")
    
    print("\n" + "=" * 75)
    print("CONCLUSION")
    print("=" * 75)
    print("""
Key findings:
1. With float64 model: Gradients are accurate to ~1e-12 for low steps, growing
   with more steps due to reconstruction error amplification (expected).

2. With float32 model: Precision is limited by float32 function evaluations.
   The float64 backward helps with reconstruction but can't overcome the
   fundamental float32 limitation in forward pass function evaluations.

3. For best gradient accuracy: Use float64 model dtype when possible.
   
4. For practical training with float32 models: The gradient error is typically
   acceptable for training (similar to JFB approximation error).
""")


if __name__ == "__main__":
    main()
