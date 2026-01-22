"""
Simple test to verify RevDEQ gradient accuracy against true unrolled backpropagation.

This uses a simple gradient descent loop (not nmAPG) to ensure clean autograd comparison.
"""

import os
import sys
import torch
import numpy as np

# Setup paths
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from training_methods.reversible_deq import ReversibleSolver, solve_reversible_adjoint
from priors import ParameterLearningWrapper, WCRR
import deepinv


def test_gradient_accuracy_simple():
    """
    Test RevDEQ gradient accuracy using a simple gradient descent loop.
    
    This is a clean test that:
    1. Uses simple gradient descent (not nmAPG) for ground truth
    2. Compares against RevDEQ's custom backward
    """
    print("=" * 70)
    print("TEST: RevDEQ Gradient Accuracy (Simple GD Baseline)")
    print("=" * 70)
    
    device = "cpu"  # Use CPU for reproducibility
    torch.manual_seed(42)
    
    # Create a simple regularizer
    reg = WCRR(sigma=0.1, weak_convexity=0.0).to(device)
    regularizer = ParameterLearningWrapper(reg, device=device)
    
    # Load pretrained weights if available
    pretrain_path = "weights/score_for_Denoising/CRR_score_training_for_Denoising.pt"
    if os.path.exists(pretrain_path):
        regularizer.load_state_dict(torch.load(pretrain_path, map_location=device, weights_only=True))
        print(f"Loaded pretrained weights from {pretrain_path}")
    
    # Enable gradients
    for p in regularizer.parameters():
        p.requires_grad_(True)
    
    # Setup simple denoising problem
    noise_level = 0.1
    physics = deepinv.physics.Denoising(
        noise_model=deepinv.physics.GaussianNoise(sigma=noise_level)
    )
    data_fidelity = deepinv.optim.L2()
    lmbd = 1.0
    
    # Create test image
    x_true = torch.rand(1, 1, 32, 32, device=device)
    y = physics(x_true)
    x_init = physics.A_dagger(y).detach()
    
    # Upper level loss
    upper_loss = lambda x_pred, x_gt: ((x_pred - x_gt) ** 2).sum()
    
    # Settings
    step_size = 0.1
    num_steps = 10
    beta = 0.8
    
    print(f"\nSettings: {num_steps} steps, step_size={step_size}, beta={beta}")
    print(f"Image size: {x_true.shape}")
    
    # =========================================================================
    # METHOD 1: Simple Gradient Descent with Full Autograd (Ground Truth)
    # =========================================================================
    print("\n--- Method 1: Full Unrolled Backprop (Simple GD) ---")
    
    # Zero gradients
    for p in regularizer.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    # Simple gradient descent with autograd graph
    z = x_init.clone()
    for step in range(num_steps):
        grad_data = data_fidelity.grad(z, y, physics)
        grad_reg = lmbd * regularizer.grad(z)
        grad_total = grad_data + grad_reg
        z = z - step_size * grad_total
    
    # Compute loss and backprop
    loss_unroll = upper_loss(z, x_true)
    loss_unroll.backward()
    
    # Collect gradients
    grads_unroll = []
    for p in regularizer.parameters():
        if p.grad is not None:
            grads_unroll.append(p.grad.detach().clone())
    grads_unroll_flat = torch.cat([g.flatten() for g in grads_unroll])
    
    print(f"  Loss: {loss_unroll.item():.6f}")
    print(f"  Gradient norm: {torch.norm(grads_unroll_flat).item():.6f}")
    
    # =========================================================================
    # METHOD 2: RevDEQ with Custom Backward
    # =========================================================================
    print("\n--- Method 2: RevDEQ (Custom Backward) ---")
    
    # Zero gradients
    for p in regularizer.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    # Define fixed-point function (same as simple GD step)
    # NOTE: We need to be careful about what's in args vs params
    # The backward pass computes VJPs w.r.t. params, but the function
    # closure captures regularizer through args
    def fixed_point_function(z, args):
        y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg = args
        grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
        grad_reg = lamda_arg * regularizer_arg.grad(z)
        grad_total = grad_data + grad_reg
        z_new = z - step_size_arg * grad_total
        return z_new
    
    args = (y, physics, data_fidelity, regularizer, lmbd, step_size)
    params = [p for p in regularizer.parameters() if p.requires_grad]
    
    print(f"\n  DEBUG: Number of params passed to RevDEQ: {len(params)}")
    print(f"  DEBUG: Param shapes: {[p.shape for p in params]}")
    
    # Quick sanity check: can autograd trace from function output to params?
    z_test = x_init.clone().requires_grad_(True)
    f_out = fixed_point_function(z_test, args)
    test_grads = torch.autograd.grad(f_out.sum(), params, allow_unused=True)
    non_none_grads = sum(1 for g in test_grads if g is not None)
    print(f"  DEBUG: Sanity check - {non_none_grads}/{len(params)} params have non-None grads from function")
    
    # Run RevDEQ
    z_revdeq, steps_taken, error = solve_reversible_adjoint(
        function=fixed_point_function,
        z0=x_init.clone(),
        args=args,
        params=params,
        beta=beta,
        tol=-1.0,  # Negative tol to force exact num_steps
        max_steps=num_steps,
    )
    
    # Compute loss and backprop through custom backward
    loss_revdeq = upper_loss(z_revdeq, x_true)
    loss_revdeq.backward()
    
    # Collect gradients
    grads_revdeq = []
    for p in regularizer.parameters():
        if p.grad is not None:
            grads_revdeq.append(p.grad.detach().clone())
    grads_revdeq_flat = torch.cat([g.flatten() for g in grads_revdeq])
    
    print(f"  Loss: {loss_revdeq.item():.6f}")
    print(f"  Gradient norm: {torch.norm(grads_revdeq_flat).item():.6f}")
    print(f"  Steps taken: {steps_taken}")
    
    # =========================================================================
    # METHOD 3: RevDEQ Forward with Unrolled Backward (Sanity Check)
    # =========================================================================
    print("\n--- Method 3: RevDEQ Forward + Unrolled Backward (Sanity Check) ---")
    
    # Zero gradients
    for p in regularizer.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    # Run the same reversible forward iterations WITH autograd enabled
    solver = ReversibleSolver(beta=beta)
    z = x_init.clone()
    y_state, fz = solver.init(fixed_point_function, z, args)
    
    for step in range(num_steps):
        z, (y_state, fz), _ = solver.step(fixed_point_function, z, args, (y_state, fz))
    
    # Compute loss and backprop (through unrolled reversible iterations)
    loss_revdeq_unroll = upper_loss(z, x_true)
    loss_revdeq_unroll.backward()
    
    # Collect gradients
    grads_revdeq_unroll = []
    for p in regularizer.parameters():
        if p.grad is not None:
            grads_revdeq_unroll.append(p.grad.detach().clone())
    grads_revdeq_unroll_flat = torch.cat([g.flatten() for g in grads_revdeq_unroll])
    
    print(f"  Loss: {loss_revdeq_unroll.item():.6f}")
    print(f"  Gradient norm: {torch.norm(grads_revdeq_unroll_flat).item():.6f}")
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("GRADIENT COMPARISON")
    print("=" * 70)
    
    def compare_gradients(name1, grad1, name2, grad2):
        cos_sim = torch.nn.functional.cosine_similarity(
            grad1.unsqueeze(0), grad2.unsqueeze(0)
        ).item()
        rel_l2 = (torch.norm(grad1 - grad2) / (torch.norm(grad1) + 1e-8)).item()
        norm_ratio = (torch.norm(grad2) / (torch.norm(grad1) + 1e-8)).item()
        max_abs_err = torch.max(torch.abs(grad1 - grad2)).item()
        
        print(f"\n{name1} vs {name2}:")
        print(f"  Cosine Similarity: {cos_sim:.6f}")
        print(f"  Relative L2 Error: {rel_l2:.6f}")
        print(f"  Norm Ratio: {norm_ratio:.6f}")
        print(f"  Max Abs Error: {max_abs_err:.6e}")
        
        return cos_sim, rel_l2, norm_ratio
    
    # Compare RevDEQ custom backward vs unrolled simple GD
    cos1, rel1, norm1 = compare_gradients(
        "Simple GD (Full Backprop)", grads_unroll_flat,
        "RevDEQ (Custom Backward)", grads_revdeq_flat
    )
    
    # Compare RevDEQ custom backward vs RevDEQ unrolled backward
    cos2, rel2, norm2 = compare_gradients(
        "RevDEQ (Unrolled Backward)", grads_revdeq_unroll_flat,
        "RevDEQ (Custom Backward)", grads_revdeq_flat
    )
    
    # Compare RevDEQ unrolled vs Simple GD unrolled
    cos3, rel3, norm3 = compare_gradients(
        "Simple GD (Full Backprop)", grads_unroll_flat,
        "RevDEQ (Unrolled Backward)", grads_revdeq_unroll_flat
    )
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print("\n[KEY INSIGHT] The comparison 'RevDEQ Unrolled vs RevDEQ Custom' shows")
    print("   whether the custom backward correctly computes the unrolled gradient.")
    
    if cos2 > 0.99:
        print(f"\n[OK] RevDEQ custom backward matches unrolled backward (cos={cos2:.4f})")
        print("   The reversible adjoint is mathematically correct!")
    else:
        print(f"\n[WARNING] RevDEQ custom backward differs from unrolled (cos={cos2:.4f})")
        print("   There may be a bug in the reversible adjoint implementation.")
    
    print("\n[INFO] The comparison 'Simple GD vs RevDEQ Unrolled' shows whether")
    print("   the forward iterations produce the same result.")
    
    if cos3 > 0.99:
        print(f"\n[OK] RevDEQ forward matches Simple GD forward (cos={cos3:.4f})")
    else:
        print(f"\n[INFO] RevDEQ forward differs from Simple GD (cos={cos3:.4f})")
        print("   This is expected! RevDEQ uses different iteration scheme (beta-averaged).")
    
    return {
        'revdeq_vs_simple_gd': {'cos': cos1, 'rel_l2': rel1, 'norm_ratio': norm1},
        'revdeq_custom_vs_unroll': {'cos': cos2, 'rel_l2': rel2, 'norm_ratio': norm2},
        'simple_gd_vs_revdeq_unroll': {'cos': cos3, 'rel_l2': rel3, 'norm_ratio': norm3},
    }


def test_gradient_accuracy_with_original_test_setup():
    """
    Test using the same setup as the passing JAX-style test to isolate the issue.
    """
    print("\n" + "=" * 70)
    print("TEST: Minimal MLP Test (Same as JAX test)")
    print("=" * 70)
    
    torch.manual_seed(42)
    dtype = torch.float64  # Use float64 like the original test
    device = "cpu"
    
    # Simple MLP like in the passing test
    class F(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(2, 10, bias=True, dtype=dtype),
                torch.nn.Tanh(),
                torch.nn.Linear(10, 1, bias=True, dtype=dtype),
            )
        
        def forward(self, z, x):
            zx = torch.cat([z, x], dim=-1)
            return self.mlp(zx)
    
    function = F().to(device)
    x = torch.tensor([[0.1]], dtype=dtype, device=device, requires_grad=True)
    z0 = torch.zeros(1, 1, dtype=dtype, device=device)
    
    beta = 0.8
    max_steps = 10
    tol = -1.0  # Fixed steps
    
    def f(z, args):
        (x_,) = args
        return function(z, x_)
    
    # =========================================================================
    # METHOD 1: Unrolled (Standard Autograd)
    # =========================================================================
    print("\n--- Method 1: Unrolled Backprop ---")
    
    for p in function.parameters():
        if p.grad is not None:
            p.grad.zero_()
    if x.grad is not None:
        x.grad.zero_()
    
    solver = ReversibleSolver(beta=beta)
    y_state, fz = solver.init(f, z0, (x,))
    z = z0.clone()
    for _ in range(max_steps):
        z, (y_state, fz), _ = solver.step(f, z, (x,), (y_state, fz))
    
    loss_unroll = z.sum()
    loss_unroll.backward()
    
    grads_unroll = [p.grad.detach().clone() for p in function.parameters()]
    grads_unroll_flat = torch.cat([g.flatten() for g in grads_unroll])
    gradx_unroll = x.grad.detach().clone()
    
    print(f"  Loss: {loss_unroll.item():.6f}")
    print(f"  Gradient norm (params): {torch.norm(grads_unroll_flat).item():.6f}")
    print(f"  Gradient norm (x): {torch.norm(gradx_unroll).item():.6f}")
    
    # =========================================================================
    # METHOD 2: RevDEQ Custom Backward
    # =========================================================================
    print("\n--- Method 2: RevDEQ Custom Backward ---")
    
    for p in function.parameters():
        if p.grad is not None:
            p.grad.zero_()
    x.grad.zero_()
    
    params = [p for p in function.parameters() if p.requires_grad] + [x]
    z1, steps, err = solve_reversible_adjoint(
        f, z0, args=(x,), params=params, beta=beta, tol=tol, max_steps=max_steps
    )
    
    loss_revdeq = z1.sum()
    loss_revdeq.backward()
    
    grads_revdeq = [p.grad.detach().clone() for p in function.parameters()]
    grads_revdeq_flat = torch.cat([g.flatten() for g in grads_revdeq])
    gradx_revdeq = x.grad.detach().clone()
    
    print(f"  Loss: {loss_revdeq.item():.6f}")
    print(f"  Gradient norm (params): {torch.norm(grads_revdeq_flat).item():.6f}")
    print(f"  Gradient norm (x): {torch.norm(gradx_revdeq).item():.6f}")
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON (MLP Test)")
    print("=" * 70)
    
    cos_params = torch.nn.functional.cosine_similarity(
        grads_unroll_flat.unsqueeze(0), grads_revdeq_flat.unsqueeze(0)
    ).item()
    cos_x = torch.nn.functional.cosine_similarity(
        gradx_unroll.unsqueeze(0), gradx_revdeq.unsqueeze(0)
    ).item()
    
    print(f"\nParam gradients - Cosine Similarity: {cos_params:.6f}")
    print(f"x gradient - Cosine Similarity: {cos_x:.6f}")
    
    if cos_params > 0.99 and cos_x > 0.99:
        print("\n[OK] MLP test passes - custom backward matches unrolled!")
    else:
        print("\n[FAIL] MLP test fails!")
    
    return cos_params, cos_x


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING BOTH TESTS")
    print("=" * 70)
    
    # Test 1: MLP (should pass)
    cos_mlp_params, cos_mlp_x = test_gradient_accuracy_with_original_test_setup()
    
    # Test 2: Regularizer (may fail)
    results = test_gradient_accuracy_simple()
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nMLP Test (like JAX test): cos_params={cos_mlp_params:.4f}, cos_x={cos_mlp_x:.4f}")
    print(f"Regularizer Test: cos={results['revdeq_custom_vs_unroll']['cos']:.4f}")
