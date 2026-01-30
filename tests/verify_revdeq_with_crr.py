"""
Verification script: RevDEQ vs Unrolled Backpropagation with Real CRR Regularizer

This script verifies the RevDEQ implementation using the actual CRR regularizer
from the paper, providing a more realistic test case.
"""

import os
import sys
import torch
import numpy as np

# Ensure imports work from various directories
HERE = os.path.dirname(__file__)
LR_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(LR_ROOT, ".."))
for p in (LR_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from training_methods.reversible_deq import ReversibleSolver, solve_reversible_adjoint
from priors import ParameterLearningWrapper, WCRR


# =============================================================================
# Identity physics and L2 data fidelity
# =============================================================================
class IdentityPhysics:
    def __call__(self, x):
        return x
    def A_dagger(self, y):
        return y


class L2DataFidelity:
    def __call__(self, x, y, physics):
        return 0.5 * ((x - y) ** 2).view(x.shape[0], -1).sum(-1)
    def grad(self, x, y, physics):
        return x - y


# =============================================================================
# Unrolled forward pass
# =============================================================================
def unroll_reversible_forward(function, z0, args, beta, max_steps):
    """Standard unrolled forward pass that retains autograd graph."""
    # nn_dtype=torch.float64 since NN is in float64 for verification
    solver = ReversibleSolver(beta=beta, nn_dtype=torch.float64)
    y, fz = solver.init(function, z0, args)
    z = z0.clone()
    
    for step in range(max_steps):
        z, (y, fz), error = solver.step(function, z, args, (y, fz))
    
    return z, max_steps, error


# =============================================================================
# Main verification
# =============================================================================
def verify_with_crr(
    num_steps: int = 5,
    beta: float = 0.8,
    step_size: float = 0.1,
    lamda: float = 1.0,
    image_size: int = 16,
    seed: int = 42
):
    """
    Verify RevDEQ gradients match unrolled backpropagation using CRR regularizer.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"
    dtype = torch.float64
    
    print("=" * 70)
    print("RevDEQ Verification with CRR Regularizer")
    print("=" * 70)
    print(f"Settings: num_steps={num_steps}, beta={beta}, step_size={step_size}")
    print(f"Image size: {image_size}x{image_size}, dtype={dtype} (RevDEQ uses float64 internally)")
    print()
    
    # Create CRR regularizer (stays in float32)
    base_reg = WCRR(sigma=0.1, weak_convexity=0.0).to(dtype).to(device)
    regularizer = ParameterLearningWrapper(base_reg, device=device)
    regularizer = regularizer.to(dtype).to(device)
    
    for p in regularizer.parameters():
        p.requires_grad_(True)
    
    # Create data fidelity and physics
    data_fidelity = L2DataFidelity()
    physics = IdentityPhysics()
    
    # Create test image and observation
    x_true = torch.randn(1, 1, image_size, image_size, dtype=dtype, device=device) * 0.5
    y = physics(x_true) + torch.randn_like(x_true) * 0.1
    z0 = physics.A_dagger(y).clone()
    
    # Define fixed-point function
    def fixed_point_function(z, args):
        y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg = args
        grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
        grad_reg = lamda_arg * regularizer_arg.grad(z)
        grad_total = grad_data + grad_reg
        z_new = z - step_size_arg * grad_total
        return z_new
    
    args = (y, physics, data_fidelity, regularizer, lamda, step_size)
    params = list(regularizer.parameters())
    
    print(f"Number of trainable parameters: {len(params)}")
    print(f"Parameter shapes: {[tuple(p.shape) for p in params]}")
    print()
    
    # =========================================================================
    # Method 1: Unrolled
    # =========================================================================
    print("-" * 70)
    print("Method 1: Unrolled Forward + Standard Autograd")
    print("-" * 70)
    
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    
    z_unroll, steps_u, err_u = unroll_reversible_forward(
        fixed_point_function, z0, args, beta=beta, max_steps=num_steps
    )
    
    loss_unroll = ((z_unroll - x_true) ** 2).sum()
    loss_unroll.backward()
    
    grads_unroll = [p.grad.detach().clone() for p in params]
    
    print(f"  Forward steps: {steps_u}, final error: {err_u:.2e}")
    print(f"  Loss: {loss_unroll.item():.6f}")
    for i, g in enumerate(grads_unroll):
        print(f"  Param {i} grad norm: {g.norm().item():.6e}")
    
    # =========================================================================
    # Method 2: RevDEQ
    # =========================================================================
    print()
    print("-" * 70)
    print("Method 2: RevDEQ with Custom Backward")
    print("-" * 70)
    
    for p in params:
        p.grad = None
    
    # nn_dtype=torch.float64 since NN is in float64 for verification
    z_revdeq, steps_r, err_r = solve_reversible_adjoint(
        fixed_point_function, z0.clone(), args, params=params,
        beta=beta, tol=-1.0, max_steps=num_steps,
        nn_dtype=torch.float64
    )
    
    loss_revdeq = ((z_revdeq - x_true) ** 2).sum()
    loss_revdeq.backward()
    
    grads_revdeq = [p.grad.detach().clone() for p in params]
    
    print(f"  Forward steps: {steps_r}, final error: {err_r:.2e}")
    print(f"  Loss: {loss_revdeq.item():.6f}")
    for i, g in enumerate(grads_revdeq):
        print(f"  Param {i} grad norm: {g.norm().item():.6e}")
    
    # =========================================================================
    # Compare
    # =========================================================================
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    z_diff = (z_unroll.detach() - z_revdeq.detach()).abs().max().item()
    print(f"Forward z difference (max abs): {z_diff:.2e}")
    
    loss_diff = abs(loss_unroll.item() - loss_revdeq.item())
    print(f"Loss difference: {loss_diff:.2e}")
    
    print()
    print("Parameter gradient comparison:")
    all_match = True
    for i, (gu, gr) in enumerate(zip(grads_unroll, grads_revdeq)):
        diff = (gu - gr).abs().max().item()
        rel_diff = diff / (gu.abs().max().item() + 1e-10)
        status = "[OK]" if rel_diff < 1e-2 else "[FAIL]"
        print(f"  Param {i}: max abs diff = {diff:.2e}, rel diff = {rel_diff:.2e} {status}")
        if rel_diff >= 1e-2:
            all_match = False
    
    print()
    if all_match and z_diff < 1e-8 and loss_diff < 1e-8:
        print("SUCCESS: RevDEQ matches unrolled backpropagation with CRR!")
    else:
        print("WARNING: Discrepancies detected!")
    
    return all_match


# =============================================================================
# Additional test: Gradient accumulation visualization
# =============================================================================
def visualize_gradient_accumulation(num_steps=3, beta=0.5):
    """
    Visualize how gradients accumulate step-by-step in RevDEQ backward pass.
    This helps understand the mechanics of the reversible adjoint method.
    """
    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float64
    
    print()
    print("=" * 70)
    print("Gradient Accumulation Visualization")
    print("=" * 70)
    print(f"Using {num_steps} steps with beta={beta}")
    print()
    
    # Simple scalar parameter for easy visualization
    w = torch.nn.Parameter(torch.tensor(0.2, dtype=dtype))
    
    def f(z, args):
        (w_,) = args
        return torch.tanh(w_ * z)
    
    z0 = torch.tensor([[1.0]], dtype=dtype)
    target = torch.tensor([[0.0]], dtype=dtype)
    
    # Forward pass (nn_dtype=float64 for verification)
    solver = ReversibleSolver(beta=beta, nn_dtype=torch.float64)
    y, fz = solver.init(f, z0, (w,))
    z = z0.clone()
    
    trajectory = [(z.clone().item(), y.clone().item())]
    
    for step in range(num_steps):
        z, (y, fz), error = solver.step(f, z, (w,), (y, fz))
        trajectory.append((z.clone().item(), y.clone().item()))
    
    print("Forward trajectory (z, y values at each step):")
    for i, (zv, yv) in enumerate(trajectory):
        print(f"  Step {i}: z={zv:.6f}, y={yv:.6f}")
    
    # Compute loss
    loss = ((z - target) ** 2).sum()
    print(f"\nFinal loss: {loss.item():.6f}")
    
    # Backward pass (manually)
    print("\nBackward pass (RevDEQ-style):")
    grad_z = 2 * (z - target)
    grad_y = torch.zeros_like(y)
    grad_w = torch.tensor(0.0, dtype=dtype)
    
    z_curr = z.detach().clone()
    y_curr = y.detach().clone()
    
    print(f"  Initial grad_z: {grad_z.item():.6f}")
    
    for step in range(num_steps, 0, -1):
        # VJP at y_curr
        with torch.enable_grad():
            y_req = y_curr.detach().requires_grad_(True)
            w_req = w.detach().requires_grad_(True)
            fy = torch.tanh(w_req * y_req)
            
            grads = torch.autograd.grad(fy, [y_req, w_req], grad_outputs=grad_z)
            dgrad_y = grads[0]
            dgrad_w_y = grads[1]
        
        grad_y = grad_y + beta * dgrad_y
        grad_y0 = (1 - beta) * grad_y
        
        # Reconstruct z0
        z0_rec = (z_curr - beta * torch.tanh(w * y_curr).detach()) / (1 - beta)
        
        # VJP at z0_rec
        with torch.enable_grad():
            z_req = z0_rec.detach().requires_grad_(True)
            w_req = w.detach().requires_grad_(True)
            fz = torch.tanh(w_req * z_req)
            
            grads = torch.autograd.grad(fz, [z_req, w_req], grad_outputs=grad_y)
            dgrad_z = grads[0]
            dgrad_w_z = grads[1]
        
        grad_z_new = (1 - beta) * grad_z + beta * dgrad_z
        y0_rec = (y_curr - beta * torch.tanh(w * z0_rec).detach()) / (1 - beta)
        
        # Accumulate parameter gradient
        grad_w = grad_w + beta * (dgrad_w_y + dgrad_w_z)
        
        print(f"  Step {step}->{step-1}: z0_rec={z0_rec.item():.6f}, y0_rec={y0_rec.item():.6f}")
        print(f"                   grad_z={grad_z_new.item():.6f}, grad_y={grad_y0.item():.6f}")
        print(f"                   dgrad_w_y={dgrad_w_y.item():.6f}, dgrad_w_z={dgrad_w_z.item():.6f}")
        print(f"                   cumulative grad_w={grad_w.item():.6f}")
        
        z_curr = z0_rec.detach()
        y_curr = y0_rec.detach()
        grad_z = grad_z_new.detach()
        grad_y = grad_y0.detach()
    
    # Compare with unrolled autograd
    w.grad = None
    solver = ReversibleSolver(beta=beta, nn_dtype=torch.float64)
    y, fz = solver.init(f, z0, (w,))
    z = z0.clone()
    for _ in range(num_steps):
        z, (y, fz), _ = solver.step(f, z, (w,), (y, fz))
    loss = ((z - target) ** 2).sum()
    loss.backward()
    
    print(f"\nFinal RevDEQ grad_w: {grad_w.item():.6f}")
    print(f"Unrolled autograd grad_w: {w.grad.item():.6f}")
    print(f"Difference: {abs(grad_w.item() - w.grad.item()):.2e}")
    
    if abs(grad_w.item() - w.grad.item()) < 1e-6:
        print("SUCCESS: Manual RevDEQ backward matches autograd!")
    else:
        print("WARNING: Mismatch detected!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REVDEQ VERIFICATION WITH CRR REGULARIZER")
    print("=" * 70 + "\n")
    
    # Test 1: CRR with small number of steps
    print("\n### Test 1: CRR Regularizer (5 steps) ###\n")
    success1 = verify_with_crr(num_steps=5, beta=0.5)
    
    # Test 2: CRR with more steps
    print("\n### Test 2: CRR Regularizer (10 steps) ###\n")
    success2 = verify_with_crr(num_steps=10, beta=0.5)
    
    # Test 3: Gradient accumulation visualization
    print("\n### Test 3: Gradient Accumulation Visualization ###")
    visualize_gradient_accumulation(num_steps=3, beta=0.5)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  CRR (5 steps): {'PASS' if success1 else 'FAIL'}")
    print(f"  CRR (10 steps): {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\nAll tests PASSED! RevDEQ implementation verified with CRR.")
    else:
        print("\nSome tests FAILED.")
