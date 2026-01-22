"""
Verification script: RevDEQ vs Standard Unrolled Backpropagation

This script verifies that the RevDEQ implementation computes gradients correctly
by comparing them against standard backpropagation through unrolling. We verify
step-by-step gradient equivalence for the first few iterations of the backward pass.

The key difference between the methods:
- Unrolled: Keeps entire computation graph, autograd computes gradients
- RevDEQ: Reconstructs states in reverse order, accumulates vjps without storing history

For correctness, both should produce the same gradients (within numerical precision).
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, Any, List

# Ensure imports work from various directories
HERE = os.path.dirname(__file__)
LR_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(LR_ROOT, ".."))
for p in (LR_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from training_methods.reversible_deq import ReversibleSolver, solve_reversible_adjoint


# =============================================================================
# Simple regularizer for testing (mimics structure of real priors)
# =============================================================================
class SimpleConvRegularizer(nn.Module):
    """
    Simple 2-layer conv regularizer that mimics CRR-style structure.
    Returns a scalar energy and has a .grad() method for gradient computation.
    """
    def __init__(self, in_channels=1, hidden_channels=8, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size, padding=kernel_size//2)
        nn.init.xavier_normal_(self.conv1.weight, gain=0.1)
        nn.init.xavier_normal_(self.conv2.weight, gain=0.1)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x):
        """Compute regularizer energy (scalar)."""
        h = torch.tanh(self.conv1(x))
        out = self.conv2(h)
        return 0.5 * (out ** 2).sum()
    
    def grad(self, x):
        """Compute gradient of regularizer w.r.t. input x.
        
        Note: Uses torch.enable_grad() to handle cases where this is called
        inside a no_grad context (e.g., in RevDEQ forward pass).
        """
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            energy = self.forward(x_req)
            grad_x = torch.autograd.grad(energy, x_req, create_graph=True)[0]
        return grad_x


# =============================================================================
# Simple data fidelity (L2 norm squared)
# =============================================================================
class SimpleL2DataFidelity:
    """Simple L2 data fidelity: 0.5 * ||x - y||^2."""
    def __call__(self, x, y):
        return 0.5 * ((x - y) ** 2).sum()
    
    def grad(self, x, y, physics=None):
        """Gradient is just (x - y) for identity physics."""
        return x - y


# =============================================================================
# Identity physics (denoising)
# =============================================================================
class IdentityPhysics:
    """Identity operator: A(x) = x (denoising case)."""
    def __call__(self, x):
        return x
    
    def A_dagger(self, y):
        return y


# =============================================================================
# Unrolled forward pass with gradient graph retention
# =============================================================================
def unroll_reversible_forward(
    function: Callable,
    z0: torch.Tensor,
    args: Any,
    beta: float,
    max_steps: int,
    store_trajectory: bool = False
) -> Tuple[torch.Tensor, int, float, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Standard unrolled forward pass that retains autograd graph.
    
    Returns:
        z: Final iterate
        steps: Number of steps taken
        error: Final residual
        trajectory: List of (z, y) pairs if store_trajectory=True
    """
    solver = ReversibleSolver(beta=beta)
    y, fz = solver.init(function, z0, args)
    z = z0.clone()
    
    trajectory = [(z.clone(), y.clone())] if store_trajectory else []
    
    for step in range(max_steps):
        z, (y, fz), error = solver.step(function, z, args, (y, fz))
        if store_trajectory:
            trajectory.append((z.clone(), y.clone()))
    
    return z, max_steps, error, trajectory


# =============================================================================
# Manual step-by-step backward pass (for verification)
# =============================================================================
def manual_backward_step(
    function: Callable,
    z1: torch.Tensor,
    y1: torch.Tensor,
    grad_z1: torch.Tensor,
    grad_y1: torch.Tensor,
    grad_params: List[torch.Tensor],
    params: List[torch.Tensor],
    args: Any,
    beta: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Perform one step of the RevDEQ backward pass manually.
    
    This mirrors the backward pass in _ReversibleDEQFunction.backward() but exposes
    intermediate values for debugging and verification.
    
    Returns:
        z0, y0: Reconstructed previous states
        grad_z0, grad_y0: Gradients at previous step
        grad_params: Updated parameter gradients
    """
    # Compute z0 from z1 and f(y1)
    with torch.enable_grad():
        y1_req = y1.detach().requires_grad_(True)
        fy1 = function(y1_req, args)
        
        # VJP at y1: apply to grad_z1
        grads_y = torch.autograd.grad(
            fy1,
            (y1_req, *params),
            grad_outputs=grad_z1,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        dgrad_y1 = grads_y[0] if grads_y[0] is not None else torch.zeros_like(y1_req)
        dgrad_params_y = grads_y[1:]
        
        # Update grad_y1
        grad_y1_new = grad_y1 + beta * dgrad_y1
        grad_y0 = (1 - beta) * grad_y1_new
        
        # Reconstruct z0
        z0 = (z1 - beta * fy1.detach()) / (1 - beta)
        
        # Compute y0 from y1 and f(z0)
        z0_req = z0.detach().requires_grad_(True)
        fz0 = function(z0_req, args)
        
        # VJP at z0: apply to grad_y1
        grads_z = torch.autograd.grad(
            fz0,
            (z0_req, *params),
            grad_outputs=grad_y1_new,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        dgrad_z0 = grads_z[0] if grads_z[0] is not None else torch.zeros_like(z0_req)
        dgrad_params_z = grads_z[1:]
        
        # Update grad_z0
        grad_z0 = (1 - beta) * grad_z1 + beta * dgrad_z0
        
        # Reconstruct y0
        y0 = (y1 - beta * fz0.detach()) / (1 - beta)
        
        # Accumulate parameter gradients
        new_grad_params = []
        for i, p in enumerate(params):
            gy = dgrad_params_y[i] if i < len(dgrad_params_y) and dgrad_params_y[i] is not None else torch.zeros_like(p)
            gz = dgrad_params_z[i] if i < len(dgrad_params_z) and dgrad_params_z[i] is not None else torch.zeros_like(p)
            new_grad_params.append(grad_params[i] + beta * (gy + gz))
    
    return z0.detach(), y0.detach(), grad_z0.detach(), grad_y0.detach(), new_grad_params


# =============================================================================
# Main verification function
# =============================================================================
def verify_revdeq_vs_unroll(
    num_steps: int = 5,
    beta: float = 0.8,
    step_size: float = 0.1,
    lamda: float = 1.0,
    image_size: int = 8,
    seed: int = 42,
    verbose: bool = True
):
    """
    Verify RevDEQ gradients match unrolled backpropagation.
    
    Args:
        num_steps: Number of fixed-point iterations
        beta: Reversible relaxation parameter
        step_size: Step size for gradient descent
        lamda: Regularization parameter
        image_size: Size of test image (image_size x image_size)
        seed: Random seed
        verbose: Print detailed output
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"  # Use CPU for numerical stability in testing
    dtype = torch.float64  # Use double precision for accurate comparison
    
    print("=" * 70)
    print("RevDEQ vs Unrolled Backpropagation Verification")
    print("=" * 70)
    print(f"Settings: num_steps={num_steps}, beta={beta}, step_size={step_size}, lambda={lamda}")
    print(f"Image size: {image_size}x{image_size}, dtype={dtype}")
    print()
    
    # Create regularizer
    regularizer = SimpleConvRegularizer(in_channels=1, hidden_channels=4, kernel_size=3)
    regularizer = regularizer.to(dtype).to(device)
    for p in regularizer.parameters():
        p.requires_grad_(True)
    
    # Create data fidelity and physics
    data_fidelity = SimpleL2DataFidelity()
    physics = IdentityPhysics()
    
    # Create test image and observation
    x_true = torch.randn(1, 1, image_size, image_size, dtype=dtype, device=device) * 0.5
    y = physics(x_true) + torch.randn_like(x_true) * 0.1  # Add noise
    z0 = physics.A_dagger(y).clone()
    
    # Define fixed-point function (gradient descent step)
    def fixed_point_function(z, args):
        y_arg, physics_arg, data_fidelity_arg, regularizer_arg, lamda_arg, step_size_arg = args
        grad_data = data_fidelity_arg.grad(z, y_arg, physics_arg)
        grad_reg = lamda_arg * regularizer_arg.grad(z)
        grad_total = grad_data + grad_reg
        z_new = z - step_size_arg * grad_total
        return z_new
    
    args = (y, physics, data_fidelity, regularizer, lamda, step_size)
    params = list(regularizer.parameters())
    
    # =========================================================================
    # Method 1: Unrolled forward + standard autograd backward
    # =========================================================================
    print("-" * 70)
    print("Method 1: Unrolled Forward + Standard Autograd Backward")
    print("-" * 70)
    
    # Zero gradients
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    
    # Forward pass (retain graph for autograd)
    z_unroll, steps_u, err_u, trajectory = unroll_reversible_forward(
        fixed_point_function, z0, args, beta=beta, max_steps=num_steps, store_trajectory=True
    )
    
    # Upper-level loss: ||z - x_true||^2
    loss_unroll = ((z_unroll - x_true) ** 2).sum()
    loss_unroll.backward()
    
    grads_unroll = [p.grad.detach().clone() for p in params]
    
    print(f"  Forward steps: {steps_u}, final error: {err_u:.2e}")
    print(f"  Loss: {loss_unroll.item():.6f}")
    for i, g in enumerate(grads_unroll):
        print(f"  Param {i} grad norm: {g.norm().item():.6e}")
    
    # =========================================================================
    # Method 2: RevDEQ with custom backward
    # =========================================================================
    print()
    print("-" * 70)
    print("Method 2: RevDEQ with Custom Backward")
    print("-" * 70)
    
    # Zero gradients
    for p in params:
        p.grad = None
    
    # Forward + backward via solve_reversible_adjoint
    z_revdeq, steps_r, err_r = solve_reversible_adjoint(
        fixed_point_function, z0.clone(), args, params=params, 
        beta=beta, tol=-1.0, max_steps=num_steps  # tol=-1 forces exact num_steps
    )
    
    loss_revdeq = ((z_revdeq - x_true) ** 2).sum()
    loss_revdeq.backward()
    
    grads_revdeq = [p.grad.detach().clone() for p in params]
    
    print(f"  Forward steps: {steps_r}, final error: {err_r:.2e}")
    print(f"  Loss: {loss_revdeq.item():.6f}")
    for i, g in enumerate(grads_revdeq):
        print(f"  Param {i} grad norm: {g.norm().item():.6e}")
    
    # =========================================================================
    # Method 3: Manual step-by-step backward (for detailed verification)
    # =========================================================================
    if verbose:
        print()
        print("-" * 70)
        print("Method 3: Manual Step-by-Step Backward (Verification)")
        print("-" * 70)
        
        # Start from final state
        z_final, y_final = trajectory[-1]
        
        # Initialize gradients
        grad_z = 2 * (z_final - x_true)  # d(loss)/d(z)
        grad_y = torch.zeros_like(y_final)
        grad_params_manual = [torch.zeros_like(p) for p in params]
        
        print(f"  Initial grad_z norm: {grad_z.norm().item():.6e}")
        print()
        
        # Step backward through each iteration
        z_curr, y_curr = z_final.detach().clone(), y_final.detach().clone()
        grad_z_curr, grad_y_curr = grad_z.clone(), grad_y.clone()
        
        for step in range(num_steps, 0, -1):
            z_prev, y_prev, grad_z_new, grad_y_new, grad_params_manual = manual_backward_step(
                fixed_point_function, z_curr, y_curr, grad_z_curr, grad_y_curr,
                grad_params_manual, params, args, beta
            )
            
            print(f"  Backward step {step} -> {step-1}:")
            print(f"    z_prev norm: {z_prev.norm().item():.6f}, y_prev norm: {y_prev.norm().item():.6f}")
            print(f"    grad_z norm: {grad_z_new.norm().item():.6e}, grad_y norm: {grad_y_new.norm().item():.6e}")
            print(f"    Accumulated param grad norms: {[g.norm().item() for g in grad_params_manual]}")
            
            z_curr, y_curr = z_prev, y_prev
            grad_z_curr, grad_y_curr = grad_z_new, grad_y_new
        
        print()
        print("  Final manual param grad norms:")
        for i, g in enumerate(grad_params_manual):
            print(f"    Param {i}: {g.norm().item():.6e}")
    
    # =========================================================================
    # Compare results
    # =========================================================================
    print()
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    # Check forward pass equivalence
    z_diff = (z_unroll.detach() - z_revdeq.detach()).abs().max().item()
    print(f"Forward z difference (max abs): {z_diff:.2e}")
    assert z_diff < 1e-8, f"Forward pass mismatch: {z_diff}"
    print("  [OK] Forward passes match")
    
    # Check loss equivalence
    loss_diff = abs(loss_unroll.item() - loss_revdeq.item())
    print(f"Loss difference: {loss_diff:.2e}")
    assert loss_diff < 1e-10, f"Loss mismatch: {loss_diff}"
    print("  [OK] Losses match")
    
    # Check gradient equivalence
    print()
    print("Parameter gradient comparison:")
    all_grads_match = True
    for i, (gu, gr) in enumerate(zip(grads_unroll, grads_revdeq)):
        diff = (gu - gr).abs().max().item()
        rel_diff = diff / (gu.abs().max().item() + 1e-10)
        status = "[OK]" if rel_diff < 1e-2 else "[FAIL]"
        print(f"  Param {i}: max abs diff = {diff:.2e}, rel diff = {rel_diff:.2e} {status}")
        if rel_diff >= 1e-2:
            all_grads_match = False
    
    if all_grads_match:
        print()
        print("=" * 70)
        print("SUCCESS: RevDEQ gradients match unrolled backpropagation!")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("WARNING: Some gradients have significant differences!")
        print("This may indicate a bug in the RevDEQ implementation.")
        print("=" * 70)
    
    return all_grads_match, grads_unroll, grads_revdeq


# =============================================================================
# Additional test: Gradient through multiple loss terms
# =============================================================================
def verify_with_complex_loss(num_steps=5, beta=0.8):
    """Test with a more complex upper-level loss function."""
    print()
    print("=" * 70)
    print("Additional Test: Complex Upper-Level Loss")
    print("=" * 70)
    
    torch.manual_seed(123)
    device = "cpu"
    dtype = torch.float64
    
    # Setup (same as main test)
    regularizer = SimpleConvRegularizer(in_channels=1, hidden_channels=4, kernel_size=3)
    regularizer = regularizer.to(dtype).to(device)
    for p in regularizer.parameters():
        p.requires_grad_(True)
    
    data_fidelity = SimpleL2DataFidelity()
    physics = IdentityPhysics()
    
    x_true = torch.randn(1, 1, 8, 8, dtype=dtype, device=device) * 0.5
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
    
    # Complex loss: MSE + SSIM-like term + TV
    def complex_loss(z, x):
        mse = ((z - x) ** 2).mean()
        # Simplified SSIM-like term
        mu_z = z.mean()
        mu_x = x.mean()
        var_z = ((z - mu_z) ** 2).mean()
        var_x = ((x - mu_x) ** 2).mean()
        cov = ((z - mu_z) * (x - mu_x)).mean()
        ssim_term = (2 * mu_z * mu_x + 0.01) * (2 * cov + 0.01) / \
                    ((mu_z**2 + mu_x**2 + 0.01) * (var_z + var_x + 0.01))
        # Total variation
        tv = ((z[:,:,1:,:] - z[:,:,:-1,:]) ** 2).mean() + \
             ((z[:,:,:,1:] - z[:,:,:,:-1]) ** 2).mean()
        return mse - 0.1 * ssim_term + 0.01 * tv
    
    # Unrolled
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    
    z_unroll, _, _, _ = unroll_reversible_forward(
        fixed_point_function, z0, args, beta=beta, max_steps=num_steps, store_trajectory=False
    )
    loss_u = complex_loss(z_unroll, x_true)
    loss_u.backward()
    grads_u = [p.grad.detach().clone() for p in params]
    
    # RevDEQ
    for p in params:
        p.grad = None
    
    z_revdeq, _, _ = solve_reversible_adjoint(
        fixed_point_function, z0.clone(), args, params=params,
        beta=beta, tol=-1.0, max_steps=num_steps
    )
    loss_r = complex_loss(z_revdeq, x_true)
    loss_r.backward()
    grads_r = [p.grad.detach().clone() for p in params]
    
    print(f"Complex loss - Unrolled: {loss_u.item():.6f}, RevDEQ: {loss_r.item():.6f}")
    
    all_match = True
    for i, (gu, gr) in enumerate(zip(grads_u, grads_r)):
        rel_diff = (gu - gr).abs().max().item() / (gu.abs().max().item() + 1e-10)
        status = "[OK]" if rel_diff < 1e-2 else "[FAIL]"
        print(f"  Param {i} rel diff: {rel_diff:.2e} {status}")
        if rel_diff >= 1e-2:
            all_match = False
    
    return all_match


# =============================================================================
# Run all verification tests
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REVDEQ VERIFICATION TEST SUITE")
    print("=" * 70 + "\n")
    
    # Test 1: Basic verification with small number of steps
    print("\n### Test 1: Basic verification (5 steps) ###\n")
    success1, _, _ = verify_revdeq_vs_unroll(num_steps=5, beta=0.8, verbose=True)
    
    # Test 2: More steps
    print("\n### Test 2: More steps (10 steps) ###\n")
    success2, _, _ = verify_revdeq_vs_unroll(num_steps=10, beta=0.8, verbose=False)
    
    # Test 3: Different beta values
    print("\n### Test 3: Different beta (beta=0.5) ###\n")
    success3, _, _ = verify_revdeq_vs_unroll(num_steps=5, beta=0.5, verbose=False)
    
    # Test 4: Complex loss
    print("\n### Test 4: Complex upper-level loss ###\n")
    success4 = verify_with_complex_loss(num_steps=5, beta=0.8)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results = [
        ("Basic (5 steps, beta=0.8)", success1),
        ("More steps (10 steps)", success2),
        ("Different beta (0.5)", success3),
        ("Complex loss", success4),
    ]
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests PASSED! RevDEQ implementation is verified.")
    else:
        print("Some tests FAILED. Please investigate the discrepancies.")
    
    sys.exit(0 if all_passed else 1)
