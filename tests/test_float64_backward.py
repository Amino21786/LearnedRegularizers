"""
Test that float64 is used for fixed-point operations while NN stays in float32.

This verifies the design:
- Forward pass: fixed-point iterations in float64, NN calls in float32
- Backward pass: state reconstruction in float64, NN calls in float32
- Output and gradients: float32 (matching NN parameters)
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_methods.reversible_deq import solve_reversible_adjoint, FP_DTYPE, NN_DTYPE


class SimpleRegularizer(nn.Module):
    """Simple CNN regularizer for testing."""
    
    def __init__(self, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, channels, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out
    
    def grad(self, x):
        """Gradient of regularizer."""
        return self.forward(x)


def verify_dtype_handling():
    """Verify that dtypes are handled correctly throughout."""
    print("=" * 60)
    print("Testing float64 fixed-point operations with float32 NN")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"FP_DTYPE (fixed-point): {FP_DTYPE}")
    print(f"NN_DTYPE (neural network): {NN_DTYPE}")
    
    # Create regularizer in float32
    regularizer = SimpleRegularizer(channels=1).to(device)
    print(f"\nRegularizer parameter dtype: {next(regularizer.parameters()).dtype}")
    assert next(regularizer.parameters()).dtype == torch.float32, "Regularizer should be float32"
    
    # Create test data in float32
    x = torch.randn(1, 1, 16, 16, device=device, dtype=torch.float32)
    physics_scale = 0.9
    y = x * physics_scale
    step_size = 0.1
    
    # Fixed-point function
    def fixed_point_fn(z, args):
        """f(z) = z - step_size * (A^T(Az - y) + grad_R(z))"""
        physics_grad = physics_scale * (physics_scale * z - y)
        reg_grad = regularizer.grad(z)
        return z - step_size * (physics_grad + reg_grad)
    
    params = list(regularizer.parameters())
    
    print("\n--- Forward Pass Test ---")
    z0 = x.clone().requires_grad_(True)
    z1, steps, error = solve_reversible_adjoint(
        fixed_point_fn, z0, args=None, params=params,
        beta=0.5, tol=1e-4, max_steps=10,
        use_float64=True
    )
    
    print(f"Input z0 dtype: {z0.dtype}")
    print(f"Output z1 dtype: {z1.dtype}")
    print(f"Steps: {steps}, Error: {error:.2e}")
    
    assert z1.dtype == torch.float32, "Output should be float32"
    
    print("\n--- Backward Pass Test ---")
    loss = z1.sum()
    loss.backward()
    
    print(f"z0.grad dtype: {z0.grad.dtype if z0.grad is not None else None}")
    for i, p in enumerate(params):
        print(f"params[{i}].grad dtype: {p.grad.dtype if p.grad is not None else None}")
        if p.grad is not None:
            assert p.grad.dtype == torch.float32, f"Parameter gradient should be float32"
    
    if z0.grad is not None:
        assert z0.grad.dtype == torch.float32, "Input gradient should be float32"
    
    print("\n[OK] All dtype checks passed!")
    return True


def test_gradient_accuracy():
    """Compare RevDEQ vs unrolled gradients with float64 fixed-point ops."""
    print("\n" + "=" * 60)
    print("Testing Gradient Accuracy: RevDEQ vs Unrolled")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    regularizer = SimpleRegularizer(channels=1).to(device)
    
    x = torch.randn(1, 1, 16, 16, device=device, dtype=torch.float32)
    physics_scale = 0.9
    y = x * physics_scale
    step_size = 0.1
    
    def fixed_point_fn(z, args):
        physics_grad = physics_scale * (physics_scale * z - y)
        reg_grad = regularizer.grad(z)
        return z - step_size * (physics_grad + reg_grad)
    
    test_configs = [
        {"steps": 5, "beta": 0.5},
        {"steps": 10, "beta": 0.5},
        {"steps": 20, "beta": 0.5},
        {"steps": 30, "beta": 0.5},
    ]
    
    print("\n" + "-" * 60)
    print(f"{'Steps':<10} {'Max Error':<15} {'Mean Error':<15} {'Status':<10}")
    print("-" * 60)
    
    for config in test_configs:
        steps = config["steps"]
        beta = config["beta"]
        
        # --- RevDEQ forward + backward ---
        regularizer.zero_grad()
        z0 = x.clone()
        params = list(regularizer.parameters())
        
        z_rev, _, _ = solve_reversible_adjoint(
            fixed_point_fn, z0, args=None, params=params,
            beta=beta, tol=1e-10, max_steps=steps,
            use_float64=True
        )
        loss_rev = z_rev.sum()
        loss_rev.backward()
        
        grad_rev = [p.grad.clone() if p.grad is not None else None for p in params]
        
        # --- Unrolled forward + backward ---
        regularizer.zero_grad()
        z = x.clone()
        y_state = z.clone()
        fz = fixed_point_fn(z, None)
        
        for _ in range(steps):
            y_state = (1 - beta) * y_state + beta * fz
            f_y = fixed_point_fn(y_state, None)
            z = (1 - beta) * z + beta * f_y
            fz = fixed_point_fn(z, None)
        
        loss_unroll = z.sum()
        loss_unroll.backward()
        
        grad_unroll = [p.grad.clone() if p.grad is not None else None for p in params]
        
        # Compare gradients
        max_errors = []
        mean_errors = []
        for g_rev, g_unroll in zip(grad_rev, grad_unroll):
            if g_rev is not None and g_unroll is not None:
                err = torch.abs(g_rev - g_unroll)
                max_errors.append(err.max().item())
                mean_errors.append(err.mean().item())
        
        max_err = max(max_errors) if max_errors else 0
        mean_err = sum(mean_errors) / len(mean_errors) if mean_errors else 0
        
        status = "OK" if max_err < 1e-4 else "~" if max_err < 1e-2 else "!"
        print(f"{steps:<10} {max_err:<15.2e} {mean_err:<15.2e} {status:<10}")
    
    print("-" * 60)
    print("Legend: OK = excellent (<1e-4), ~ = good (<1e-2), ! = check needed")
    
    return True


def compare_float32_vs_float64():
    """Compare gradient accuracy with and without float64."""
    print("\n" + "=" * 60)
    print("Comparing float32 vs float64 Fixed-Point Operations")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.randn(1, 1, 16, 16, device=device, dtype=torch.float32)
    physics_scale = 0.9
    y = x * physics_scale
    step_size = 0.1
    
    steps = 30
    beta = 0.5
    
    for use_float64 in [False, True]:
        regularizer = SimpleRegularizer(channels=1).to(device)
        
        def fixed_point_fn(z, args):
            physics_grad = physics_scale * (physics_scale * z - y)
            reg_grad = regularizer.grad(z)
            return z - step_size * (physics_grad + reg_grad)
        
        # RevDEQ
        regularizer.zero_grad()
        params = list(regularizer.parameters())
        z_rev, _, _ = solve_reversible_adjoint(
            fixed_point_fn, x.clone(), args=None, params=params,
            beta=beta, tol=1e-10, max_steps=steps,
            use_float64=use_float64
        )
        z_rev.sum().backward()
        grad_rev = [p.grad.clone() for p in params]
        
        # Unrolled
        regularizer.zero_grad()
        z = x.clone()
        y_state = z.clone()
        fz = fixed_point_fn(z, None)
        for _ in range(steps):
            y_state = (1 - beta) * y_state + beta * fz
            f_y = fixed_point_fn(y_state, None)
            z = (1 - beta) * z + beta * f_y
            fz = fixed_point_fn(z, None)
        z.sum().backward()
        grad_unroll = [p.grad.clone() for p in params]
        
        # Compare
        max_errors = []
        for g_rev, g_unroll in zip(grad_rev, grad_unroll):
            err = torch.abs(g_rev - g_unroll)
            max_errors.append(err.max().item())
        
        max_err = max(max_errors)
        mode = "float64" if use_float64 else "float32"
        print(f"{mode} fixed-point ops: max gradient error = {max_err:.2e}")
    
    return True


if __name__ == "__main__":
    print("Testing RevDEQ with float64 fixed-point operations")
    print("Neural network stays in float32\n")
    
    verify_dtype_handling()
    test_gradient_accuracy()
    compare_float32_vs_float64()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
