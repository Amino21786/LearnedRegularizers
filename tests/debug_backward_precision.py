"""
Debug: Trace exactly where precision is lost in the backward pass.
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


def test_pure_float64():
    """Test with EVERYTHING in float64 - no mixed precision at all."""
    print("=" * 70)
    print("TEST: Pure Float64 (no mixed precision)")
    print("=" * 70)
    
    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float64
    
    # Simple convolution as "neural network" - entirely in float64
    weight = torch.randn(1, 1, 3, 3, dtype=dtype, device=device, requires_grad=True)
    
    x = torch.randn(1, 1, 16, 16, dtype=dtype, device=device)
    y = x + torch.randn_like(x) * 0.1
    z0 = y.clone()
    
    step_size = 0.1
    lamda = 0.5
    beta = 0.8
    num_steps = 20
    
    def fixed_point_function(z, args):
        y_arg, weight_arg, lamda_arg, step_size_arg = args
        # Simple data fidelity + regularizer gradient
        grad_data = z - y_arg
        # "Regularizer" - just a convolution
        reg_out = torch.nn.functional.conv2d(z, weight_arg, padding=1)
        grad_reg = lamda_arg * reg_out
        return z - step_size_arg * (grad_data + grad_reg)
    
    args = (y, weight, lamda, step_size)
    target = torch.zeros_like(x)  # Target for loss
    
    # ========================================
    # Method 1: Unrolled with autograd
    # ========================================
    z = z0.clone()
    y_state = z.clone()
    fz = fixed_point_function(z, args)
    
    for _ in range(num_steps):
        # Forward: y1 = (1-beta)*y0 + beta*f(z)
        y_state = (1 - beta) * y_state + beta * fz
        f_y = fixed_point_function(y_state, args)
        # Forward: z1 = (1-beta)*z0 + beta*f(y1)
        z = (1 - beta) * z + beta * f_y
        fz = fixed_point_function(z, args)
    
    loss_unroll = ((z - target) ** 2).sum()
    loss_unroll.backward()
    grad_unroll = weight.grad.clone()
    
    print(f"Unrolled loss: {loss_unroll.item():.8f}")
    print(f"Unrolled grad norm: {grad_unroll.norm().item():.8e}")
    
    # ========================================
    # Method 2: Manual reversible backward
    # ========================================
    weight.grad = None
    
    # Forward pass (same as above, but without autograd)
    with torch.no_grad():
        z = z0.clone()
        y_state = z.clone()
        fz = fixed_point_function(z, args)
        
        for _ in range(num_steps):
            y_state = (1 - beta) * y_state + beta * fz
            f_y = fixed_point_function(y_state, args)
            z = (1 - beta) * z + beta * f_y
            fz = fixed_point_function(z, args)
        
        z1_final = z.clone()
        y1_final = y_state.clone()
    
    # Manual backward with reversible reconstruction
    z1 = z1_final.clone()
    y1 = y1_final.clone()
    
    # Initial gradients
    loss_manual = ((z1 - target) ** 2).sum()
    grad_z1 = 2 * (z1 - target)  # d(loss)/d(z1)
    grad_y1 = torch.zeros_like(y1)
    grad_weight = torch.zeros_like(weight)
    
    print(f"\nManual loss: {loss_manual.item():.8f}")
    
    for step in range(num_steps):
        # ---- Step 1: Compute f(y1) and VJP at y1 ----
        with torch.enable_grad():
            y1_req = y1.detach().requires_grad_(True)
            w_req = weight.detach().requires_grad_(True)
            # Recompute fixed-point function with grad
            grad_data = y1_req - y
            reg_out = torch.nn.functional.conv2d(y1_req, w_req, padding=1)
            grad_reg = lamda * reg_out
            fy1 = y1_req - step_size * (grad_data + grad_reg)
            
            # VJP at y1
            grad_outs = torch.autograd.grad(
                fy1, (y1_req, w_req), grad_outputs=grad_z1, retain_graph=True
            )
            dgrad_y1 = grad_outs[0]
            dgrad_w_y = grad_outs[1]
        
        # Update grad_y1
        grad_y1 = grad_y1 + beta * dgrad_y1
        grad_y0 = (1 - beta) * grad_y1
        
        # ---- Step 2: Reconstruct z0 ----
        with torch.no_grad():
            z0 = (z1 - beta * fy1.detach()) / (1 - beta)
        
        # ---- Step 3: Compute f(z0) and VJP at z0 ----
        with torch.enable_grad():
            z0_req = z0.detach().requires_grad_(True)
            w_req2 = weight.detach().requires_grad_(True)
            # Recompute fixed-point function with grad
            grad_data_z = z0_req - y
            reg_out_z = torch.nn.functional.conv2d(z0_req, w_req2, padding=1)
            grad_reg_z = lamda * reg_out_z
            fz0 = z0_req - step_size * (grad_data_z + grad_reg_z)
            
            # VJP at z0
            grad_outs_z = torch.autograd.grad(
                fz0, (z0_req, w_req2), grad_outputs=grad_y1, retain_graph=True
            )
            dgrad_z0 = grad_outs_z[0]
            dgrad_w_z = grad_outs_z[1]
        
        # Update grad_z0
        grad_z0 = (1 - beta) * grad_z1 + beta * dgrad_z0
        
        # ---- Step 4: Reconstruct y0 ----
        with torch.no_grad():
            y0 = (y1 - beta * fz0.detach()) / (1 - beta)
        
        # ---- Step 5: Accumulate parameter gradients ----
        grad_weight = grad_weight + beta * (dgrad_w_y + dgrad_w_z)
        
        # ---- Step 6: Move to previous state ----
        z1 = z0.detach()
        y1 = y0.detach()
        grad_z1 = grad_z0.detach()
        grad_y1 = grad_y0.detach()
        
        # Debug: check reconstruction error
        if step < 3 or step >= num_steps - 2:
            print(f"  Step {step}: z0 norm={z0.norm().item():.4f}, grad_z0 norm={grad_z0.norm().item():.4f}")
    
    print(f"\nManual grad norm: {grad_weight.norm().item():.8e}")
    
    # Compare
    grad_diff = (grad_unroll - grad_weight).abs().max().item()
    rel_diff = grad_diff / (grad_unroll.abs().max().item() + 1e-15)
    
    print(f"\n{'='*40}")
    print(f"Gradient absolute difference: {grad_diff:.2e}")
    print(f"Gradient relative difference: {rel_diff:.2e}")
    print(f"{'='*40}")
    
    if rel_diff < 1e-10:
        print("[OK] Gradients match to machine precision!")
    elif rel_diff < 1e-6:
        print("[WARN] Small gradient difference")
    else:
        print("[FAIL] Significant gradient mismatch!")
    
    return rel_diff


def test_vary_steps_pure_f64():
    """Test gradient error vs steps with pure float64."""
    print("\n" + "=" * 70)
    print("GRADIENT ERROR vs STEPS (Pure Float64)")
    print("=" * 70)
    
    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float64
    
    weight = torch.randn(1, 1, 3, 3, dtype=dtype, device=device, requires_grad=True)
    x = torch.randn(1, 1, 16, 16, dtype=dtype, device=device)
    y = x + torch.randn_like(x) * 0.1
    z0 = y.clone()
    
    step_size = 0.1
    lamda = 0.5
    target = torch.zeros_like(x)
    
    def fixed_point_function(z, args):
        y_arg, weight_arg, lamda_arg, step_size_arg = args
        grad_data = z - y_arg
        reg_out = torch.nn.functional.conv2d(z, weight_arg, padding=1)
        grad_reg = lamda_arg * reg_out
        return z - step_size_arg * (grad_data + grad_reg)
    
    args = (y, weight, lamda, step_size)
    
    print(f"\n{'Steps':>6} | {'Beta':>6} | {'Rel Error':>12} | Status")
    print("-" * 50)
    
    for num_steps in [5, 10, 15, 20, 30, 50]:
        for beta in [0.5, 0.8]:
            # Unrolled
            weight.grad = None
            z = z0.clone()
            y_state = z.clone()
            fz = fixed_point_function(z, args)
            
            for _ in range(num_steps):
                y_state = (1 - beta) * y_state + beta * fz
                f_y = fixed_point_function(y_state, args)
                z = (1 - beta) * z + beta * f_y
                fz = fixed_point_function(z, args)
            
            loss = ((z - target) ** 2).sum()
            loss.backward()
            grad_unroll = weight.grad.clone()
            
            # Manual reversible
            weight.grad = None
            with torch.no_grad():
                z = z0.clone()
                y_state = z.clone()
                fz = fixed_point_function(z, args)
                
                for _ in range(num_steps):
                    y_state = (1 - beta) * y_state + beta * fz
                    f_y = fixed_point_function(y_state, args)
                    z = (1 - beta) * z + beta * f_y
                    fz = fixed_point_function(z, args)
                
                z1 = z.clone()
                y1 = y_state.clone()
            
            grad_z1 = 2 * (z1 - target)
            grad_y1 = torch.zeros_like(y1)
            grad_weight = torch.zeros_like(weight)
            
            for _ in range(num_steps):
                with torch.enable_grad():
                    y1_req = y1.detach().requires_grad_(True)
                    w_req = weight.detach().requires_grad_(True)
                    fy1 = fixed_point_function(y1_req, (y, w_req, lamda, step_size))
                    grad_outs = torch.autograd.grad(fy1, (y1_req, w_req), grad_outputs=grad_z1)
                    dgrad_y1, dgrad_w_y = grad_outs
                
                grad_y1 = grad_y1 + beta * dgrad_y1
                grad_y0 = (1 - beta) * grad_y1
                
                with torch.no_grad():
                    z0 = (z1 - beta * fy1.detach()) / (1 - beta)
                
                with torch.enable_grad():
                    z0_req = z0.detach().requires_grad_(True)
                    w_req2 = weight.detach().requires_grad_(True)
                    fz0 = fixed_point_function(z0_req, (y, w_req2, lamda, step_size))
                    grad_outs_z = torch.autograd.grad(fz0, (z0_req, w_req2), grad_outputs=grad_y1)
                    dgrad_z0, dgrad_w_z = grad_outs_z
                
                grad_z0 = (1 - beta) * grad_z1 + beta * dgrad_z0
                
                with torch.no_grad():
                    y0 = (y1 - beta * fz0.detach()) / (1 - beta)
                
                grad_weight = grad_weight + beta * (dgrad_w_y + dgrad_w_z)
                
                z1, y1 = z0.detach(), y0.detach()
                grad_z1, grad_y1 = grad_z0.detach(), grad_y0.detach()
            
            rel_diff = (grad_unroll - grad_weight).abs().max().item() / (grad_unroll.abs().max().item() + 1e-15)
            status = "OK" if rel_diff < 1e-10 else ("WARN" if rel_diff < 1e-6 else "BAD")
            print(f"{num_steps:>6} | {beta:>6.2f} | {rel_diff:>12.2e} | {status}")


if __name__ == "__main__":
    test_pure_float64()
    test_vary_steps_pure_f64()
