"""
Step-by-Step Verification: RevDEQ vs Unrolled Backpropagation

This script provides a detailed side-by-side comparison of how gradients are computed
in RevDEQ versus standard unrolled backpropagation, showing the equivalence at each step.

The key insight is that RevDEQ's backward pass:
1. Reconstructs states z_k, y_k from z_{k+1}, y_{k+1} using the reversible formula
2. Computes VJPs (vector-Jacobian products) at each reconstructed state
3. Accumulates parameter gradients as it goes backward

This is mathematically equivalent to storing all intermediate states and using autograd,
but much more memory efficient.
"""

import os
import sys
import torch
import torch.nn as nn

# Ensure imports work
HERE = os.path.dirname(__file__)
LR_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(LR_ROOT, ".."))
for p in (LR_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from training_methods.reversible_deq import ReversibleSolver, solve_reversible_adjoint


def step_by_step_comparison(num_steps=3, beta=0.8):
    """
    Perform a detailed step-by-step comparison of RevDEQ vs unrolled backpropagation.
    """
    torch.manual_seed(42)
    dtype = torch.float64
    
    print("=" * 80)
    print("STEP-BY-STEP COMPARISON: RevDEQ vs Unrolled Backpropagation")
    print("=" * 80)
    print(f"\nSettings: {num_steps} steps, beta={beta}")
    print()
    
    # Create a simple parameterized function
    # f(z, w) = tanh(w * z) - this is a contraction for |w| < 1
    w = torch.nn.Parameter(torch.tensor(0.3, dtype=dtype))
    
    def f(z, args):
        (w_,) = args
        return torch.tanh(w_ * z)
    
    z0 = torch.tensor([[0.8]], dtype=dtype)
    target = torch.tensor([[0.1]], dtype=dtype)
    
    # ==========================================================================
    # FORWARD PASS (same for both methods)
    # ==========================================================================
    print("-" * 80)
    print("FORWARD PASS")
    print("-" * 80)
    print("\nReversible iteration formula:")
    print("  y_{k+1} = (1-beta)*y_k + beta*f(z_k)")
    print("  z_{k+1} = (1-beta)*z_k + beta*f(y_{k+1})")
    print()
    
    solver = ReversibleSolver(beta=beta)
    y, fz = solver.init(f, z0, (w,))
    z = z0.clone()
    
    # Store trajectory for analysis
    trajectory = [(z.item(), y.item())]
    
    print(f"Step 0: z_0 = {z.item():.6f}, y_0 = {y.item():.6f}")
    
    for step in range(num_steps):
        z_old = z.item()
        y_old = y.item()
        z, (y, fz), error = solver.step(f, z, (w,), (y, fz))
        
        f_z_old = torch.tanh(w * z_old).item()
        f_y_new = torch.tanh(w * y.detach()).item()
        
        print(f"Step {step+1}:")
        print(f"  f(z_{step}) = tanh({w.item():.4f} * {z_old:.6f}) = {f_z_old:.6f}")
        print(f"  y_{step+1} = (1-{beta})*{y_old:.6f} + {beta}*{f_z_old:.6f} = {y.item():.6f}")
        print(f"  f(y_{step+1}) = tanh({w.item():.4f} * {y.item():.6f}) = {f_y_new:.6f}")
        print(f"  z_{step+1} = (1-{beta})*{z_old:.6f} + {beta}*{f_y_new:.6f} = {z.item():.6f}")
        
        trajectory.append((z.item(), y.item()))
    
    print(f"\nFinal: z_{num_steps} = {z.item():.6f}")
    
    loss = ((z - target) ** 2).sum()
    print(f"Loss = ||z_{num_steps} - target||^2 = ||{z.item():.6f} - {target.item():.6f}||^2 = {loss.item():.6f}")
    
    # ==========================================================================
    # BACKWARD PASS: UNROLLED (standard autograd)
    # ==========================================================================
    print()
    print("-" * 80)
    print("BACKWARD PASS: Unrolled (Standard Autograd)")
    print("-" * 80)
    print("\nUsing PyTorch autograd with retained computation graph.")
    print("Autograd automatically computes d(loss)/dw by chain rule through all operations.")
    
    # Run forward again with gradient tracking
    w.grad = None
    y, fz = solver.init(f, z0, (w,))
    z = z0.clone()
    for _ in range(num_steps):
        z, (y, fz), _ = solver.step(f, z, (w,), (y, fz))
    
    loss = ((z - target) ** 2).sum()
    loss.backward()
    
    grad_w_unrolled = w.grad.item()
    print(f"\nAutograd result: d(loss)/dw = {grad_w_unrolled:.10f}")
    
    # ==========================================================================
    # BACKWARD PASS: RevDEQ (manual reconstruction)
    # ==========================================================================
    print()
    print("-" * 80)
    print("BACKWARD PASS: RevDEQ (Reversible Adjoint)")
    print("-" * 80)
    print("\nKey idea: Reconstruct previous states from current states using:")
    print("  z_k = (z_{k+1} - beta*f(y_{k+1})) / (1-beta)")
    print("  y_k = (y_{k+1} - beta*f(z_k)) / (1-beta)")
    print()
    print("Then compute VJPs (vector-Jacobian products) at each reconstructed point.")
    
    # Final state
    z_curr = trajectory[-1][0]
    y_curr = trajectory[-1][1]
    
    # Initial gradient from loss
    grad_z = 2 * (z_curr - target.item())  # d(loss)/dz_final
    grad_y = 0.0
    grad_w_revdeq = 0.0
    
    print(f"\nInitial: grad_z_{num_steps} = 2*(z_{num_steps} - target) = 2*({z_curr:.6f} - {target.item():.6f}) = {grad_z:.6f}")
    print()
    
    for step in range(num_steps, 0, -1):
        print(f"Backward step {step} -> {step-1}:")
        
        # Reconstruct z_{k-1} from z_k and f(y_k)
        f_y_curr = torch.tanh(torch.tensor(w.item() * y_curr, dtype=dtype)).item()
        z_prev = (z_curr - beta * f_y_curr) / (1 - beta)
        
        print(f"  Reconstruct z_{step-1}:")
        print(f"    f(y_{step}) = tanh({w.item():.4f} * {y_curr:.6f}) = {f_y_curr:.6f}")
        print(f"    z_{step-1} = (z_{step} - beta*f(y_{step})) / (1-beta)")
        print(f"            = ({z_curr:.6f} - {beta}*{f_y_curr:.6f}) / {1-beta}")
        print(f"            = {z_prev:.6f}")
        
        # VJP at y_curr: df/dy = w * sech^2(w*y)
        sech2_y = 1 - torch.tanh(torch.tensor(w.item() * y_curr, dtype=dtype)).item() ** 2
        df_dy = w.item() * sech2_y
        df_dw_at_y = y_curr * sech2_y
        
        # Gradient updates
        dgrad_y = df_dy * grad_z
        dgrad_w_y = df_dw_at_y * grad_z
        
        grad_y_new = grad_y + beta * dgrad_y
        grad_y_prev = (1 - beta) * grad_y_new
        
        print(f"  VJP at y_{step}:")
        print(f"    df/dy = w * sech^2(w*y) = {w.item():.4f} * {sech2_y:.6f} = {df_dy:.6f}")
        print(f"    df/dw at y = y * sech^2(w*y) = {y_curr:.6f} * {sech2_y:.6f} = {df_dw_at_y:.6f}")
        print(f"    dgrad_y = df/dy * grad_z = {df_dy:.6f} * {grad_z:.6f} = {dgrad_y:.6f}")
        print(f"    dgrad_w_y = df/dw * grad_z = {df_dw_at_y:.6f} * {grad_z:.6f} = {dgrad_w_y:.6f}")
        
        # Reconstruct y_{k-1} from y_k and f(z_{k-1})
        f_z_prev = torch.tanh(torch.tensor(w.item() * z_prev, dtype=dtype)).item()
        y_prev = (y_curr - beta * f_z_prev) / (1 - beta)
        
        print(f"  Reconstruct y_{step-1}:")
        print(f"    f(z_{step-1}) = tanh({w.item():.4f} * {z_prev:.6f}) = {f_z_prev:.6f}")
        print(f"    y_{step-1} = (y_{step} - beta*f(z_{step-1})) / (1-beta)")
        print(f"            = ({y_curr:.6f} - {beta}*{f_z_prev:.6f}) / {1-beta}")
        print(f"            = {y_prev:.6f}")
        
        # VJP at z_prev
        sech2_z = 1 - f_z_prev ** 2
        df_dz = w.item() * sech2_z
        df_dw_at_z = z_prev * sech2_z
        
        dgrad_z = df_dz * grad_y_new
        dgrad_w_z = df_dw_at_z * grad_y_new
        
        grad_z_new = (1 - beta) * grad_z + beta * dgrad_z
        
        print(f"  VJP at z_{step-1}:")
        print(f"    df/dz = w * sech^2(w*z) = {w.item():.4f} * {sech2_z:.6f} = {df_dz:.6f}")
        print(f"    df/dw at z = z * sech^2(w*z) = {z_prev:.6f} * {sech2_z:.6f} = {df_dw_at_z:.6f}")
        print(f"    dgrad_z = df/dz * grad_y = {df_dz:.6f} * {grad_y_new:.6f} = {dgrad_z:.6f}")
        print(f"    dgrad_w_z = df/dw * grad_y = {df_dw_at_z:.6f} * {grad_y_new:.6f} = {dgrad_w_z:.6f}")
        
        # Accumulate parameter gradient
        grad_w_contrib = beta * (dgrad_w_y + dgrad_w_z)
        grad_w_revdeq += grad_w_contrib
        
        print(f"  Gradient accumulation:")
        print(f"    grad_w += beta * (dgrad_w_y + dgrad_w_z)")
        print(f"           += {beta} * ({dgrad_w_y:.6f} + {dgrad_w_z:.6f})")
        print(f"           += {grad_w_contrib:.6f}")
        print(f"    cumulative grad_w = {grad_w_revdeq:.10f}")
        
        # Update for next iteration
        z_curr = z_prev
        y_curr = y_prev
        grad_z = grad_z_new
        grad_y = grad_y_prev
        
        # Verify reconstruction matches forward trajectory
        expected_z = trajectory[step-1][0]
        expected_y = trajectory[step-1][1]
        print(f"  Verification: z_{step-1} = {z_prev:.6f} (expected {expected_z:.6f}, diff = {abs(z_prev-expected_z):.2e})")
        print(f"                y_{step-1} = {y_prev:.6f} (expected {expected_y:.6f}, diff = {abs(y_prev-expected_y):.2e})")
        print()
    
    # ==========================================================================
    # COMPARISON
    # ==========================================================================
    print("-" * 80)
    print("COMPARISON")
    print("-" * 80)
    print(f"\nUnrolled autograd:   d(loss)/dw = {grad_w_unrolled:.10f}")
    print(f"RevDEQ reconstruction: d(loss)/dw = {grad_w_revdeq:.10f}")
    print(f"Difference: {abs(grad_w_unrolled - grad_w_revdeq):.2e}")
    
    if abs(grad_w_unrolled - grad_w_revdeq) < 1e-8:
        print("\nSUCCESS: RevDEQ backward pass is mathematically equivalent to unrolled autograd!")
    else:
        print("\nWARNING: Discrepancy detected!")
    
    # ==========================================================================
    # Also verify with actual RevDEQ implementation
    # ==========================================================================
    print()
    print("-" * 80)
    print("VERIFICATION WITH ACTUAL RevDEQ IMPLEMENTATION")
    print("-" * 80)
    
    w.grad = None
    z_rev, steps, err = solve_reversible_adjoint(
        f, z0, args=(w,), params=[w], beta=beta, tol=-1.0, max_steps=num_steps
    )
    loss_rev = ((z_rev - target) ** 2).sum()
    loss_rev.backward()
    
    grad_w_impl = w.grad.item()
    print(f"\nActual RevDEQ implementation: d(loss)/dw = {grad_w_impl:.10f}")
    print(f"Difference from unrolled: {abs(grad_w_unrolled - grad_w_impl):.2e}")
    print(f"Difference from manual: {abs(grad_w_revdeq - grad_w_impl):.2e}")
    
    if abs(grad_w_unrolled - grad_w_impl) < 1e-8:
        print("\nSUCCESS: All three methods agree!")
    
    return True


def main():
    print("\n" + "=" * 80)
    print("REVDEQ STEP-BY-STEP VERIFICATION")
    print("=" * 80 + "\n")
    
    # Run with 3 steps for detailed visualization
    step_by_step_comparison(num_steps=3, beta=0.8)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The RevDEQ backward pass works by:

1. Starting from the final state (z_n, y_n) and the gradient d(loss)/dz_n

2. For each step k from n down to 1:
   a) Reconstruct z_{k-1} = (z_k - beta*f(y_k)) / (1-beta)
   b) Compute VJP at y_k to get gradients w.r.t. y_k and parameters
   c) Reconstruct y_{k-1} = (y_k - beta*f(z_{k-1})) / (1-beta)
   d) Compute VJP at z_{k-1} to get gradients w.r.t. z_{k-1} and parameters
   e) Accumulate parameter gradients: grad_w += beta * (vjp_w_at_y + vjp_w_at_z)
   f) Update state gradients for next iteration

3. The final accumulated grad_w equals what autograd would compute

Key benefits of RevDEQ:
- O(1) memory: Only need to store final state, not entire trajectory
- Same gradients: Mathematically equivalent to unrolled backpropagation
- Enables deep equilibrium models with many iterations

This is why RevDEQ is particularly useful for bilevel optimization where the
lower-level problem may require many iterations to converge.
""")


if __name__ == "__main__":
    main()
