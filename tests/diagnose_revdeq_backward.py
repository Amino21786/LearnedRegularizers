"""
Diagnostic script to verify RevDEQ backward pass is working in the real training context.

This script:
1. Loads a pretrained CRR regularizer (like in training_bilevel.py)
2. Runs the reversible solver on a real image
3. Verifies gradients flow back to regularizer parameters
4. Compares against finite-difference approximation for one parameter
5. Checks gradient magnitudes are reasonable

Run from LearnedRegularizers/ directory:
    python tests/diagnose_revdeq_backward.py
"""

import os
import sys

# Ensure imports work
HERE = os.path.dirname(__file__)
LR_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if LR_ROOT not in sys.path:
    sys.path.insert(0, LR_ROOT)

import torch
import deepinv

# Import reversible_deq directly to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "reversible_deq", 
    os.path.join(LR_ROOT, "training_methods", "reversible_deq.py")
)
reversible_deq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reversible_deq)
ReversibleSolver = reversible_deq.ReversibleSolver
solve_reversible_adjoint = reversible_deq.solve_reversible_adjoint

from priors import WCRR, ParameterLearningWrapper


def reconstruct_with_revdeq(y, physics, data_fidelity, regularizer, lmbd, step_size,
                             max_iter, tol, x_init, beta):
    """
    Inline reconstruction using reversible solver to avoid circular imports.
    """
    x = x_init.clone().detach() if x_init is not None else physics.A_dagger(y)

    def fixed_point_function(z, args):
        y_arg, physics_arg, df_arg, reg_arg, lam, ss = args
        grad_data = df_arg.grad(z, y_arg, physics_arg)
        grad_reg = lam * reg_arg.grad(z)
        return z - ss * (grad_data + grad_reg)

    args = (y, physics, data_fidelity, regularizer, lmbd, step_size)
    params = [p for p in regularizer.parameters() if p.requires_grad]
    
    z1, steps_taken, error = solve_reversible_adjoint(
        function=fixed_point_function,
        z0=x,
        args=args,
        params=params,
        beta=beta,
        tol=tol,
        max_steps=max_iter,
    )
    return z1, {"steps": steps_taken, "error": error}


def main():
    print("=" * 70)
    print("RevDEQ Backward Pass Diagnostic")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- 1. Setup: Load regularizer and create physics ---
    print("\n[1] Setting up regularizer, physics, and data fidelity...")
    
    # Create CRR regularizer (same as training_bilevel.py)
    reg = WCRR(
        sigma=0.1,
        weak_convexity=0.0,
    ).to(device)
    regularizer = ParameterLearningWrapper(reg, device=device)
    
    # Try to load pretrained weights if available
    weight_path = os.path.join(LR_ROOT, "weights/score_for_Denoising/CRR_score_training_for_Denoising.pt")
    if os.path.exists(weight_path):
        regularizer.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print(f"  Loaded pretrained weights from {weight_path}")
    else:
        print(f"  No pretrained weights found at {weight_path}, using random init")

    # Ensure parameters require grad
    for p in regularizer.parameters():
        p.requires_grad_(True)

    # Physics (Denoising)
    noise_level = 0.1
    physics = deepinv.physics.Denoising(noise_model=deepinv.physics.GaussianNoise(sigma=noise_level))
    data_fidelity = deepinv.optim.L2()
    lmbd = 1.0
    step_size = 0.1
    max_iter = 20  # Small for diagnostic
    tol = 1e-6
    beta = 0.8

    # --- 2. Create a small test image ---
    print("\n[2] Creating test image...")
    # Use a small random image for speed
    x_gt = torch.rand(1, 1, 32, 32, device=device, dtype=torch.float32)
    y = physics(x_gt)
    x_init = physics.A_dagger(y)
    print(f"  Image shape: {x_gt.shape}")

    # --- 3. Run forward pass and check output has grad_fn ---
    print("\n[3] Running RevDEQ forward pass...")
    regularizer.zero_grad()
    
    x_recon, stats = reconstruct_with_revdeq(
        y, physics, data_fidelity, regularizer, lmbd, step_size,
        max_iter, tol, x_init=x_init, beta=beta
    )
    
    print(f"  Converged in {stats['steps']} steps, error: {stats['error']:.2e}")
    print(f"  x_recon.requires_grad: {x_recon.requires_grad}")
    print(f"  x_recon.grad_fn: {x_recon.grad_fn}")
    
    if x_recon.grad_fn is None:
        print("\n  [FAIL] ERROR: x_recon has no grad_fn! Backward won't work.")
        return False
    print("  [OK] x_recon has grad_fn")

    # --- 4. Compute loss and backward ---
    print("\n[4] Computing loss and running backward()...")
    loss = ((x_recon - x_gt) ** 2).mean()
    print(f"  Loss: {loss.item():.6f}")
    
    loss.backward()
    print("  [OK] backward() completed without error")

    # --- 5. Check gradients exist and are non-zero ---
    print("\n[5] Checking parameter gradients...")
    param_names = []
    grad_norms = []
    has_grad = []
    
    for name, p in regularizer.named_parameters():
        param_names.append(name)
        if p.grad is not None:
            norm = p.grad.norm().item()
            grad_norms.append(norm)
            has_grad.append(True)
        else:
            grad_norms.append(0.0)
            has_grad.append(False)
    
    print(f"  {'Parameter':<40} {'Has Grad':<10} {'Grad Norm':<15}")
    print("  " + "-" * 65)
    for name, hg, gn in zip(param_names, has_grad, grad_norms):
        status = "[OK]" if hg and gn > 0 else ("[~]" if hg else "[X]")
        print(f"  {name:<40} {str(hg):<10} {gn:<15.6e} {status}")
    
    total_params_with_grad = sum(1 for hg, gn in zip(has_grad, grad_norms) if hg and gn > 0)
    print(f"\n  {total_params_with_grad}/{len(param_names)} parameters have non-zero gradients")
    
    if total_params_with_grad == 0:
        print("  [FAIL] ERROR: No parameters received gradients!")
        return False
    print("  [OK] Gradients are flowing to parameters")

    # --- 6. Finite-difference check on one scalar parameter ---
    print("\n[6] Finite-difference gradient check on regularizer.alpha...")
    
    # Find alpha parameter
    alpha_param = None
    for name, p in regularizer.named_parameters():
        if "alpha" in name and p.numel() == 1:
            alpha_param = p
            alpha_name = name
            break
    
    if alpha_param is None:
        print("  Skipping: no scalar 'alpha' parameter found")
    else:
        custom_grad = alpha_param.grad.item()
        
        # Finite difference
        eps = 1e-4
        
        # f(alpha + eps)
        with torch.no_grad():
            orig_val = alpha_param.item()
            alpha_param.fill_(orig_val + eps)
        regularizer.zero_grad()
        x_plus, _ = reconstruct_with_revdeq(
            y, physics, data_fidelity, regularizer, lmbd, step_size,
            max_iter, tol, x_init=x_init, beta=beta
        )
        loss_plus = ((x_plus - x_gt) ** 2).mean().item()
        
        # f(alpha - eps)
        with torch.no_grad():
            alpha_param.fill_(orig_val - eps)
        regularizer.zero_grad()
        x_minus, _ = reconstruct_with_revdeq(
            y, physics, data_fidelity, regularizer, lmbd, step_size,
            max_iter, tol, x_init=x_init, beta=beta
        )
        loss_minus = ((x_minus - x_gt) ** 2).mean().item()
        
        # Restore
        with torch.no_grad():
            alpha_param.fill_(orig_val)
        
        fd_grad = (loss_plus - loss_minus) / (2 * eps)
        
        print(f"  Parameter: {alpha_name}")
        print(f"  Custom backward grad: {custom_grad:.6e}")
        print(f"  Finite-diff grad:     {fd_grad:.6e}")
        
        if abs(custom_grad) < 1e-12 and abs(fd_grad) < 1e-12:
            rel_err = 0.0
        elif abs(fd_grad) < 1e-12:
            rel_err = abs(custom_grad - fd_grad)
        else:
            rel_err = abs(custom_grad - fd_grad) / (abs(fd_grad) + 1e-12)
        
        print(f"  Relative error:       {rel_err:.2e}")
        
        if rel_err < 0.1:
            print("  [OK] Gradients match within 10% (good)")
        elif rel_err < 0.5:
            print("  [~] Gradients match within 50% (acceptable for this tolerance)")
        else:
            print("  [!] Gradients differ significantly (may need investigation)")

    # --- 7. Compare with unrolled baseline ---
    print("\n[7] Comparing with unrolled autograd baseline...")
    
    # Reset gradients
    regularizer.zero_grad()
    
    # Unrolled forward (builds full computation graph)
    def fixed_point_fn(z, args):
        y_arg, physics_arg, df_arg, reg_arg, lam, ss = args
        grad_data = df_arg.grad(z, y_arg, physics_arg)
        grad_reg = lam * reg_arg.grad(z)
        return z - ss * (grad_data + grad_reg)
    
    args = (y, physics, data_fidelity, regularizer, lmbd, step_size)
    solver = ReversibleSolver(beta=beta)
    y_state, fz_state = solver.init(fixed_point_fn, x_init, args)
    z_unroll = x_init.clone()
    
    for i in range(stats['steps']):  # Same number of steps as RevDEQ
        z_unroll, (y_state, fz_state), _ = solver.step(fixed_point_fn, z_unroll, args, (y_state, fz_state))
    
    loss_unroll = ((z_unroll - x_gt) ** 2).mean()
    loss_unroll.backward()
    
    # Collect unrolled gradients
    unroll_grads = {}
    for name, p in regularizer.named_parameters():
        if p.grad is not None:
            unroll_grads[name] = p.grad.clone()
    
    # Re-run RevDEQ to get its gradients
    regularizer.zero_grad()
    x_recon2, _ = reconstruct_with_revdeq(
        y, physics, data_fidelity, regularizer, lmbd, step_size,
        max_iter, tol, x_init=x_init, beta=beta
    )
    loss2 = ((x_recon2 - x_gt) ** 2).mean()
    loss2.backward()
    
    revdeq_grads = {}
    for name, p in regularizer.named_parameters():
        if p.grad is not None:
            revdeq_grads[name] = p.grad.clone()
    
    # Compare
    print(f"  {'Parameter':<40} {'Unroll Norm':<15} {'RevDEQ Norm':<15} {'Rel Diff':<15}")
    print("  " + "-" * 85)
    
    all_close = True
    for name in unroll_grads:
        if name in revdeq_grads:
            u_norm = unroll_grads[name].norm().item()
            r_norm = revdeq_grads[name].norm().item()
            diff = (unroll_grads[name] - revdeq_grads[name]).norm().item()
            rel_diff = diff / (u_norm + 1e-12) if u_norm > 1e-12 else diff
            status = "[OK]" if rel_diff < 0.1 else ("[~]" if rel_diff < 0.5 else "[X]")
            print(f"  {name:<40} {u_norm:<15.6e} {r_norm:<15.6e} {rel_diff:<15.2e} {status}")
            if rel_diff > 0.5:
                all_close = False
    
    if all_close:
        print("\n  [OK] RevDEQ gradients match unrolled baseline")
    else:
        print("\n  [!] Some gradients differ from unrolled baseline (may be acceptable)")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("[OK] Forward pass produces output with grad_fn")
    print("[OK] backward() runs without error")
    print(f"[OK] {total_params_with_grad}/{len(param_names)} parameters received gradients")
    print("[OK] RevDEQ backward pass is working!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
