"""
Debug: Trace forward vs backward state reconstruction exactly.
"""

import torch

def test_reconstruction():
    """Check if backward reconstruction exactly recovers forward states."""
    print("=" * 70)
    print("FORWARD vs BACKWARD STATE RECONSTRUCTION")
    print("=" * 70)
    
    torch.manual_seed(42)
    dtype = torch.float64
    
    weight = torch.randn(1, 1, 3, 3, dtype=dtype, requires_grad=False)
    y_data = torch.randn(1, 1, 16, 16, dtype=dtype)
    z0_init = y_data.clone()
    
    step_size = 0.1
    lamda = 0.5
    
    def f(z):
        grad_data = z - y_data
        reg_out = torch.nn.functional.conv2d(z, weight, padding=1)
        grad_reg = lamda * reg_out
        return z - step_size * (grad_data + grad_reg)
    
    # Test for different beta values
    for beta in [0.5, 0.8, 0.9]:
        print(f"\n{'='*60}")
        print(f"BETA = {beta}")
        print(f"{'='*60}")
        
        num_steps = 10
        
        # Forward pass - store all states
        z_history = [z0_init.clone()]
        y_history = [z0_init.clone()]  # y0 = z0
        fz_history = [f(z0_init)]  # f(z0)
        
        z = z0_init.clone()
        y = z0_init.clone()
        fz = f(z)
        
        for step in range(num_steps):
            # y1 = (1-beta)*y0 + beta*f(z0)
            y = (1 - beta) * y + beta * fz
            # f(y1)
            fy = f(y)
            # z1 = (1-beta)*z0 + beta*f(y1)
            z = (1 - beta) * z + beta * fy
            # f(z1) for next step
            fz = f(z)
            
            z_history.append(z.clone())
            y_history.append(y.clone())
            fz_history.append(fz.clone())
        
        # Backward pass - reconstruct states
        z_recon = z_history[-1].clone()
        y_recon = y_history[-1].clone()
        
        print(f"\n{'Step':>4} | {'z error':>12} | {'y error':>12} | {'z recon norm':>14}")
        print("-" * 55)
        
        max_z_err = 0
        max_y_err = 0
        
        for step in range(num_steps, 0, -1):
            # Reconstruct z_{k-1} from z_k
            # z_k = (1-beta)*z_{k-1} + beta*f(y_k)
            # z_{k-1} = (z_k - beta*f(y_k)) / (1-beta)
            fy_recon = f(y_recon)
            z_prev_recon = (z_recon - beta * fy_recon) / (1 - beta)
            
            # Reconstruct y_{k-1} from y_k
            # y_k = (1-beta)*y_{k-1} + beta*f(z_{k-1})
            # y_{k-1} = (y_k - beta*f(z_{k-1})) / (1-beta)
            fz_prev_recon = f(z_prev_recon)
            y_prev_recon = (y_recon - beta * fz_prev_recon) / (1 - beta)
            
            # Compare with stored values
            z_true = z_history[step - 1]
            y_true = y_history[step - 1]
            
            z_err = (z_prev_recon - z_true).abs().max().item()
            y_err = (y_prev_recon - y_true).abs().max().item()
            
            max_z_err = max(max_z_err, z_err)
            max_y_err = max(max_y_err, y_err)
            
            if step <= 5 or step == num_steps:
                print(f"{step:>4} | {z_err:>12.2e} | {y_err:>12.2e} | {z_prev_recon.norm().item():>14.2f}")
            
            # Step back
            z_recon = z_prev_recon
            y_recon = y_prev_recon
        
        print(f"\nMax z reconstruction error: {max_z_err:.2e}")
        print(f"Max y reconstruction error: {max_y_err:.2e}")
        
        if max_z_err < 1e-10:
            print("[OK] Reconstruction is exact!")
        elif max_z_err < 1e-5:
            print("[WARN] Small reconstruction error")
        else:
            print("[FAIL] Significant reconstruction error!")


def test_longer_runs():
    """Test reconstruction error vs number of steps."""
    print("\n\n" + "=" * 70)
    print("RECONSTRUCTION ERROR vs STEPS")
    print("=" * 70)
    
    torch.manual_seed(42)
    dtype = torch.float64
    
    weight = torch.randn(1, 1, 3, 3, dtype=dtype)
    y_data = torch.randn(1, 1, 16, 16, dtype=dtype)
    z0_init = y_data.clone()
    
    step_size = 0.1
    lamda = 0.5
    
    def f(z):
        grad_data = z - y_data
        reg_out = torch.nn.functional.conv2d(z, weight, padding=1)
        grad_reg = lamda * reg_out
        return z - step_size * (grad_data + grad_reg)
    
    print(f"\n{'Steps':>6} | {'Beta':>6} | {'Max Recon Err':>14} | {'Theory':>14}")
    print("-" * 55)
    
    for num_steps in [5, 10, 15, 20, 30]:
        for beta in [0.5, 0.8]:
            # Forward
            z_history = [z0_init.clone()]
            y_history = [z0_init.clone()]
            
            z = z0_init.clone()
            y = z0_init.clone()
            fz = f(z)
            
            for _ in range(num_steps):
                y = (1 - beta) * y + beta * fz
                fy = f(y)
                z = (1 - beta) * z + beta * fy
                fz = f(z)
                z_history.append(z.clone())
                y_history.append(y.clone())
            
            # Backward
            z_recon = z_history[-1].clone()
            y_recon = y_history[-1].clone()
            max_err = 0
            
            for step in range(num_steps, 0, -1):
                fy_recon = f(y_recon)
                z_prev_recon = (z_recon - beta * fy_recon) / (1 - beta)
                fz_prev_recon = f(z_prev_recon)
                y_prev_recon = (y_recon - beta * fz_prev_recon) / (1 - beta)
                
                z_true = z_history[step - 1]
                z_err = (z_prev_recon - z_true).abs().max().item()
                max_err = max(max_err, z_err)
                
                z_recon = z_prev_recon
                y_recon = y_prev_recon
            
            theory = 1e-15 * ((1/(1-beta)) ** num_steps)
            print(f"{num_steps:>6} | {beta:>6.2f} | {max_err:>14.2e} | {theory:>14.2e}")


if __name__ == "__main__":
    test_reconstruction()
    test_longer_runs()
