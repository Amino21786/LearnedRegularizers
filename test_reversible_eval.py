"""Quick test for reversible DEQ evaluation."""
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from dataset import get_dataset
from priors import WCRR, ParameterLearningWrapper
from operators import get_evaluation_setting
from evaluation.reconstruct_reversible import reconstruct_reversible

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CRR regularizer
reg = WCRR(sigma=0.1, weak_convexity=0.0).to(device)
regularizer = ParameterLearningWrapper(reg, device=device)

# Load RevDEQ weights
weights = torch.load(
    "weights/bilevel_Denoising/CRR_bilevel_RevDEQ_for_Denoising.pt",
    map_location=device,
    weights_only=True,
)
regularizer.load_state_dict(weights)
regularizer.eval()

# Get evaluation setting
dataset, physics, data_fidelity = get_evaluation_setting("Denoising", device)

# Get one test image
x = dataset[0].unsqueeze(0).to(torch.float32).to(device)
y = physics(x)

print(f"Image shape: {x.shape}")
print("Running reversible reconstruction (max 5 iters)...")

# Run reconstruction with limited iterations for quick test
recon, stats = reconstruct_reversible(
    y, physics, data_fidelity, regularizer,
    lamda=1.0, step_size=0.1, max_iter=5, tol=1e-4,
    beta=0.8, verbose=True, return_stats=True,
)

# Compute PSNR
from deepinv.loss.metric import PSNR
psnr = PSNR()
psnr_val = psnr(recon, x).item()

print(f"\nResults:")
print(f"  Steps: {stats['steps']}")
print(f"  Error: {stats['error']:.6e}")
print(f"  PSNR: {psnr_val:.2f} dB")
print("\nTest passed!")
