"""
Comparison script for JFB and RevDEQ hypergradient methods on CT (LoDoPaB dataset).

This script:
1. Trains regularizers using JFB and RevDEQ methods with normalized function evaluations
2. Runs post-training evaluation using variational reconstruction
3. Outputs JSON files compatible with plot_comparison_results.py

Usage:
    # Small-scale run (default - for testing)
    python compare_hypergradient_ct.py --epochs 2 --regularizer_name CRR --train_limit 50 --val_limit 20 --eval_limit 10
    
    # Full run with pretrained weights
    python compare_hypergradient_ct.py --epochs 4 --regularizer_name CRR
    
    # Train from scratch
    python compare_hypergradient_ct.py --epochs 4 --no_load_pretrain --pretrain_epochs 10
"""

import os
import sys
import argparse
import datetime
import logging
import json
import torch
import numpy as np
from copy import deepcopy

from training_methods.bilevel_training import bilevel_training
from training_methods.score_training import score_training
from dataset import get_dataset
from priors import (
    ParameterLearningWrapper,
    WCRR,
    ICNNPrior,
    IDCNNPrior,
    LSR,
    TDV,
    LocalAR,
)
from torchvision.transforms import (
    RandomCrop,
    RandomVerticalFlip,
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    RandomApply,
    RandomRotation,
)
from hyperparameters.hyperparameters_bilevel import get_bilevel_hyperparameters
from operators import get_operator, get_evaluation_setting
from evaluation import evaluate
import deepinv


def create_regularizer(regularizer_name, device):
    """Create regularizer based on name."""
    if regularizer_name == "CRR":
        reg = WCRR(sigma=0.1, weak_convexity=0.0).to(device)
    elif regularizer_name == "WCRR":
        reg = WCRR(sigma=0.1, weak_convexity=1.0).to(device)
    elif regularizer_name == "ICNN":
        reg = ICNNPrior(in_channels=1, channels=32, device=device, kernel_size=5).to(device)
    elif regularizer_name == "IDCNN":
        reg = IDCNNPrior(in_channels=1, channels=32, device=device, kernel_size=5).to(device)
    elif regularizer_name == "TDV":
        reg = TDV(in_channels=1, num_features=16).to(device)
    elif regularizer_name == "LSR":
        reg = LSR(device=device).to(device)
    else:
        raise ValueError(f"Unknown regularizer: {regularizer_name}")
    return ParameterLearningWrapper(reg, device=device)


def generate_run_id(log_dir: str, problem: str, regularizer: str) -> str:
    """
    Generate a unique run ID based on date and auto-incrementing run number.
    
    Format: YYYYMMDD_runN (e.g., 20260122_run1, 20260122_run2)
    """
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    base_pattern = f"comparison_{problem}_{regularizer}_{date_str}_run"
    
    # Find existing run numbers for today
    existing_runs = []
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            if filename.startswith(base_pattern):
                try:
                    parts = filename.replace(".log", "").replace(".json", "").split("_run")
                    if len(parts) == 2:
                        run_num = int(parts[1])
                        existing_runs.append(run_num)
                except (ValueError, IndexError):
                    pass
    
    next_run = max(existing_runs, default=0) + 1
    return f"{date_str}_run{next_run}"


def run_evaluation(
    regularizer,
    device,
    eval_limit=None,
    logger=None,
):
    """
    Run post-training evaluation using variational reconstruction.
    
    Returns dict with evaluation metrics.
    """
    # Get evaluation setting for CT
    dataset, physics, data_fidelity = get_evaluation_setting("CT", device)
    
    # Optionally limit evaluation dataset
    if eval_limit is not None and eval_limit < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(eval_limit))
    
    if logger:
        logger.info(f"Running evaluation on {len(dataset)} test images...")
    
    # Evaluation parameters (from variational_reconstruction.py)
    step_size = 1e-1
    max_iter = 1000
    tol = 1e-4
    
    # Run evaluation
    mean_psnr, x_out, y_out, recon_out = evaluate(
        physics=physics,
        data_fidelity=data_fidelity,
        dataset=dataset,
        regularizer=regularizer,
        lmbd=1.0,
        step_size=step_size,
        max_iter=max_iter,
        tol=tol,
        only_first=False,
        adaptive_range=True,  # CT uses adaptive range PSNR
        device=device,
        adam=False,
        verbose=False,
        save_path=None,
        logger=logger,
    )
    
    return {
        "mean_psnr": float(mean_psnr),
        "num_test_images": len(dataset),
    }


def run_comparison(args):
    """Run comparison of JFB and RevDEQ hypergradient methods on CT."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Setup logging with user-friendly filename
    log_dir = "comparison_logs"
    os.makedirs(log_dir, exist_ok=True)
    run_id = generate_run_id(log_dir, args.problem, args.regularizer_name)
    log_file = os.path.join(log_dir, f"comparison_{args.problem}_{args.regularizer_name}_{run_id}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    # Get hyperparameters
    hyper_params = get_bilevel_hyperparameters(args.regularizer_name, args.problem)
    
    # Setup physics and data fidelity for CT
    # Training uses different settings than evaluation (smaller images, fewer angles)
    noise_level = 0.02
    physics = deepinv.physics.Tomography(
        img_width=128,
        angles=30,
        noise_model=deepinv.physics.GaussianNoise(sigma=noise_level),
        device=device,
    )
    lmbd = 1
    data_fidelity = deepinv.optim.L2()
    
    # Load datasets
    transform_train = Compose([
        RandomCrop(128),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomApply([RandomRotation((90, 90))], 0.5),
    ])
    transform_val = CenterCrop(128)
    
    # Load LoDoPaB dataset
    train_dataset = get_dataset("LoDoPaB", test=False, transform=transform_train)
    val_dataset = get_dataset("LoDoPaB", test=False, transform=transform_val)
    
    # Split into train/val (90%/10%)
    test_ratio = 0.1
    test_len = int(len(train_dataset) * test_ratio)
    train_len = len(train_dataset) - test_len
    
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(val_dataset)))
    
    # Optionally limit dataset size for quick testing
    if args.train_limit is not None and args.train_limit < len(train_set):
        train_set = torch.utils.data.Subset(train_set, range(args.train_limit))
    if args.val_limit is not None and args.val_limit < len(val_set):
        val_set = torch.utils.data.Subset(val_set, range(args.val_limit))
    
    logger.info(f"Dataset: LoDoPaB")
    logger.info(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}")
    
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    
    # Also create a pretrain dataloader for score training (larger batch size)
    pretrain_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    
    # Paths for pretrained weights
    pretrain_path = f"weights/score_for_{args.problem}/{args.regularizer_name}_score_training_for_{args.problem}.pt"
    param_fit_path = f"weights/score_parameter_fitting_for_{args.problem}/{args.regularizer_name}_fitted_parameters_with_IFT_for_{args.problem}.pt"
    
    # Create a base regularizer that will be pretrained (or loaded) and cloned for each method
    base_regularizer = None
    
    if args.load_pretrain:
        # Try to load existing pretrained weights
        if os.path.exists(pretrain_path):
            base_regularizer = create_regularizer(args.regularizer_name, device)
            base_regularizer.load_state_dict(torch.load(pretrain_path, map_location=device, weights_only=True))
            logger.info(f"Loaded pretrained weights from {pretrain_path}")
        else:
            logger.warning(f"Pretrained weights not found at {pretrain_path}, will train from scratch")
            args.load_pretrain = False
    
    if not args.load_pretrain:
        # Train regularizer from scratch using score training
        logger.info("\n" + "=" * 70)
        logger.info("SCORE PRETRAINING (training regularizer from scratch)")
        logger.info("=" * 70)
        
        base_regularizer = create_regularizer(args.regularizer_name, device)
        
        # Enable gradients for pretraining
        for p in base_regularizer.parameters():
            p.requires_grad_(True)
        
        # Use hyperparameters for pretraining
        pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else hyper_params.pretrain_epochs
        
        if pretrain_epochs > 0:
            logger.info(f"Pretraining for {pretrain_epochs} epochs...")
            base_regularizer, pt_loss_train, pt_loss_val, pt_psnr_train, pt_psnr_val = score_training(
                base_regularizer,
                pretrain_dataloader,
                val_dataloader,
                sigma=hyper_params.score_sigma,
                epochs=pretrain_epochs,
                lr=hyper_params.pretrain_lr,
                weight_decay=hyper_params.pretrain_weight_decay,
                lr_decay=0.1 ** (1 / max(pretrain_epochs, 1)),
                device=device,
                validation_epochs=5,
                logger=logger,
                adabelief=hyper_params.adabelief,
                dynamic_range_psnr=True,
                model_selection=False,
            )
            logger.info(f"Pretraining complete. Final PSNR: {pt_psnr_val[-1]:.2f} dB")
            
            # Save pretrained weights
            os.makedirs(os.path.dirname(pretrain_path), exist_ok=True)
            torch.save(base_regularizer.state_dict(), pretrain_path)
            logger.info(f"Saved pretrained weights to {pretrain_path}")
        else:
            logger.info("Skipping pretraining (pretrain_epochs=0)")
    
    # Methods to compare with their lower-level max iterations
    # RevDEQ: 2 f-evals per step, JFB: 1 f-eval per step
    # NOTE: RevDEQ uses fixed step size, while nmAPG (JFB) uses backtracking.
    # For CT with lmbd=150, RevDEQ needs a much smaller step size to avoid divergence.
    methods = {
        "JFB": {"max_iter": args.jfb_max_iter, "f_evals_per_step": 1, "step_size": 1e-1},
        "RevDEQ": {"max_iter": args.revdeq_max_iter, "f_evals_per_step": 2, "step_size": args.revdeq_step_size},
    }
    
    results = {}
    trained_regularizers = {}  # Store trained regularizers for evaluation
    
    logger.info("=" * 70)
    logger.info("HYPERGRADIENT METHOD COMPARISON (CT)")
    logger.info("=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Problem: {args.problem}")
    logger.info(f"Regularizer: {args.regularizer_name}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Validation epochs: {args.validation_epochs}")
    logger.info(f"Load pretrained weights: {args.load_pretrain}")
    logger.info(f"Load parameter-fitted weights: {args.load_param_fit}")
    logger.info(f"RevDEQ beta: {args.revdeq_beta}")
    logger.info(f"RevDEQ step size: {args.revdeq_step_size}")
    logger.info(f"Function evaluations per batch:")
    for method, config in methods.items():
        f_evals = config["max_iter"] * config["f_evals_per_step"]
        logger.info(f"  {method}: {config['max_iter']} steps x {config['f_evals_per_step']} = {f_evals} f-evals, step_size={config['step_size']}")
    logger.info("=" * 70)
    
    for method_name, config in methods.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Running {method_name}")
        logger.info(f"{'='*70}")
        
        # Clone base regularizer for this method (so each method starts from same point)
        regularizer = create_regularizer(args.regularizer_name, device)
        if base_regularizer is not None:
            regularizer.load_state_dict(base_regularizer.state_dict())
            logger.info("Using pretrained base regularizer")
        
        # Optionally load parameter-fitted weights (overrides pretrain)
        if args.load_param_fit and os.path.exists(param_fit_path):
            regularizer.load_state_dict(torch.load(param_fit_path, map_location=device, weights_only=True))
            logger.info(f"Loaded parameter-fitted weights from {param_fit_path}")
        
        # Enable gradients
        for p in regularizer.parameters():
            p.requires_grad_(True)
        
        # Run bilevel training
        try:
            start_time = datetime.datetime.now()
            
            regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
                regularizer,
                physics,
                data_fidelity,
                lmbd,
                train_dataloader,
                val_dataloader,
                epochs=args.epochs,
                mode=method_name,
                lower_level_step_size=config["step_size"],
                lower_level_max_iter=config["max_iter"],
                lower_level_tol_train=1e-4,
                lower_level_tol_val=1e-4,
                lr=hyper_params.lr,
                lr_decay=0.1 ** (1 / args.epochs) if args.epochs > 0 else 1.0,
                reg=False,
                reg_para=hyper_params.jacobian_regularization_parameter,
                device=device,
                verbose=False,
                logger=logger,
                adabelief=hyper_params.adabelief,
                dynamic_range_psnr=True,
                validation_epochs=args.validation_epochs,
                revdeq_beta=args.revdeq_beta,
                use_embedded_beta=args.use_embedded_beta,
            )
            
            end_time = datetime.datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            # Store trained regularizer for evaluation
            trained_regularizers[method_name] = deepcopy(regularizer)
            
            results[method_name] = {
                "loss_train": loss_train,
                "loss_val": loss_val,
                "psnr_train": psnr_train,
                "psnr_val": psnr_val,
                "training_duration_seconds": training_duration,
                "max_iter": config["max_iter"],
                "f_evals_per_step": config["f_evals_per_step"],
                "total_f_evals_per_batch": config["max_iter"] * config["f_evals_per_step"],
            }
            
            logger.info(f"\n{method_name} training completed in {training_duration:.1f}s")
            logger.info(f"  Final Train Loss: {loss_train[-1]:.4f}")
            logger.info(f"  Final Train PSNR: {psnr_train[-1]:.2f} dB")
            logger.info(f"  Final Val Loss: {loss_val[-1]:.4f}")
            logger.info(f"  Final Val PSNR: {psnr_val[-1]:.2f} dB")
            
            # Save trained weights
            weights_dir = f"weights/bilevel_{args.problem}"
            os.makedirs(weights_dir, exist_ok=True)
            weights_path = os.path.join(weights_dir, f"{args.regularizer_name}_bilevel_{method_name}_for_{args.problem}_{run_id}.pt")
            torch.save(regularizer.state_dict(), weights_path)
            results[method_name]["weights_path"] = weights_path
            logger.info(f"  Saved weights to {weights_path}")
            
        except Exception as e:
            logger.error(f"{method_name} training failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[method_name] = {"error": str(e)}
    
    # ---- Post-training Evaluation ----
    if args.run_evaluation:
        logger.info("\n" + "=" * 70)
        logger.info("POST-TRAINING EVALUATION (Variational Reconstruction)")
        logger.info("=" * 70)
        
        for method_name, regularizer in trained_regularizers.items():
            logger.info(f"\nEvaluating {method_name}...")
            
            try:
                eval_start = datetime.datetime.now()
                
                eval_results = run_evaluation(
                    regularizer,
                    device,
                    eval_limit=args.eval_limit,
                    logger=logger,
                )
                
                eval_duration = (datetime.datetime.now() - eval_start).total_seconds()
                
                results[method_name]["evaluation"] = {
                    "test_psnr": eval_results["mean_psnr"],
                    "num_test_images": eval_results["num_test_images"],
                    "evaluation_duration_seconds": eval_duration,
                }
                
                logger.info(f"  {method_name} Evaluation PSNR: {eval_results['mean_psnr']:.2f} dB")
                logger.info(f"  Evaluation time: {eval_duration:.1f}s")
                
            except Exception as e:
                logger.error(f"{method_name} evaluation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results[method_name]["evaluation"] = {"error": str(e)}
    
    # Calculate total duration including evaluation
    for method_name in results:
        if "error" not in results[method_name]:
            training_time = results[method_name].get("training_duration_seconds", 0)
            eval_time = results[method_name].get("evaluation", {}).get("evaluation_duration_seconds", 0)
            results[method_name]["duration_seconds"] = training_time + eval_time
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    summary_table = []
    for method_name in methods.keys():
        if method_name in results and "error" not in results[method_name]:
            r = results[method_name]
            row = {
                "Method": method_name,
                "Max Iter": r["max_iter"],
                "F-Evals/Batch": r["total_f_evals_per_batch"],
                "Train Loss": f"{r['loss_train'][-1]:.2f}",
                "Val Loss": f"{r['loss_val'][-1]:.2f}",
                "Train PSNR": f"{r['psnr_train'][-1]:.2f}",
                "Val PSNR": f"{r['psnr_val'][-1]:.2f}",
                "Duration (s)": f"{r['duration_seconds']:.1f}",
            }
            if "evaluation" in r and "test_psnr" in r["evaluation"]:
                row["Test PSNR"] = f"{r['evaluation']['test_psnr']:.2f}"
            summary_table.append(row)
    
    # Print table
    if summary_table:
        headers = list(summary_table[0].keys())
        col_widths = {h: max(len(h), max(len(str(row[h])) for row in summary_table)) for h in headers}
        
        header_str = " | ".join(h.ljust(col_widths[h]) for h in headers)
        logger.info(header_str)
        logger.info("-" * len(header_str))
        
        for row in summary_table:
            row_str = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
            logger.info(row_str)
    
    # Save results to JSON
    results_file = os.path.join(log_dir, f"comparison_results_{args.problem}_{args.regularizer_name}_{run_id}.json")
    
    # Convert numpy/tensor values to python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    # Add metadata to results
    results_with_metadata = {
        "metadata": {
            "problem": args.problem,
            "regularizer": args.regularizer_name,
            "run_id": run_id,
            "epochs": args.epochs,
            "validation_epochs": args.validation_epochs,
            "train_set_size": len(train_set),
            "val_set_size": len(val_set),
            "eval_limit": args.eval_limit,
            "revdeq_beta": args.revdeq_beta,
            "timestamp": datetime.datetime.now().isoformat(),
        },
        "results": results,
    }
    
    with open(results_file, "w") as f:
        json.dump(convert_to_serializable(results_with_metadata), f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Log saved to: {log_file}")
    
    return results


if __name__ == "__main__":
    rev_deq_max_iter = 20
    parser = argparse.ArgumentParser(description="Compare JFB and RevDEQ hypergradient methods on CT")
    parser.add_argument("--problem", type=str, default="CT", choices=["CT"],
                        help="Problem type (CT only for this script)")
    parser.add_argument("--regularizer_name", type=str, default="CRR", 
                        choices=["CRR", "WCRR", "ICNN", "IDCNN", "TDV", "LSR"])
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--revdeq_max_iter", type=int, default=rev_deq_max_iter, 
                        help="Max iterations for RevDEQ (2 f-evals per step)")
    parser.add_argument("--jfb_max_iter", type=int, default=2*rev_deq_max_iter, 
                        help="Max iterations for JFB (1 f-eval per step)")
    parser.add_argument("--train_limit", type=int, default=50, 
                        help="Limit training set size (None for full dataset)")
    parser.add_argument("--val_limit", type=int, default=20, 
                        help="Limit validation set size (None for full dataset)")
    parser.add_argument("--eval_limit", type=int, default=10, 
                        help="Limit evaluation set size (None for full test set)")
    parser.add_argument("--load_pretrain", action="store_true", default=True,
                        help="Load pretrained score weights (default: True)")
    parser.add_argument("--no_load_pretrain", action="store_false", dest="load_pretrain",
                        help="Do NOT load pretrained weights; train regularizer from scratch")
    parser.add_argument("--load_param_fit", action="store_true", 
                        help="Load parameter-fitted weights instead of just pretrained")
    parser.add_argument("--pretrain_epochs", type=int, default=None,
                        help="Number of score pretraining epochs if training from scratch")
    parser.add_argument("--revdeq_beta", type=float, default=0.5,
                        help="Relaxation parameter for RevDEQ reversible iterations (default 0.5 for numerical stability)")
    parser.add_argument("--revdeq_step_size", type=float, default=5e-4,
                        help="Step size for RevDEQ fixed-point iterations (default 1e-4, smaller than JFB due to fixed step)")
    parser.add_argument("--use_embedded_beta", action="store_true",
                        help="Embed beta directly into fixed-point function (experimental)")
    parser.add_argument("--validation_epochs", type=int, default=None,
                        help="Run validation every N epochs (default: min(5, epochs))")
    parser.add_argument("--run_evaluation", action="store_true", default=True,
                        help="Run post-training evaluation (default: True)")
    parser.add_argument("--no_evaluation", action="store_false", dest="run_evaluation",
                        help="Skip post-training evaluation")
    
    args = parser.parse_args()
    
    # Set validation_epochs default: min(5, epochs) to ensure it's <= epochs
    if args.validation_epochs is None:
        args.validation_epochs = min(5, args.epochs)
    elif args.validation_epochs > args.epochs:
        print(f"Warning: validation_epochs ({args.validation_epochs}) > epochs ({args.epochs}), setting to {args.epochs}")
        args.validation_epochs = args.epochs
    
    results = run_comparison(args)
