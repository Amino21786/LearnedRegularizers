"""
Plot comparison results from hypergradient method comparison experiments.

This script takes a JSON results file from compare_hypergradient_methods.py
and creates visualizations comparing IFT, JFB, and RevDEQ methods.

Usage:
    python plot_comparison_results.py comparison_logs/comparison_results_Denoising_CRR_*.json
    python plot_comparison_results.py comparison_logs/comparison_results_*.json --output plots/
    python plot_comparison_results.py results.json --no-show --save
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Style configuration
COLORS = {
    "IFT": "#2ecc71",      # Green
    "JFB": "#3498db",      # Blue  
    "RevDEQ": "#e74c3c",   # Red
}

MARKERS = {
    "IFT": "o",
    "JFB": "s",
    "RevDEQ": "^",
}

LINESTYLES = {
    "IFT": "-",
    "JFB": "--",
    "RevDEQ": "-.",
}


def load_results(json_path: str) -> Tuple[Dict, Optional[Dict]]:
    """
    Load results from JSON file.
    
    Returns:
        Tuple of (results dict, metadata dict or None)
        
    Handles both old format (flat dict) and new format (with metadata and results keys).
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Check if it's the new format with metadata
    if "metadata" in data and "results" in data:
        return data["results"], data["metadata"]
    else:
        # Old format - flat dict
        return data, None


def parse_log_for_epoch_times(log_path: str) -> Dict[str, List[float]]:
    """
    Parse the log file to extract per-epoch timing information.
    
    Returns a dict mapping method name to list of cumulative times (seconds from method start).
    """
    epoch_times = {}
    current_method = None
    method_start_time = None
    
    # Pattern to match "Running METHOD" lines
    method_pattern = re.compile(r"Running (IFT|JFB|RevDEQ)")
    # Pattern to match epoch completion lines with timestamp
    epoch_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+: \[Epoch (\d+)\] Train Loss")
    
    try:
        with open(log_path, "r") as f:
            for line in f:
                # Check for method start
                method_match = method_pattern.search(line)
                if method_match:
                    current_method = method_match.group(1)
                    epoch_times[current_method] = []
                    # Extract timestamp from this line
                    timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                    if timestamp_match:
                        method_start_time = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                    continue
                
                # Check for epoch completion
                epoch_match = epoch_pattern.search(line)
                if epoch_match and current_method and method_start_time:
                    timestamp_str = epoch_match.group(1)
                    epoch_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    elapsed = (epoch_time - method_start_time).total_seconds()
                    epoch_times[current_method].append(elapsed)
    except FileNotFoundError:
        print(f"Warning: Log file not found at {log_path}")
    
    return epoch_times


def plot_psnr_vs_epochs(results: Dict, ax: plt.Axes, metric: str = "train") -> None:
    """Plot PSNR vs epochs for all methods."""
    for method in ["IFT", "JFB", "RevDEQ"]:
        if method not in results or "error" in results[method]:
            continue
        
        key = f"psnr_{metric}"
        psnr = results[method][key]
        epochs = np.arange(1, len(psnr) + 1)
        
        ax.plot(
            epochs, psnr,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2,
            markersize=6,
            label=method,
        )
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(f"{'Training' if metric == 'train' else 'Validation'} PSNR (dB)", fontsize=12)
    ax.set_title(f"{'Training' if metric == 'train' else 'Validation'} PSNR vs Epochs", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def get_duration(results: Dict, method: str) -> float:
    """Get duration from results, preferring training_duration_seconds if available."""
    r = results[method]
    if "training_duration_seconds" in r:
        return r["training_duration_seconds"]
    return r.get("duration_seconds", 0)


def plot_psnr_vs_time(
    results: Dict,
    ax: plt.Axes,
    epoch_times: Optional[Dict[str, List[float]]] = None,
    metric: str = "train"
) -> None:
    """Plot PSNR vs wall-clock time for all methods."""
    for method in ["IFT", "JFB", "RevDEQ"]:
        if method not in results or "error" in results[method]:
            continue
        
        key = f"psnr_{metric}"
        psnr = results[method][key]
        total_time = get_duration(results, method)
        
        # Use parsed epoch times if available, otherwise estimate
        if epoch_times and method in epoch_times and len(epoch_times[method]) == len(psnr):
            times = np.array(epoch_times[method])
        else:
            # Adjust for validation which may have fewer points
            times = np.linspace(total_time / len(psnr), total_time, len(psnr))
        
        ax.plot(
            times, psnr,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2,
            markersize=6,
            label=f"{method} ({total_time:.0f}s total)",
        )
    
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel(f"{'Training' if metric == 'train' else 'Validation'} PSNR (dB)", fontsize=12)
    ax.set_title(f"{'Training' if metric == 'train' else 'Validation'} PSNR vs Time", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)


def plot_loss_vs_epochs(results: Dict, ax: plt.Axes, metric: str = "train") -> None:
    """Plot loss vs epochs for all methods."""
    for method in ["IFT", "JFB", "RevDEQ"]:
        if method not in results or "error" in results[method]:
            continue
        
        key = f"loss_{metric}"
        loss = results[method][key]
        epochs = np.arange(1, len(loss) + 1)
        
        ax.plot(
            epochs, loss,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2,
            markersize=6,
            label=method,
        )
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(f"{'Training' if metric == 'train' else 'Validation'} Loss", fontsize=12)
    ax.set_title(f"{'Training' if metric == 'train' else 'Validation'} Loss vs Epochs", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_summary_bar(results: Dict, ax: plt.Axes, include_evaluation: bool = True) -> None:
    """Plot summary bar chart comparing final metrics and time."""
    methods = [m for m in ["IFT", "JFB", "RevDEQ"] if m in results and "error" not in results[m]]
    
    x = np.arange(len(methods))
    
    # Check if evaluation data exists
    has_evaluation = include_evaluation and any(
        "evaluation" in results.get(m, {}) and "test_psnr" in results.get(m, {}).get("evaluation", {})
        for m in methods
    )
    
    if has_evaluation:
        width = 0.35
        # Get final train and test PSNR values
        final_train_psnr = [results[m]["psnr_train"][-1] for m in methods]
        final_test_psnr = [
            results[m].get("evaluation", {}).get("test_psnr", 0) for m in methods
        ]
        
        bars1 = ax.bar(x - width/2, final_train_psnr, width, label="Val PSNR (dB)", 
                       color=[COLORS[m] for m in methods], alpha=0.6)
        bars2 = ax.bar(x + width/2, final_test_psnr, width, label="Test PSNR (dB)", 
                       color=[COLORS[m] for m in methods], alpha=1.0, hatch="//")
        
        # Add value labels on bars
        for bar, val in zip(bars1, final_train_psnr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars2, final_test_psnr):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        
        ax.set_title("Final PSNR Comparison (Val vs Test)", fontsize=14)
    else:
        width = 0.5
        # Get final PSNR values
        final_psnr = [results[m]["psnr_train"][-1] for m in methods]
        
        bars1 = ax.bar(x, final_psnr, width, label="Final Train PSNR (dB)", 
                       color=[COLORS[m] for m in methods], alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars1, final_psnr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        
        ax.set_title("Final Training PSNR Comparison", fontsize=14)
    
    # Get durations (prefer training_duration_seconds if available, fallback to duration_seconds)
    durations = []
    for m in methods:
        r = results[m]
        if "training_duration_seconds" in r:
            durations.append(r["training_duration_seconds"])
        else:
            durations.append(r.get("duration_seconds", 0))
    
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add duration text below each bar
    ylim = ax.get_ylim()
    for i, (method, duration) in enumerate(zip(methods, durations)):
        ax.text(i, ylim[0] - (ylim[1] - ylim[0]) * 0.05, f"{duration:.0f}s",
                ha="center", va="top", fontsize=9, color="gray")


def plot_efficiency(results: Dict, ax: plt.Axes) -> None:
    """Plot PSNR improvement per second (efficiency)."""
    methods = [m for m in ["IFT", "JFB", "RevDEQ"] if m in results and "error" not in results[m]]
    
    # Calculate PSNR improvement rate (dB per second)
    efficiencies = []
    for method in methods:
        psnr_values = results[method]["psnr_train"]
        # Prefer training_duration_seconds if available
        if "training_duration_seconds" in results[method]:
            duration = results[method]["training_duration_seconds"]
        else:
            duration = results[method].get("duration_seconds", 1)
        psnr_improvement = psnr_values[-1] - psnr_values[0]
        efficiency = psnr_improvement / max(duration, 1) * 60  # dB per minute
        efficiencies.append(efficiency)
    
    x = np.arange(len(methods))
    bars = ax.bar(x, efficiencies, color=[COLORS[m] for m in methods], alpha=0.8)
    
    for bar, val in zip(bars, efficiencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    
    ax.set_ylabel("PSNR Improvement (dB/min)", fontsize=12)
    ax.set_title("Training Efficiency", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")


def plot_evaluation_comparison(results: Dict, ax: plt.Axes) -> None:
    """Plot evaluation PSNR comparison bar chart."""
    methods = [m for m in ["IFT", "JFB", "RevDEQ"] 
               if m in results and "error" not in results[m] 
               and "evaluation" in results[m] and "test_psnr" in results[m].get("evaluation", {})]
    
    if not methods:
        ax.text(0.5, 0.5, "No evaluation data available", 
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Evaluation PSNR", fontsize=14)
        return
    
    x = np.arange(len(methods))
    test_psnrs = [results[m]["evaluation"]["test_psnr"] for m in methods]
    
    bars = ax.bar(x, test_psnrs, color=[COLORS[m] for m in methods], alpha=0.8)
    
    for bar, val in zip(bars, test_psnrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    
    ax.set_ylabel("Test PSNR (dB)", fontsize=12)
    ax.set_title("Evaluation PSNR (Test Set)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")


def create_full_comparison_plot(
    results: Dict,
    epoch_times: Optional[Dict[str, List[float]]] = None,
    title: str = "Hypergradient Method Comparison",
    metadata: Optional[Dict] = None,
) -> plt.Figure:
    """Create a comprehensive comparison figure with multiple subplots."""
    # Check if we have evaluation data
    has_evaluation = any(
        "evaluation" in results.get(m, {}) and "test_psnr" in results.get(m, {}).get("evaluation", {})
        for m in ["IFT", "JFB", "RevDEQ"] if m in results
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    # Top row: PSNR plots
    plot_psnr_vs_epochs(results, axes[0, 0], metric="train")
    plot_psnr_vs_time(results, axes[0, 1], epoch_times, metric="train")
    plot_summary_bar(results, axes[0, 2], include_evaluation=has_evaluation)
    
    # Bottom row: Loss, validation PSNR, and efficiency/evaluation
    plot_loss_vs_epochs(results, axes[1, 0], metric="train")
    plot_psnr_vs_epochs(results, axes[1, 1], metric="val")
    
    if has_evaluation:
        plot_evaluation_comparison(results, axes[1, 2])
    else:
        plot_efficiency(results, axes[1, 2])
    
    # Add metadata annotation if available
    if metadata:
        meta_text = []
        if "problem" in metadata:
            meta_text.append(f"Problem: {metadata['problem']}")
        if "regularizer" in metadata:
            meta_text.append(f"Regularizer: {metadata['regularizer']}")
        if "epochs" in metadata:
            meta_text.append(f"Epochs: {metadata['epochs']}")
        if "train_set_size" in metadata:
            meta_text.append(f"Train: {metadata['train_set_size']}")
        if meta_text:
            fig.text(0.02, 0.02, " | ".join(meta_text), fontsize=9, 
                    color="gray", ha="left", va="bottom")
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot comparison results from hypergradient method experiments"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to JSON results file (e.g., comparison_results_*.json)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for saved plots (default: same as input)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save plots to files"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful for batch processing)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Output format for saved plots (default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved plots (default: 150)"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.json_file}")
    results, metadata = load_results(args.json_file)
    
    # Try to find and parse corresponding log file for epoch times
    json_path = Path(args.json_file)
    log_path = json_path.parent / json_path.name.replace("comparison_results_", "comparison_").replace(".json", ".log")
    
    epoch_times = None
    if log_path.exists():
        print(f"Parsing epoch times from: {log_path}")
        epoch_times = parse_log_for_epoch_times(str(log_path))
    
    # Extract title from metadata or filename
    if metadata and "problem" in metadata and "regularizer" in metadata:
        problem = metadata["problem"]
        regularizer = metadata["regularizer"]
        run_id = metadata.get("run_id", "")
        run_info = f" ({run_id})" if run_id else ""
        title = f"Hypergradient Comparison: {problem} with {regularizer}{run_info}"
    else:
        # Fallback to filename parsing
        # Supports both old format (comparison_results_Problem_Reg_YYYY-MM-DD_HH-MM-SS.json)
        # and new format (comparison_results_Problem_Reg_YYYYMMDD_runN.json)
        filename = json_path.stem
        parts = filename.replace("comparison_results_", "").split("_")
        if len(parts) >= 2:
            problem = parts[0]
            regularizer = parts[1]
            # Try to extract run ID for new format
            run_info = ""
            if len(parts) >= 4 and parts[-1].startswith("run"):
                run_info = f" ({parts[-2]}_{parts[-1]})"
            title = f"Hypergradient Comparison: {problem} with {regularizer}{run_info}"
        else:
            title = "Hypergradient Method Comparison"
    
    # Create plot
    fig = create_full_comparison_plot(results, epoch_times, title, metadata)
    
    # Save if requested
    if args.save:
        output_dir = args.output if args.output else str(json_path.parent)
        os.makedirs(output_dir, exist_ok=True)
        
        output_name = json_path.stem.replace("comparison_results_", "plot_") + f".{args.format}"
        output_path = os.path.join(output_dir, output_name)
        
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    
    # Show plot
    if not args.no_show:
        plt.show()
    
    # Check if evaluation data exists
    has_evaluation = any(
        "evaluation" in results.get(m, {}) and "test_psnr" in results.get(m, {}).get("evaluation", {})
        for m in ["IFT", "JFB", "RevDEQ"] if m in results
    )
    
    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    if has_evaluation:
        print(f"{'Method':<10} {'Train PSNR':<12} {'Val PSNR':<12} {'Test PSNR':<12} {'Train Time (s)':<16} {'F-evals/batch':<14}")
        print("-" * 90)
        for method in ["IFT", "JFB", "RevDEQ"]:
            if method in results and "error" not in results[method]:
                r = results[method]
                train_psnr = r["psnr_train"][-1]
                val_psnr = r["psnr_val"][-1] if r["psnr_val"] else "N/A"
                test_psnr = r.get("evaluation", {}).get("test_psnr", "N/A")
                # Prefer training_duration_seconds if available
                duration = r.get("training_duration_seconds", r.get("duration_seconds", 0))
                f_evals = r.get("total_f_evals_per_batch", "N/A")
                
                val_str = f"{val_psnr:.2f}" if isinstance(val_psnr, (int, float)) else val_psnr
                test_str = f"{test_psnr:.2f}" if isinstance(test_psnr, (int, float)) else test_psnr
                f_evals_str = str(f_evals)
                print(f"{method:<10} {train_psnr:<12.2f} {val_str:<12} {test_str:<12} {duration:<16.1f} {f_evals_str:<14}")
    else:
        print(f"{'Method':<10} {'Final Train PSNR':<18} {'Final Val PSNR':<16} {'Duration (s)':<14} {'F-evals/batch':<14}")
        print("-" * 90)
        for method in ["IFT", "JFB", "RevDEQ"]:
            if method in results and "error" not in results[method]:
                r = results[method]
                train_psnr = r["psnr_train"][-1]
                val_psnr = r["psnr_val"][-1] if r["psnr_val"] else "N/A"
                duration = r.get("training_duration_seconds", r.get("duration_seconds", 0))
                f_evals = r.get("total_f_evals_per_batch", "N/A")
                
                val_str = f"{val_psnr:.2f}" if isinstance(val_psnr, (int, float)) else val_psnr
                f_evals_str = str(f_evals)
                print(f"{method:<10} {train_psnr:<18.2f} {val_str:<16} {duration:<14.1f} {f_evals_str:<14}")
    print("=" * 90)
    
    # Print metadata if available
    if metadata:
        print("\nRun Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
