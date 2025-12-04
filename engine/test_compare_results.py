#!/usr/bin/env python3
"""
Compare training results across different experiment folders.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Increase all font sizes by 30% for clearer plots
FONT_SCALE = 1.65
DEFAULT_FONT_SIZE = plt.rcParams.get('font.size', 14)
plt.rcParams.update({'font.size': DEFAULT_FONT_SIZE * FONT_SCALE})
sns.set_context("notebook", font_scale=FONT_SCALE)

# Predefined plots the script should generate.
TARGET_PLOTS = [
    {
        "title": "Success Rate @ 1% of Image Width (6.4 pixels)",
        "metric": "2%_win_acc",
        "experiments": ["heatmap_sigma_4", "coords_amplifier_100"],
        "output": "2%_win_acc_comparison_heatmap_sigma_4_vs_coords_amplifier_100.png",
    },
    {
        "title": "Mean Euclidean Error (MEE)",
        "metric": "dist",
        "experiments": ["heatmap_sigma_4", "coords_amplifier_100"],
        "output": "dist_comparison_heatmap_sigma_4_vs_coords_amplifier_100.png",
    },
    {
        "title": "Success Rate @ 1% of Image Width (6.4 pixels)",
        "metric": "2%_win_acc",
        "experiments": ["heatmap_sigma_1", "heatmap_sigma_4", "heatmap_sigma_16"],
        "output": "2%_win_acc_comparison_heatmap_sigma_1_4_16.png",
    },
    {
        "title": "Mean Euclidean Error (MEE)",
        "metric": "dist",
        "experiments": ["heatmap_sigma_1", "heatmap_sigma_4", "heatmap_sigma_16"],
        "output": "dist_comparison_heatmap_sigma_1_4_16.png",
    },
]

METRIC_Y_LIMITS = {
    "2%_win_acc": (0.0, 1.0),
    "dist": (0.0, 0.9),
}
Y_LIMIT_PADDING = 0.05


def moving_average(data: np.ndarray, window: int = 200) -> np.ndarray:
    """Apply moving average to smooth the data."""
    if len(data) < window:
        return data
    
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    # Pad the beginning with original values
    padded = np.full(len(data), np.nan)
    padded[window-1:] = smoothed
    padded[:window-1] = data[:window-1]
    return padded


def load_training_data(folder_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training step data and validation data from a folder."""
    step_file = folder_path / "step.csv"
    val_file = folder_path / "val_loss.csv"
    
    if not step_file.exists():
        raise FileNotFoundError(f"step.csv not found in {folder_path}")
    if not val_file.exists():
        raise FileNotFoundError(f"val_loss.csv not found in {folder_path}")
    
    step_df = pd.read_csv(step_file)
    val_df = pd.read_csv(val_file)
    
    return step_df, val_df


def calculate_steps_per_epoch(step_df: pd.DataFrame) -> int:
    """Calculate number of steps per epoch from training data."""
    # Find the first step of epoch 2
    epoch_2_start = step_df[step_df['epoch'] == 2]['step'].iloc[0] if len(step_df[step_df['epoch'] == 2]) > 0 else None
    epoch_1_start = step_df[step_df['epoch'] == 1]['step'].iloc[0] if len(step_df[step_df['epoch'] == 1]) > 0 else None
    
    if epoch_2_start is not None and epoch_1_start is not None:
        return epoch_2_start - epoch_1_start
    else:
        # Fallback: estimate from total steps and epochs
        max_epoch = step_df['epoch'].max()
        max_step = step_df['step'].max()
        return max_step // max_epoch


def build_color_map(experiments: List[str]) -> Dict[str, Tuple[float, float, float]]:
    """Assign a consistent color to each experiment."""
    palette = sns.color_palette("husl", len(experiments))
    return {exp: palette[idx] for idx, exp in enumerate(experiments)}


def format_metric_label(metric: str) -> str:
    """Create a human-readable label for a metric name."""
    label = metric.replace('_', ' ').title()
    return label.replace('% ', '%')


def plot_metric_subset(results_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                       metric: str,
                       experiments: List[str],
                       colors: Dict[str, Tuple[float, float, float]],
                       output_file: Path,
                       title: str):
    """Plot a specific metric for a subset of experiments."""
    plt.figure(figsize=(12, 8))

    for exp_name in experiments:
        if exp_name not in results_data:
            print(f"Skipping {exp_name}: data not available.")
            continue

        step_df, val_df = results_data[exp_name]
        color = colors[exp_name]

        if metric not in step_df.columns or metric not in val_df.columns:
            print(f"Skipping {exp_name}: metric '{metric}' not found in data.")
            continue

        steps_per_epoch = calculate_steps_per_epoch(step_df)

        train_metric_smoothed = moving_average(step_df[metric].values)
        plt.plot(step_df['step'], train_metric_smoothed,
                 label=f'{exp_name} (train)', color=color, linewidth=2)

        val_steps = val_df['epoch'] * steps_per_epoch
        plt.plot(val_steps, val_df[metric],
                 label=f'{exp_name} (val)', color=color, linestyle='--', linewidth=2)

    plt.xlabel('Step')
    plt.ylabel(format_metric_label(metric))
    if metric in METRIC_Y_LIMITS:
        y_min, y_max = METRIC_Y_LIMITS[metric]
        plt.ylim(y_min - Y_LIMIT_PADDING, y_max + Y_LIMIT_PADDING)
    plt.title(title)
    plt.legend(bbox_to_anchor=(0.57, 0.53), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{title} saved to: {output_file}")


def main():
    """Generate the predefined comparison plots."""
    parser = argparse.ArgumentParser(description='Generate targeted comparison plots')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default='results/summary',
                       help='Path to output directory for plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_experiments = sorted({exp for plot in TARGET_PLOTS for exp in plot['experiments']})
    print("Required experiments:")
    for exp in required_experiments:
        print(f"  - {exp}")

    results_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for exp_name in required_experiments:
        folder = results_dir / exp_name
        if not folder.is_dir():
            print(f"Warning: experiment folder not found: {folder}")
            continue

        try:
            step_df, val_df = load_training_data(folder)
            results_data[exp_name] = (step_df, val_df)
            print(f"Loaded data from {exp_name}")
        except Exception as exc:
            print(f"Error loading data from {folder}: {exc}")

    if not results_data:
        print("No valid data found for the requested plots.")
        return

    colors = build_color_map(sorted(results_data.keys()))

    print("\nGenerating targeted plots...")
    for plot_cfg in TARGET_PLOTS:
        output_file = output_dir / plot_cfg['output']
        plot_metric_subset(
            results_data=results_data,
            metric=plot_cfg['metric'],
            experiments=plot_cfg['experiments'],
            colors=colors,
            output_file=output_file,
            title=plot_cfg['title'],
        )

    print(f"\nAll requested plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
