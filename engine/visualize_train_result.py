#!/usr/bin/env python3
"""
Visualization script for training results from guidewire tip detector.
This script reads step.csv and val_loss.csv files and creates comprehensive visualizations.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(results_dir):
    """Load training data from CSV files."""
    step_csv_path = os.path.join(results_dir, 'step.csv')
    val_csv_path = os.path.join(results_dir, 'val_loss.csv')
    
    if not os.path.exists(step_csv_path):
        raise FileNotFoundError(f"step.csv not found in {results_dir}")
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"val_loss.csv not found in {results_dir}")
    
    # Load data
    step_df = pd.read_csv(step_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    print(f"Loaded step data: {len(step_df)} rows")
    print(f"Loaded validation data: {len(val_df)} rows")
    
    return step_df, val_df

def plot_loss_curves(step_df, val_df, save_dir):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Overview', fontsize=16, fontweight='bold')
    
    # 1. Total Loss (Training vs Validation) with dual y-axes for loss and accuracy
    ax1 = axes[0, 0]
    
    # Calculate steps per epoch from the data
    steps_per_epoch = len(step_df) // val_df['epoch'].iloc[-1]
    
    # Convert validation epochs to steps for proper x-axis alignment
    val_steps = (val_df['epoch']) * steps_per_epoch
    
    # Plot training and validation loss on left y-axis
    ax1.plot(step_df['step'], step_df['loss_total'], alpha=0.7, label='Training Loss', linewidth=1, color='blue')
    ax1.plot(val_steps, val_df['val_loss_total'], marker='o', markersize=4, 
             label='Validation Loss', linewidth=2, color='red')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Total Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Total Loss: Training vs Validation')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for accuracy
    ax1_twin = ax1.twinx()
    
    # Plot training accuracy (5% and 1% window accuracy)
    if '5%_win_acc' in step_df.columns:
        # Apply smoothing to training accuracy for better visualization
        window_size = max(1, len(step_df) // 100)
        train_acc_5_smooth = step_df['5%_win_acc'].rolling(window=window_size, center=True).mean()
        ax1_twin.plot(step_df['step'], train_acc_5_smooth, alpha=0.7, 
                     label='Training 5% Acc', linewidth=1, color='green', linestyle='--')
    
    if '1%_win_acc' in step_df.columns:
        train_acc_1_smooth = step_df['1%_win_acc'].rolling(window=window_size, center=True).mean()
        ax1_twin.plot(step_df['step'], train_acc_1_smooth, alpha=0.7, 
                     label='Training 1% Acc', linewidth=1, color='orange', linestyle='--')
    
    # Plot validation accuracy
    if '5%_win_acc' in val_df.columns:
        ax1_twin.plot(val_steps, val_df['5%_win_acc'], marker='s', markersize=3, 
                     label='Val 5% Acc', linewidth=2, color='darkgreen')
    
    if '1%_win_acc' in val_df.columns:
        ax1_twin.plot(val_steps, val_df['1%_win_acc'], marker='^', markersize=3, 
                     label='Val 1% Acc', linewidth=2, color='darkorange')
    
    ax1_twin.set_ylabel('Accuracy', color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1_twin.set_ylim(0, 1)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    # 2. Learning Rate Schedule
    ax2 = axes[0, 1]
    ax2.plot(step_df['step'], step_df['lr'], color='green', linewidth=1)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Individual Loss Components (Training)
    ax3 = axes[1, 0]
    loss_columns = [col for col in step_df.columns if col not in ['epoch', 'step', 'lr', 'loss_total']]
    for col in loss_columns:
        if col in step_df.columns:
            ax3.plot(step_df['step'], step_df[col], label=col, alpha=0.7, linewidth=1)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Loss Value')
    ax3.set_title('Individual Loss Components (Training)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy Metrics (Validation)
    ax4 = axes[1, 1]
    acc_columns = [col for col in val_df.columns if 'acc' in col]
    for col in acc_columns:
        if col in val_df.columns:
            ax4.plot(val_df['epoch'], val_df[col], marker='o', markersize=4, label=col, linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy Metrics (Validation)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_detailed_losses(step_df, val_df, save_dir):
    """Plot detailed loss analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Detailed Loss Analysis', fontsize=16, fontweight='bold')
    
    # 1. Loss Components Comparison (Training vs Validation on same scale)
    ax1 = axes[0, 0]
    
    # Calculate steps per epoch from the data
    steps_per_epoch = len(step_df) // val_df['epoch'].iloc[-1]
    val_steps = (val_df['epoch']) * steps_per_epoch
    
    loss_components = ['bce', 'mse']
    for component in loss_components:
        if component in val_df.columns:
            ax1.plot(val_steps, val_df[component], marker='o', markersize=4, 
                    label=f'Val {component.upper()}', linewidth=2)
        if component in step_df.columns:
            # Apply smoothing to training data
            window_size = max(1, len(step_df) // 100)
            train_smooth = step_df[component].rolling(window=window_size, center=True).mean()
            ax1.plot(step_df['step'], train_smooth, alpha=0.7, 
                    label=f'Train {component.upper()}', linewidth=1, linestyle='--')
    
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('Loss Components: Training vs Validation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Progression (Training vs Validation on same scale)
    ax2 = axes[0, 1]
    acc_metrics = ['5%_win_acc', '1%_win_acc', '0.5%_win_acc']
    for metric in acc_metrics:
        if metric in val_df.columns:
            ax2.plot(val_steps, val_df[metric], marker='o', markersize=4, 
                    label=f'Val {metric}', linewidth=2)
        if metric in step_df.columns:
            # Apply smoothing to training data
            window_size = max(1, len(step_df) // 100)
            train_smooth = step_df[metric].rolling(window=window_size, center=True).mean()
            ax2.plot(step_df['step'], train_smooth, alpha=0.7, 
                    label=f'Train {metric}', linewidth=1, linestyle='--')
    
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Metrics: Training vs Validation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Loss Smoothing
    ax3 = axes[1, 0]
    # Apply moving average for smoother visualization
    window_size = max(1, len(step_df) // 100)  # Adaptive window size
    step_df_smooth = step_df.copy()
    step_df_smooth['loss_total_smooth'] = step_df_smooth['loss_total'].rolling(window=window_size, center=True).mean()
    
    ax3.plot(step_df['step'], step_df['loss_total'], alpha=0.3, label='Raw Loss', linewidth=0.5)
    ax3.plot(step_df_smooth['step'], step_df_smooth['loss_total_smooth'], 
             label=f'Smoothed (window={window_size})', linewidth=2, color='red')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Total Loss')
    ax3.set_title('Training Loss (Smoothed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss vs Learning Rate
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(step_df['step'], step_df['loss_total'], color='blue', alpha=0.7, label='Loss')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Total Loss', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')
    
    line2 = ax4_twin.plot(step_df['step'], step_df['lr'], color='red', alpha=0.7, label='Learning Rate')
    ax4_twin.set_ylabel('Learning Rate', color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4_twin.set_yscale('log')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    ax4.set_title('Loss vs Learning Rate')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detailed_loss_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy_analysis(step_df, val_df, save_dir):
    """Plot detailed accuracy analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Accuracy Analysis', fontsize=16, fontweight='bold')
    
    # 1. All Accuracy Metrics (Training vs Validation on same scale)
    ax1 = axes[0, 0]
    
    # Calculate steps per epoch from the data
    steps_per_epoch = len(step_df) // val_df['epoch'].iloc[-1]
    val_steps = (val_df['epoch']) * steps_per_epoch
    
    acc_columns = [col for col in val_df.columns if 'acc' in col]
    colors = plt.cm.Set3(np.linspace(0, 1, len(acc_columns)))
    
    for i, col in enumerate(acc_columns):
        if col in val_df.columns:
            ax1.plot(val_steps, val_df[col], marker='o', markersize=4, 
                    label=f'Val {col}', color=colors[i], linewidth=2)
        if col in step_df.columns:
            # Apply smoothing to training data
            window_size = max(1, len(step_df) // 100)
            train_smooth = step_df[col].rolling(window=window_size, center=True).mean()
            ax1.plot(step_df['step'], train_smooth, alpha=0.7, 
                    label=f'Train {col}', color=colors[i], linewidth=1, linestyle='--')
    
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('All Accuracy Metrics: Training vs Validation')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Improvement Rate
    ax2 = axes[0, 1]
    for col in acc_columns:
        if col in val_df.columns:
            # Calculate improvement rate
            improvement = val_df[col].diff().fillna(0)
            ax2.plot(val_df['epoch'][1:], improvement[1:], marker='o', markersize=3, 
                    label=f'{col} improvement', linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy Improvement per Epoch')
    ax2.set_title('Accuracy Improvement Rate')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Training vs Validation Accuracy (on same scale)
    ax3 = axes[1, 0]
    train_acc_columns = [col for col in step_df.columns if 'acc' in col]
    if train_acc_columns:
        for col in train_acc_columns:
            if col in step_df.columns:
                # Apply smoothing for training data
                window_size = max(1, len(step_df) // 50)
                smoothed = step_df[col].rolling(window=window_size, center=True).mean()
                ax3.plot(step_df['step'], smoothed, label=f'Training {col}', alpha=0.8, linewidth=1, linestyle='--')
        
        # Add validation accuracy for comparison (convert to steps)
        for col in acc_columns:
            if col in val_df.columns:
                ax3.plot(val_steps, val_df[col], marker='o', markersize=4, 
                        label=f'Val {col}', linewidth=2)
        
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Training vs Validation Accuracy (Same Scale)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax3.text(0.5, 0.5, 'No training accuracy data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Training Accuracy (Not Available)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Final Performance Summary
    ax4 = axes[1, 1]
    final_epoch = val_df['epoch'].iloc[-1]
    final_accuracies = {}
    for col in acc_columns:
        if col in val_df.columns:
            final_accuracies[col] = val_df[col].iloc[-1]
    
    if final_accuracies:
        bars = ax4.bar(range(len(final_accuracies)), list(final_accuracies.values()), 
                      color=colors[:len(final_accuracies)])
        ax4.set_xticks(range(len(final_accuracies)))
        ax4.set_xticklabels(list(final_accuracies.keys()), rotation=45, ha='right')
        ax4.set_ylabel('Final Accuracy')
        ax4.set_title(f'Final Performance (Epoch {final_epoch})')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, final_accuracies.values())):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No accuracy data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Final Performance (No Data)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_statistics(step_df, val_df, save_dir):
    """Plot training statistics and summary."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Statistics', fontsize=16, fontweight='bold')
    
    # 1. Loss Distribution
    ax1 = axes[0, 0]
    ax1.hist(step_df['loss_total'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(step_df['loss_total'].mean(), color='red', linestyle='--', 
               label=f'Mean: {step_df["loss_total"].mean():.4f}')
    ax1.axvline(step_df['loss_total'].median(), color='green', linestyle='--', 
               label=f'Median: {step_df["loss_total"].median():.4f}')
    ax1.set_xlabel('Total Loss')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Training Loss Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Learning Rate vs Loss
    ax2 = axes[0, 1]
    scatter = ax2.scatter(step_df['lr'], step_df['loss_total'], 
                        c=step_df['step'], cmap='viridis', alpha=0.6, s=1)
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Total Loss')
    ax2.set_xscale('log')
    ax2.set_title('Learning Rate vs Loss')
    plt.colorbar(scatter, ax=ax2, label='Step')
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Progress Summary
    ax3 = axes[1, 0]
    epochs = val_df['epoch'].values
    val_losses = val_df['val_loss_total'].values
    
    # Create a summary table
    summary_data = {
        'Metric': ['Initial Loss', 'Final Loss', 'Best Loss', 'Improvement'],
        'Value': [
            f"{val_losses[0]:.6f}",
            f"{val_losses[-1]:.6f}",
            f"{val_losses.min():.6f}",
            f"{((val_losses[0] - val_losses[-1]) / val_losses[0] * 100):.1f}%"
        ]
    }
    
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(cellText=[[summary_data['Metric'][i], summary_data['Value'][i]] 
                               for i in range(len(summary_data['Metric']))],
                     colLabels=['Metric', 'Value'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax3.set_title('Training Summary')
    
    # 4. Convergence Analysis
    ax4 = axes[1, 1]
    # Calculate moving average of validation loss
    window_size = max(1, len(val_df) // 5)
    val_df_smooth = val_df.copy()
    val_df_smooth['val_loss_smooth'] = val_df_smooth['val_loss_total'].rolling(window=window_size, center=True).mean()
    
    ax4.plot(val_df['epoch'], val_df['val_loss_total'], alpha=0.5, label='Raw Validation Loss', linewidth=1)
    ax4.plot(val_df_smooth['epoch'], val_df_smooth['val_loss_smooth'], 
             label=f'Smoothed (window={window_size})', linewidth=2, color='red')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss')
    ax4.set_title('Convergence Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_statistics.png'), dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(step_df, val_df):
    """Print training summary statistics."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Basic info
    print(f"Total training steps: {len(step_df)}")
    print(f"Total epochs: {val_df['epoch'].iloc[-1]}")
    print(f"Final learning rate: {step_df['lr'].iloc[-1]:.2e}")
    
    # Loss statistics
    print(f"\nLoss Statistics:")
    print(f"  Initial validation loss: {val_df['val_loss_total'].iloc[0]:.6f}")
    print(f"  Final validation loss: {val_df['val_loss_total'].iloc[-1]:.6f}")
    print(f"  Best validation loss: {val_df['val_loss_total'].min():.6f}")
    print(f"  Loss improvement: {((val_df['val_loss_total'].iloc[0] - val_df['val_loss_total'].iloc[-1]) / val_df['val_loss_total'].iloc[0] * 100):.1f}%")
    
    # Accuracy statistics
    acc_columns = [col for col in val_df.columns if 'acc' in col]
    if acc_columns:
        print(f"\nFinal Accuracy Metrics:")
        for col in acc_columns:
            if col in val_df.columns:
                initial_acc = val_df[col].iloc[0]
                final_acc = val_df[col].iloc[-1]
                print(f"  {col}: {initial_acc:.3f} → {final_acc:.3f} (+{final_acc-initial_acc:.3f})")
    
    # Training efficiency
    print(f"\nTraining Efficiency:")
    print(f"  Average loss per step: {step_df['loss_total'].mean():.6f}")
    print(f"  Loss standard deviation: {step_df['loss_total'].std():.6f}")
    print(f"  Steps per epoch: {len(step_df) // val_df['epoch'].iloc[-1]}")
    
    print("="*60)

def main():
    parser = ArgumentParser(description='Visualize training results from guidewire tip detector')
    parser.add_argument('--config', type=str, default='default',
                       help='Config name (e.g., "default") - results will be loaded from project_root/results/config_name')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots (default: same as results_dir)')
    parser.add_argument('--all', action='store_true',
                       help='Generate all visualizations')
    parser.add_argument('--overview', action='store_true',
                       help='Generate overview plots')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed loss analysis')
    parser.add_argument('--accuracy', action='store_true',
                       help='Generate accuracy analysis')
    parser.add_argument('--stats', action='store_true',
                       help='Generate training statistics')
    
    args = parser.parse_args()
    
    # Construct results directory path based on config name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'results', args.config)
    
    # Set default save directory
    if args.save_dir is None:
        args.save_dir = results_dir
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Load data
        print(f"Loading training data from: {results_dir}")
        step_df, val_df = load_data(results_dir)
        
        # Print summary
        print_summary(step_df, val_df)
        
        # Generate plots based on arguments
        if args.all or args.overview:
            print("\nGenerating overview plots...")
            plot_loss_curves(step_df, val_df, args.save_dir)
        
        if args.all or args.detailed:
            print("Generating detailed loss analysis...")
            plot_detailed_losses(step_df, val_df, args.save_dir)
        
        if args.all or args.accuracy:
            print("Generating accuracy analysis...")
            plot_accuracy_analysis(step_df, val_df, args.save_dir)
        
        if args.all or args.stats:
            print("Generating training statistics...")
            plot_training_statistics(step_df, val_df, args.save_dir)
        
        # If no specific plots requested, generate all
        if not any([args.all, args.overview, args.detailed, args.accuracy, args.stats]):
            print("\nGenerating all visualizations...")
            plot_loss_curves(step_df, val_df, args.save_dir)
            plot_detailed_losses(step_df, val_df, args.save_dir)
            plot_accuracy_analysis(step_df, val_df, args.save_dir)
            plot_training_statistics(step_df, val_df, args.save_dir)
        
        print(f"\nVisualization complete! Plots saved to: {args.save_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
