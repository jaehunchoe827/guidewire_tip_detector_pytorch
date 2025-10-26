import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os

def focal_loss(pred, target, alpha=1.0, gamma=2.0):
    """
    Focal Loss implementation
    Args:
        pred: predicted probabilities (0-1)
        target: ground truth labels (0-1)
        alpha: weighting factor for rare class
        gamma: focusing parameter
    """
    # Convert to tensors if needed
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)
    
    # Calculate BCE loss
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    
    # Calculate p_t
    p_t = pred * target + (1 - pred) * (1 - target)
    
    # Calculate focal weight
    focal_weight = alpha * (1 - p_t) ** gamma
    
    # Apply focal weight
    focal_loss = focal_weight * bce_loss
    
    return focal_loss

def bce_loss(pred, target):
    """
    Binary Cross Entropy Loss
    Args:
        pred: predicted probabilities (0-1)
        target: ground truth labels (0-1)
    """
    # Convert to tensors if needed
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)
    
    return F.binary_cross_entropy(pred, target, reduction='none')

def plot_focal_vs_bce():
    """
    Plot comparison between focal loss and BCE loss for different target values
    """
    # Create prediction range from 0 to 1
    y_pred = np.linspace(0.001, 0.999, 1000)  # Avoid log(0) by using small values
    
    # Target values to compare
    target_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot for each target value
    for target in target_values:
        # Calculate focal loss
        focal_losses = []
        bce_losses = []
        
        for pred in y_pred:
            # Focal loss
            focal = focal_loss(pred, target, alpha=1.0, gamma=2.0)
            focal_losses.append(focal.item())
            
            # BCE loss
            bce = bce_loss(pred, target)
            bce_losses.append(bce.item())
        
        # Plot focal loss
        plt.plot(y_pred, focal_losses, '--', linewidth=2, 
                label=f'Focal Loss (target={target})', alpha=0.8)
        
        # Plot BCE loss
        plt.plot(y_pred, bce_losses, '-', linewidth=2, 
                label=f'BCE Loss (target={target})', alpha=0.8)
    
    # Customize plot
    plt.xlabel('Prediction (y)', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Focal Loss vs BCE Loss Comparison\n(Dashed: Focal Loss, Solid: BCE Loss)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, 1)
    plt.ylim(0, max(plt.ylim()[1], 3))  # Set reasonable y-axis limit
    
    # Add some annotations
    plt.text(0.5, 0.95, 'Focal Loss focuses on hard examples\nBCE Loss treats all examples equally', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             ha='center', va='top')
    
    plt.tight_layout()
    
    # Save the plot as image file
    os.makedirs('/home/jaehun/workspace/guidewire_tip_detector_pytorch/results/test', exist_ok=True)
    output_path = '/home/jaehun/workspace/guidewire_tip_detector_pytorch/results/test/focal_vs_bce_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")
    plt.close()  # Close the figure to free memory
    
    # Print some key observations
    print("Key Observations:")
    print("1. Focal Loss (dashed lines) shows higher loss for predictions far from target")
    print("2. BCE Loss (solid lines) shows symmetric behavior around target")
    print("3. Focal Loss with gamma=2.0 focuses more on hard examples")
    print("4. When target=0.5, both losses are symmetric")
    print("5. Focal Loss reduces the loss for well-classified examples (close to target)")

def analyze_specific_points():
    """
    Analyze loss values at specific prediction points
    """
    target_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    prediction_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print("\nDetailed Analysis:")
    print("=" * 60)
    
    for target in target_values:
        print(f"\nTarget = {target}")
        print("-" * 30)
        for pred in prediction_points:
            focal = focal_loss(pred, target, alpha=1.0, gamma=2.0).item()
            bce = bce_loss(pred, target).item()
            print(f"Pred={pred:.1f}: Focal={focal:.4f}, BCE={bce:.4f}, Ratio={focal/bce:.2f}")

if __name__ == "__main__":
    print("Comparing Focal Loss vs BCE Loss")
    print("=" * 40)
    
    # Plot the comparison
    plot_focal_vs_bce()
    
    # Analyze specific points
    analyze_specific_points()
