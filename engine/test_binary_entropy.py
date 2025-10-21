import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the loss function
def binary_cross_entropy_loss(t, p):
    """
    Binary cross-entropy loss: -(t * log(p) + (1-t) * log(1-p))
    where t is the true label (0 or 1) and p is the predicted probability
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)
    
    return -(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))

# Create meshgrid for t and p
epsilon = 1e-5
t_range = np.linspace(0, 1, 100)
p_range = np.linspace(epsilon, 1 - epsilon, 100)
T, P = np.meshgrid(t_range, p_range)

# Calculate loss for each combination
Loss = binary_cross_entropy_loss(T, P)

# Create 2x2 subplot layout
fig = plt.figure(figsize=(20, 16))

# 3D plot (top left)
ax1 = fig.add_subplot(221, projection='3d')

# Plot the surface
surf = ax1.plot_surface(T, P, Loss, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)

# Set labels and title for 3D plot
ax1.set_xlabel('True Label (t)', fontsize=12)
ax1.set_ylabel('Predicted Probability (p)', fontsize=12)
ax1.set_zlabel('Loss', fontsize=12)
ax1.set_title('3D Surface: Binary Cross-Entropy Loss', fontsize=14)

# Set viewing angle for better visualization
ax1.view_init(elev=20, azim=45)

# Add some specific points of interest to 3D plot
# When t=0 (perfect prediction for class 0)
ax1.scatter([0], [epsilon], [binary_cross_entropy_loss(0, epsilon)], 
           color='red', s=100, label='t≈0, p≈0 (good prediction)')

# When t=1 (perfect prediction for class 1)  
ax1.scatter([1], [1 - epsilon], [binary_cross_entropy_loss(1, 1 - epsilon)], 
           color='green', s=100, label='t≈1, p≈1 (good prediction)')

# When t=0 but p=1 (worst prediction)
ax1.scatter([0], [1 - epsilon], [binary_cross_entropy_loss(0, 1 - epsilon)], 
           color='blue', s=100, label='t≈0, p≈1 (worst prediction)')

# When t=1 but p=0 (worst prediction)
ax1.scatter([1], [epsilon], [binary_cross_entropy_loss(1, epsilon)], 
           color='orange', s=100, label='t≈1, p≈0 (worst prediction)')

ax1.legend()

# 2D contour plot (top right)
ax2 = fig.add_subplot(222)
contour = ax2.contourf(T, P, Loss, levels=20, cmap='viridis')
ax2.contour(T, P, Loss, levels=20, colors='black', alpha=0.3, linewidths=0.5)

# Add colorbar for contour plot
cbar = plt.colorbar(contour, ax=ax2, label='Loss Value')
ax2.set_xlabel('True Label (t)')
ax2.set_ylabel('Predicted Probability (p)')
ax2.set_title('2D Contour: Binary Cross-Entropy Loss', fontsize=14)

# Add diagonal line where t = p (perfect predictions)
ax2.plot([0, 1], [epsilon, 1 - epsilon], 'r--', linewidth=2, label='Perfect predictions (t=p)')

# Add the same key points to contour plot
ax2.scatter([0], [epsilon], color='red', s=100, label='t≈0, p≈0 (good)')
ax2.scatter([1], [1 - epsilon], color='green', s=100, label='t≈1, p≈1 (good)')
ax2.scatter([0], [1 - epsilon], color='blue', s=100, label='t≈0, p≈1 (worst)')
ax2.scatter([1], [epsilon], color='orange', s=100, label='t≈1, p≈0 (worst)')

ax2.legend()

# Loss along t=0 (bottom left)
ax3 = fig.add_subplot(223)
p_line = np.linspace(epsilon, 1 - epsilon, 100)
loss_t0 = binary_cross_entropy_loss(0, p_line)
ax3.plot(p_line, loss_t0, 'b-', linewidth=2, label='t=0 (true class 0)')
ax3.set_xlabel('Predicted Probability (p)')
ax3.set_ylabel('Loss')
ax3.set_title('Loss when True Label t=0')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Add some key points
ax3.scatter([epsilon], [binary_cross_entropy_loss(0, epsilon)], 
           color='red', s=100, label=f'p≈0 (good: loss={binary_cross_entropy_loss(0, epsilon):.3f})')
ax3.scatter([1-epsilon], [binary_cross_entropy_loss(0, 1-epsilon)], 
           color='blue', s=100, label=f'p≈1 (worst: loss={binary_cross_entropy_loss(0, 1-epsilon):.3f})')
ax3.legend()

# Loss along t=1 (bottom right)
ax4 = fig.add_subplot(224)
loss_t1 = binary_cross_entropy_loss(1, p_line)
ax4.plot(p_line, loss_t1, 'g-', linewidth=2, label='t=1 (true class 1)')
ax4.set_xlabel('Predicted Probability (p)')
ax4.set_ylabel('Loss')
ax4.set_title('Loss when True Label t=1')
ax4.grid(True, alpha=0.3)
ax4.legend()

# Add some key points
ax4.scatter([1-epsilon], [binary_cross_entropy_loss(1, 1-epsilon)], 
           color='green', s=100, label=f'p≈1 (good: loss={binary_cross_entropy_loss(1, 1-epsilon):.3f})')
ax4.scatter([epsilon], [binary_cross_entropy_loss(1, epsilon)], 
           color='orange', s=100, label=f'p≈0 (worst: loss={binary_cross_entropy_loss(1, epsilon):.3f})')
ax4.legend()

plt.tight_layout()
plt.show()

print("Loss function analysis:")
print(f"Minimum loss (when t=p): {binary_cross_entropy_loss(0.5, 0.5):.4f}")
print(f"Maximum loss (when t=0, p=1): {binary_cross_entropy_loss(0, 1 - epsilon):.4f}")
print(f"Maximum loss (when t=1, p=0): {binary_cross_entropy_loss(1, epsilon):.4f}")