import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

image_size = 640
center_x = image_size // 2
center_y = image_size // 2
window_color = [0, 255, 0]

image_window_acc_5 = np.zeros((image_size, image_size, 3), dtype=np.uint8)
window_size = 0.05
window_width = int(image_size * window_size)
window_height = int(image_size * window_size)
window_x = center_x - window_width // 2
window_y = center_y - window_height // 2
image_window_acc_5[window_y:window_y+window_height, window_x:window_x+window_width] = window_color

image_window_acc_1 = np.zeros((image_size, image_size, 3), dtype=np.uint8)
window_size = 0.01
window_width = int(image_size * window_size)
window_height = int(image_size * window_size)
window_x = center_x - window_width // 2
window_y = center_y - window_height // 2
image_window_acc_1[window_y:window_y+window_height, window_x:window_x+window_width] = window_color

image_window_acc_2 = np.zeros((image_size, image_size, 3), dtype=np.uint8)
window_size = 0.02
window_width = int(image_size * window_size)
window_height = int(image_size * window_size)
window_x = center_x - window_width // 2
window_y = center_y - window_height // 2
image_window_acc_2[window_y:window_y+window_height, window_x:window_x+window_width] = window_color

image_window_acc_0_5 = np.zeros((image_size, image_size, 3), dtype=np.uint8)
window_size = 0.005
window_width = int(image_size * window_size)
window_height = int(image_size * window_size)
window_x = center_x - window_width // 2
window_y = center_y - window_height // 2
image_window_acc_0_5[window_y:window_y+window_height, window_x:window_x+window_width] = window_color



# save the images
cv2.imwrite('image_window_acc_5.png', image_window_acc_5)
cv2.imwrite('image_window_acc_1.png', image_window_acc_1)
cv2.imwrite('image_window_acc_2.png', image_window_acc_2)
cv2.imwrite('image_window_acc_0_5.png', image_window_acc_0_5)