# visualize 1% and 2% accuracy window and save the images
# 
import cv2
import sys
import yaml
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# first generate two 640x640x3 images, pure black
image1 = np.zeros((640, 640, 3), dtype=np.uint8)
image2 = np.zeros((640, 640, 3), dtype=np.uint8)
# then draw a white rectangle on the image, the rectangle is centered at the center of the image
# the width of the rectangle is 12.8 pixels, and the height of the rectangle is 12.8 pixels
# the color of the rectangle is white
window_size_1 = 640 * 0.01
window_size_2 = 640 * 0.02

x1 = 320-round(window_size_1/2)
y1 = 320-round(window_size_1/2)
x2 = 320+round(window_size_1/2)
y2 = 320+round(window_size_1/2)
image1[x1:x2, y1:y2] = (0, 255, 0)
x1 = 320-round(window_size_2/2)
y1 = 320-round(window_size_2/2)
x2 = 320+round(window_size_2/2)
y2 = 320+round(window_size_2/2)
image2[x1:x2, y1:y2] = (0, 255, 0)
# save the images
cv2.imwrite('1%_accuracy_window.png', image1)
cv2.imwrite('2%_accuracy_window.png', image2)