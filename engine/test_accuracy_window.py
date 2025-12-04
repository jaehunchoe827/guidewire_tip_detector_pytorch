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
# then draw a green circle on the image, the circle is centered at the center of the image
# the diameter of the circle is 1% and 2% of the image size respectively
# the color of the circle is green
window_size_1 = 640 * 0.01
window_size_2 = 640 * 0.02

center = (320, 320)
radius_1 = int(window_size_1 / 2)
radius_2 = int(window_size_2 / 2)
cv2.circle(image1, center, radius_1, (0, 255, 0), -1)
cv2.circle(image2, center, radius_2, (0, 255, 0), -1)
# save the images
cv2.imwrite('1%_accuracy_window.png', image1)
cv2.imwrite('2%_accuracy_window.png', image2)