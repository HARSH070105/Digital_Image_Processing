import cv2
import numpy as np

img = cv2.imread('input.jpg')

mag = 50
bright = np.clip(img.astype(int), 0, 255).astype(np.uint8)
cv2.imwrite('bright_input.jpg', bright)