import cv2
import numpy as np

img = cv2.imread('input.png', 0)
_, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5,5), np.uint8)

# Basic
eroded = cv2.erode(mask, kernel, iterations=1)
dilated = cv2.dilate(mask, kernel, iterations=1)

# Advanced
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close holes
grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel) # Outline
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)   # Bright spots

cv2.imshow('Gradient', grad)
cv2.imshow('Opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()