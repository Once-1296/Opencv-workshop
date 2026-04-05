import cv2
import numpy as np
from constants import input_img
img = cv2.imread(input_img)

# 1. Custom Kernels
sharpen = np.array([[0, -1, 0], [-1, 10, -1], [0, -1, 0]])
emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

res_sharp = cv2.filter2D(img, -1, sharpen)
res_emboss = cv2.filter2D(img, -1, emboss)

# 2. Padding (Borders)
# top, bottom, left, right
constant = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0,0,255])
reflect = cv2.copyMakeBorder(img, 150, 150, 150, 150, cv2.BORDER_REFLECT)

cv2.imshow('Sharpened', res_sharp)
cv2.imshow('Embossed', res_emboss)
cv2.imshow('Padded border', constant)
cv2.imshow('Reflected boorder', reflect)
cv2.waitKey(0)
cv2.destroyAllWindows()