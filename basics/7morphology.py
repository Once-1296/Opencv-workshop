import cv2
import numpy as np

img = cv2.imread('input.png', 0)
_, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

kernel = np.array([[1,1,1], [1,1,1], [1,1,1]])

eroded = cv2.erode(mask, kernel, iterations=1)
dilated = cv2.dilate(mask, kernel, iterations=1)
cv2.imshow('Erode', eroded)
cv2.imshow('Dilate', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()

opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  

cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel) 
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)   
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)   

cv2.imshow('Gradient', grad)
cv2.imshow('Tophat', tophat)
cv2.imshow('Blackhat', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()