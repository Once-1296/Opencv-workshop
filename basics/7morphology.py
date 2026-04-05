import cv2
import numpy as np
from constants import erosion_dilation_img, hat_input_img, opening_input_img,closing_input_img
img = cv2.imread(erosion_dilation_img, 0)
_, threshed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)

eroded = cv2.erode(threshed, kernel, iterations=5)
dilated = cv2.dilate(threshed, kernel, iterations=5)
grad = cv2.morphologyEx(threshed, cv2.MORPH_GRADIENT, kernel) 
cv2.imshow('Original', img)
cv2.imshow('Original_clearer', threshed)
cv2.imshow('Erode', eroded)
cv2.imshow('Dilate', dilated)
cv2.imshow('Gradient', grad)
cv2.waitKey(0)
cv2.destroyAllWindows()

to_open = cv2.imread(opening_input_img)
to_open = cv2.cvtColor(to_open, cv2.COLOR_BGR2GRAY)
to_close = cv2.imread(closing_input_img)
to_close = cv2.cvtColor(to_close, cv2.COLOR_BGR2GRAY)
opening = cv2.morphologyEx(to_open, cv2.MORPH_OPEN, kernel)   
closing = cv2.morphologyEx(to_close, cv2.MORPH_CLOSE, kernel)  

cv2.imshow('Open_input', to_open)
cv2.imshow('Opening', opening)
cv2.imshow('Close_input', to_close)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(hat_input_img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((10,10), np.uint8)
tophat = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)   
blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)   

cv2.imshow('Tophat', tophat)
cv2.imshow('Blackhat', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()