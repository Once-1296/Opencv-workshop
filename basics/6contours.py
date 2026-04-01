import cv2
import numpy as np

img = cv2.imread('input.png') # Use an image with clear shapes
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Edges
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
canny = cv2.Canny(gray, 50, 150)

# 2. Contours & Hierarchy
# RETR_TREE retrieves all contours and creates a full family hierarchy
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500: # Ignore tiny noise
        # Perimeter
        peri = cv2.arcLength(cnt, True)
        # Shape Approx
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # Bounding Box & Circle
        x, y, w, h = cv2.boundingRect(cnt)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        
        # Draw everything
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(img, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

cv2.imshow('Analysis', img)
cv2.waitKey(0)
cv2.destroyAllWindows()