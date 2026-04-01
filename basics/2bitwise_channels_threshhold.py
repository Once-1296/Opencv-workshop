import cv2
import numpy as np

# 1. LOAD IMAGE
img = cv2.imread('input.png')
img = cv2.resize(img, (500, 500))

# 2. CREATE A MASK
# A mask must be the same size as the image, but in grayscale (1 channel)
mask = np.zeros(img.shape[:2], dtype="uint8")

# Let's draw a white circle on our black mask
# The white area (255) is what we want to "keep"
cv2.circle(mask, (250, 250), 200, 255, -1)

# 3. BITWISE OPERATIONS ON ACTUAL IMAGE
# bitwise_and: Keeps only the pixels where the mask is white
masked_img = cv2.bitwise_and(img, img, mask=mask)

# bitwise_not: Inverts the mask (White becomes Black, Black becomes White)
mask_inv = cv2.bitwise_not(mask)

# bitwise_or: Using the inverted mask to "brighten" or merge (advanced demo)
# Usually, we use 'and' for masking, but 'not' is great for showing background subtraction
inverted_img = cv2.bitwise_not(img)

# 4. DISPLAY
cv2.imshow("Original", img)
cv2.imshow("The Mask (Alpha Channel Concept)", mask)
cv2.imshow("Masked Output (The Cutout)", masked_img)
cv2.imshow("Inverted Image", inverted_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Colors
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Geometry
cropped = img[100:400, 100:400] # [y_start:y_end, x_start:x_end]
flipped = cv2.flip(img, 1) # 1=Horizontal, 0=Vertical

# Thresholding
# Simple Binary
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Adaptive (Handles shadows better)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

cv2.imshow('Binary', thresh)
cv2.imshow('Adaptive', adaptive)
cv2.imshow('Cropped', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()