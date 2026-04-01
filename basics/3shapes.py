import cv2
import numpy as np

# 1. LOAD & CHECK
img = cv2.imread('input.png')
if img is None:
    print("Error: Could not load image. Check the filename!")
    # Create a dummy black image so the code still runs for the demo
    img = np.zeros((500, 500, 3), dtype="uint8")

img = cv2.resize(img, (600, 600))
canvas = img.copy() # Keep original clean

# 2. DEFINING COLORS (BGR Format)
# Explain to students: OpenCV uses Blue-Green-Red, not RGB!
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

# 3. BASIC SHAPES
# Rectangle: (image, top-left, bottom-right, color, thickness)
cv2.rectangle(canvas, (50, 50), (200, 200), BLUE, 3)

# Circle: (image, center, radius, color, thickness) 
# Use -1 for thickness to FILL the shape
cv2.circle(canvas, (400, 150), 60, GREEN, -1)

# Line: (image, start, end, color, thickness)
cv2.line(canvas, (50, 250), (550, 250), RED, 5)

# 4. POLYGONS (Polyshape)
# Defining points for a diamond/star shape
pts = np.array([[300, 350], [400, 450], [300, 550], [200, 450]], np.int32)
# Reshape points for OpenCV (required for polylines)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(canvas, [pts], True, YELLOW, 3)

# 5. TEXT
# (image, text, position, font, scale, color, thickness)
cv2.putText(canvas, "VJTI OpenCV Workshop", (150, 300), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

# 6. PRO TIP: SEMI-TRANSPARENT OVERLAY
# Create a copy, draw a filled shape, then blend it
overlay = canvas.copy()
cv2.rectangle(overlay, (50, 400), (150, 550), (255, 0, 255), -1)
# alpha = transparency of original, beta = transparency of overlay
cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

# 7. DISPLAY
cv2.imshow('Final Drawings', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()