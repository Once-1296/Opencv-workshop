import cv2
import numpy as np
import math
import time

# Open webcam
cap = cv2.VideoCapture(0)

# Define the 8 vertices of a 3D unit cube
vertices = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1]
])

# Define the 12 edges connecting the vertices
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting lines
]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Center of screen and scale (size of the cube)
    cx, cy = w // 2, h // 2
    scale = 100

    # Continuous time-based angle for Y-axis spin
    angle_time = time.time() * 1.5 
    
    # Fixed 45-degree angle for X-axis tilt (np.pi / 4 radians)
    angle_45 = math.pi / 4

    # Rotation Matrix for X-axis (Fixed 45 deg tilt so you can see the top/bottom)
    rot_x = np.array([
        [1, 0, 0],
        [0, math.cos(angle_45), -math.sin(angle_45)],
        [0, math.sin(angle_45),  math.cos(angle_45)]
    ])

    # Rotation Matrix for Y-axis (Continuous spin)
    rot_y = np.array([
        [math.cos(angle_time), 0, math.sin(angle_time)],
        [0, 1, 0],
        [-math.sin(angle_time), 0, math.cos(angle_time)]
    ])

    # Store 2D projected points
    projected_points = []

    for v in vertices:
        # Apply Y rotation (spin), then X rotation (tilt)
        rotated_v = np.dot(rot_y, v)
        rotated_v = np.dot(rot_x, rotated_v)
        
        # Orthographic Projection: Ignore Z-axis, map X and Y to the 2D screen
        x = int(cx + rotated_v[0] * scale)
        y = int(cy + rotated_v[1] * scale)
        projected_points.append((x, y))

    # Draw the lines connecting the projected points
    for edge in edges:
        pt1 = projected_points[edge[0]]
        pt2 = projected_points[edge[1]]
        cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

    # Show the frame
    cv2.imshow("Rotating 3D Cube", frame)

    # Press 'ESC' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()