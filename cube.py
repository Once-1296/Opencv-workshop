import cv2
import mediapipe as mp
import numpy as np
import time

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- State Variables ---
cube_visible = False
last_v_time = 0  # To prevent flickering when toggling
scale = 150
rot_x, rot_y = 0, 0
last_finger_pos = None

def get_cube_points(center, size, rx, ry):
    s = size / 2
    points = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
    ])
    rx, ry = np.radians(rx), np.radians(ry)
    rotX = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    rotY = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    rotated = points @ rotX.T @ rotY.T
    return (rotated[:, :2] + center).astype(int)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    screen_center = np.array([w // 2, h // 2])
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            index_tip = lm.landmark[8]
            middle_tip = lm.landmark[12]
            thumb_tip = lm.landmark[4]

            # 1. Toggle Visibility with "V" Symbol
            # Check if index and middle are up, and ring/pinky are down
            is_v = index_tip.y < lm.landmark[6].y and middle_tip.y < lm.landmark[10].y \
                   and lm.landmark[16].y > lm.landmark[14].y # Ring finger down
            
            if is_v and (time.time() - last_v_time > 0.5): # 0.5s delay between toggles
                cube_visible = not cube_visible
                last_v_time = time.time()

            # 2. Rotation Control (Move Index Finger to rotate)
            curr_finger_pos = (index_tip.x, index_tip.y)
            if last_finger_pos:
                # Calculate movement delta
                dx = curr_finger_pos[0] - last_finger_pos[0]
                dy = curr_finger_pos[1] - last_finger_pos[1]
                rot_y += dx * 500
                rot_x -= dy * 500
            last_finger_pos = curr_finger_pos

            # 3. Zoom Control (Thumb to Index distance)
            dist = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            scale = int(dist * 700) 

    # --- Draw the Cube in the Center ---
    if cube_visible:
        pts = get_cube_points(screen_center, scale, rot_x, rot_y)
        edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
        for start, end in edges:
            cv2.line(frame, tuple(pts[start]), tuple(pts[end]), (255, 0, 255), 3) # Magenta
        
        cv2.putText(frame, "CUBE ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Center Cube Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()