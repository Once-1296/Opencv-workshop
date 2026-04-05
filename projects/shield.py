import cv2
import mediapipe as mp
import numpy as np
import math
import time
import warnings
from collections import deque

# Suppress those annoying Protobuf warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# --- INITIALIZATION ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils # For debugging
# model_complexity=0 is faster; min_detection lowered to 0.5 to be more 'generous'
hands = mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

shields_active = False
gesture_timers = {'Left': 0, 'Right': 0}
last_toggle_time = 0
hand_trails = {'Right': deque(maxlen=6), 'Left': deque(maxlen=6)}

def is_v_gesture(hand_landmarks):
    """Detects V-sign. Lowered thresholds to make detection easier."""
    wrist = hand_landmarks.landmark[0]
    def get_d(idx):
        t = hand_landmarks.landmark[idx]
        return math.sqrt((t.x - wrist.x)**2 + (t.y - wrist.y)**2)
    
    # 8=Index, 12=Middle, 16=Ring, 20=Pinky
    # Lowered from 0.25 to 0.15 to account for different camera distances
    if get_d(8) > 0.15 and get_d(12) > 0.15:
        if get_d(16) < 0.18 and get_d(20) < 0.18:
            return True
    return False

def draw_shield(img, center, radius, angle, label, alpha=1.0):
    if radius < 15: return
    overlay = np.zeros_like(img)
    b_val = int(255 * alpha)
    color = (0, int(140 * alpha), b_val)
    
    cv2.circle(overlay, center, int(radius), color, 3)
    # Drawing simple hexagon for performance test
    pts = []
    for i in range(6):
        theta = np.deg2rad(time.time()*60 + i*60)
        pts.append((int(center[0] + radius * math.cos(theta)), 
                    int(center[1] + radius * math.sin(theta))))
    for i in range(6):
        cv2.line(overlay, pts[i], pts[(i+1)%6], color, 2)
    
    cv2.addWeighted(img, 1.0, overlay, 1.0, 0, img)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0) # If screen is black, try changing 0 to 1

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    current_labels = []

    if results.multi_hand_landmarks:
        for i, landmarks in enumerate(results.multi_hand_landmarks):
            lbl = results.multi_handedness[i].classification[0].label
            current_labels.append(lbl)

            # --- DEBUG: Draw the hand skeleton so we can see if AI is working ---
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Logic
            wrist, mcp = landmarks.landmark[0], landmarks.landmark[9]
            cx, cy = int(((wrist.x + mcp.x)/2)*w), int(((wrist.y + mcp.y)/2)*h)

            if is_v_gesture(landmarks):
                if (time.time() - last_toggle_time) > 1.0:
                    shields_active = not shields_active
                    last_toggle_time = time.time()

            if shields_active:
                dist = math.sqrt((wrist.x - mcp.x)**2 + (wrist.y - mcp.y)**2)
                radius = int(dist * 1.3 * w)
                draw_shield(frame, (cx, cy), radius, 0, lbl)

    cv2.putText(frame, f"Shield: {'ON' if shields_active else 'OFF'}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Debug Shield', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()