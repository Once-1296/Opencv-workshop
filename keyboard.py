import cv2
import mediapipe as mp
import numpy as np
import time

# --- INITIALIZE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

# --- KEYBOARD CONFIG ---
rows = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M"],
    ["BKSP", "SPC", "CLR"]
]

# State Variables
sentence = ""
last_key_press = 0
CYAN_GLOW = (255, 180, 0) # BGR

def draw_kb(img, finger_pos, is_pressing):
    """Renders the interactive keyboard overlay."""
    kb_overlay = np.zeros_like(img)
    
    for r_idx, row in enumerate(rows):
        for c_idx, key in enumerate(row):
            # Dynamic width for special keys
            w_box = 120 if key in ["BKSP", "SPC", "CLR"] else 50
            # Offset rows to look like a real keyboard
            off = 80 if r_idx == 3 else (r_idx * 20)
            
            x, y = 30 + (c_idx * (w_box + 8)) + off, 50 + (r_idx * 60)
            
            # Hover/Press Logic
            col = CYAN_GLOW
            if x < finger_pos[0] < x + w_box and y < finger_pos[1] < y + 50:
                col = (255, 255, 255) if is_pressing else (255, 100, 0)
                if is_pressing: 
                    cv2.rectangle(kb_overlay, (x, y), (x + w_box, y + 50), (255, 150, 0), -1)
            
            # Draw Key Border and Text
            cv2.rectangle(kb_overlay, (x, y), (x + w_box, y + 50), col, 2)
            cv2.putText(kb_overlay, key, (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # Add a slight glow for the 'Sci-Fi' look
    glow = cv2.GaussianBlur(kb_overlay, (15, 15), 0)
    cv2.addWeighted(img, 1.0, glow, 0.6, 0, img)
    cv2.addWeighted(img, 1.0, kb_overlay, 1.0, 0, img)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Process Hand
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    ix, iy = 0, 0 # Finger pointer coordinates
    is_poking = False

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            wrist = hand_lms.landmark[0]
            idx_tip = hand_lms.landmark[8] # Index Finger Tip
            
            # Convert normalized coords to pixels
            ix, iy = int(idx_tip.x * w), int(idx_tip.y * h)
            
            # --- THE "CLICK" LOGIC ---
            # We compare the Z (depth) of the index tip to the wrist.
            # If the index finger moves "forward" toward the camera, it's a click.
            is_poking = idx_tip.z < wrist.z - 0.12 
            
            draw_kb(frame, (ix, iy), is_poking)

            # --- INPUT HANDLING ---
            if is_poking and (time.time() - last_key_press) > 0.8:
                for r_idx, row in enumerate(rows):
                    for c_idx, key in enumerate(row):
                        wb = 120 if key in ["BKSP", "SPC", "CLR"] else 50
                        xk, yk = 30 + (c_idx*(wb+8)) + (80 if r_idx==3 else r_idx*20), 50 + (r_idx*60)
                        
                        # Check if index tip is inside the key boundaries
                        if xk < ix < xk + wb and yk < iy < yk + 50:
                            if key == "SPC": sentence += " "
                            elif key == "CLR": sentence = ""
                            elif key == "BKSP": sentence = sentence[:-1]
                            else: sentence += key
                            last_key_press = time.time()

    # --- RENDER TEXT LOG ---
    # Draw a glass-style box for the typed sentence
    cv2.rectangle(frame, (40, h-85), (w-40, h-25), CYAN_GLOW, 2)
    cv2.putText(frame, f"TYPED: {sentence}", (60, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Air Keyboard v1.0', frame)
    if cv2.waitKey(1) & 0xFF == 27: break # ESC to quit

cap.release()
cv2.destroyAllWindows()