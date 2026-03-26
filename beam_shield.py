import cv2
import mediapipe as mp
import numpy as np
import random
import math

# Initialize Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

p1_beam_buffer = 0
p2_shield_buffer = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    beam_origin = None
    shield_center = None

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            wrist = hand_lms.landmark[0]
            index_tip = hand_lms.landmark[8]
            knuckle = hand_lms.landmark[9] 
            
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            kx, ky = int(knuckle.x * w), int(knuckle.y * h)
            
            hand_size = math.sqrt((wx-kx)**2 + (wy-ky)**2) + 1

            if wx < w/2:
                beam_origin = (ix, iy)
                if ix > wx + (hand_size * 0.5): p1_beam_buffer = 15
            else:
                shield_center = (kx, ky)
                if iy < wy - (hand_size * 0.5): p2_shield_buffer = 15

    # --- RENDER LOGIC ---
    if p1_beam_buffer > 0 and beam_origin:
        is_blocked = (p2_shield_buffer > 0 and shield_center)
        
        if is_blocked:
            shield_radius = int(hand_size * 1.3)
            end_x = shield_center[0] - shield_radius 
            impact_y = beam_origin[1]
            
            # 1. DRAW DOME
            axes = (shield_radius, int(hand_size * 2.2))
            cv2.ellipse(frame, shield_center, axes, 0, 110, 250, (255, 180, 50), 3)
            cv2.ellipse(frame, shield_center, (axes[0]-5, axes[1]-10), 0, 120, 240, (255, 255, 255), 8)
            
            # 2. LONGER DEFLECTION SPARKS (The "Splash" Effect)
            for _ in range(8): # More sparks
                angle = random.uniform(-1.2, 1.2)
                # INCREASED LENGTH: from (20-60) to (80-150)
                spark_len = random.randint(80, 150) 
                
                s_x = int(end_x - math.cos(angle) * spark_len)
                s_y = int(impact_y + math.sin(angle) * spark_len)
                
                # Draw thick base spark
                cv2.line(frame, (end_x, impact_y), (s_x, s_y), (0, 255, 255), 3)
                # Draw white core for the spark to make it "pop"
                cv2.line(frame, (end_x, impact_y), (int(end_x + (s_x-end_x)*0.7), int(impact_y + (s_y-impact_y)*0.7)), (255, 255, 255), 1)
        else:
            end_x = w
            if shield_center:
                frame[:, :, 2] = np.clip(frame[:, :, 2] + 40, 0, 255).astype(np.uint8)

        # 3. THE BEAM
        cv2.line(frame, beam_origin, (end_x, beam_origin[1]), (255, 100, 0), 40) 
        cv2.line(frame, beam_origin, (end_x, beam_origin[1]), (255, 255, 255), 10) 

        # 4. START OF BEAM BLAST (The Flare)
        flare_size = random.randint(40, 60)
        cv2.circle(frame, beam_origin, flare_size, (255, 150, 50), -1)
        cv2.circle(frame, beam_origin, flare_size - 15, (255, 255, 255), -1)
        
        p1_beam_buffer -= 1

    if p2_shield_buffer > 0: p2_shield_buffer -= 1

    cv2.imshow("HIGH INTENSITY BATTLE", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()