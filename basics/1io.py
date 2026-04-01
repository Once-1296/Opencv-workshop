import cv2

# --- IMAGE IO ---
img = cv2.imread('input.jpg')
if img is not None:
    cv2.imshow('Static Image (Press any key)', img)
    cv2.imwrite('copy_of_input.jpg', img)
    cv2.waitKey(0)

# --- VIDEO & WEBCAM IO ---
cap = cv2.VideoCapture(0) # '0' is default webcam
# Define codec and create VideoWriter object to save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('workshop_output.avi', fourcc, 20.0, (640, 480))

print("Capturing... Press 'q' to stop and save the last frame.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    cv2.imshow('Webcam Feed', frame)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('captured_frame.jpg', frame) # Capture single frame
        break

cap.release()
out.release()
cv2.destroyAllWindows()