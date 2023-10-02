import cv2

cap0 = cv2.VideoCapture(0)
ret0, frame0 = cap0.read()

while True:

    ret0, frame0 = cap0.read()
    if not ret0:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame0 = cv2.flip(frame0, 0)
    cv2.imshow('Webcam', frame0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap0.release()
cv2.destroyAllWindows()

