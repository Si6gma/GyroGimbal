import cv2
import math

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Open the webcam
cap = cv2.VideoCapture(1)

# Storing the resolution of the camera (in pixels)
f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Function to calculate angle
def anglecalc(horizontal,vertical):
    tan1 = vertical/horizontal
    yaw = math.atan(tan1)
    yaw = math.floor(yaw*(180/math.pi))
    print(yaw)
    

# Displaying the resolution
print(f"Video resolution: {f_w}x{f_h}")

while True:
    # Read frames from the webcam
    ret, frame = cap.read()

    # Check if the frames were read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and a red dot at the center
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Finding the center of the rectangle
        c_x, c_y = x + w//2, (y + h//2)

        # Drawing a point at the center of the rectangle
        cv2.circle(frame, (c_x, c_y), radius=2, color=(0, 0, 255), thickness=-1)

        # Sending the coordinates to the function to calculate the angle, (y is inverted since 0,0 is on the top left and we want it to be on the bottom left)
        anglecalc(c_x,f_h-c_y)


    # Draws a green point at the center of the frame
    cv2.circle(frame, (int(f_w/2), int(f_h/2)), radius=2, color=(0,255,0), thickness=-1)

    # Display the resulting frames
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
