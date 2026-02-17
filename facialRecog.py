import cv2
import math
import serial
import time
import os

# Import configuration
try:
    from config_local import SERIAL_PORT, SERIAL_BAUD, CAMERA_INDEX, DEFAULT_PITCH
except ImportError:
    # Default configuration - update these for your system
    SERIAL_PORT = "/dev/cu.usbmodemDC5475C3BB642"  # Change to your Arduino port
    SERIAL_BAUD = 9600
    CAMERA_INDEX = 1  # 0 for default camera, 1 for external webcam
    DEFAULT_PITCH = 40

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Establish a serial connection
print(f"Connecting to Arduino on {SERIAL_PORT}...")
ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD)

# Allow some time for the connection to establish
time.sleep(2)

print("⚠️  SAFETY REMINDER: Ensure laser is removed or replaced with LED before use!")
print("Press 'q' to quit\n")

# Open the webcam (the value inside depends on your system)
cap = cv2.VideoCapture(CAMERA_INDEX)

# Store the resolution of the camera (in pixels)
f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Function to calculate angle
def anglecalc(horizontal, vertical):
    return math.floor(math.atan(vertical / horizontal) * (180 / math.pi))

    # Print the video resolution


print(f"Video resolution: {f_w}x{f_h}")

while True:
    # Read frames from the webcam
    ret, frame = cap.read()

    # Check if frames were read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frames to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around faces and a red dot at the center
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        c_x, c_y = x + w // 2, y + h // 2  # Find center of rectangle

        # Draw a point at the center of rectangle
        cv2.circle(frame, (c_x, c_y), radius=2, color=(0, 0, 255), thickness=-1)

        # Calculate angles and send to Arduino via serial connection
        yaw = anglecalc(c_x, f_h - c_y)
        # pitch = anglecalc(c_z,c_x)
        pitch = DEFAULT_PITCH
        data = str(yaw).encode()
        print(data)
        # time.sleep(1)
        ser.write(data)

    # Draw a green point at center of frame
    cv2.circle(frame, (f_w // 2, f_h // 2), radius=2, color=(0, 255, 0), thickness=-1)

    # Display resulting frames
    cv2.imshow("Webcam", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam, destroy all windows and close serial connection after loop ends
cap.release()
cv2.destroyAllWindows()
ser.close()
