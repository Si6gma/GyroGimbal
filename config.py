"""Configuration for LaserTurret - Face Tracking Servo Control

Copy this file to config.local.py and update values for your setup.
config.local.py is gitignored and won't be committed.
"""

# Serial Configuration
SERIAL_PORT = "/dev/cu.usbmodemDC5475C3BB642"  # Update for your system
SERIAL_BAUD = 9600

# Camera Configuration
CAMERA_INDEX = 1  # 0 for default, 1 for external webcam

# Servo Configuration
DEFAULT_PITCH = 40  # Fixed pitch angle (degrees)

# Face Detection Configuration
FACE_DETECTION_SCALE = 1.1
FACE_DETECTION_NEIGHBORS = 4
