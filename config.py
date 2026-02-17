"""
Configuration for Pi Gimbal Stabilizer

Edit these values to match your hardware setup.
"""

# =============================================================================
# CAMERA SETTINGS
# =============================================================================

# Camera index (0 for Pi Camera via V4L2, 1 for USB camera)
CAMERA_INDEX = 0

# Resolution (lower = faster processing, higher = better detection)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# =============================================================================
# SERVO SETTINGS (PCA9685)
# =============================================================================

# I2C address of PCA9685 (default is 0x40)
PCA9685_ADDRESS = 0x40

# Servo channels
SERVO_PITCH_CHANNEL = 0
SERVO_YAW_CHANNEL = 1

# Servo angle limits (degrees)
PITCH_MIN = 0
PITCH_MAX = 180
YAW_MIN = 0
YAW_MAX = 180

# Center positions
PITCH_CENTER = 90
YAW_CENTER = 90

# Servo pulse width limits (microseconds) - adjust for your servos
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

# =============================================================================
# IMU SETTINGS (MPU6050/MPU9250)
# =============================================================================

# I2C address (MPU6050 default: 0x68, AD0 high: 0x69)
IMU_ADDRESS = 0x68

# Sample rate (Hz)
IMU_SAMPLE_RATE = 100

# Ranges
ACCEL_RANGE = "2G"   # Options: "2G", "4G", "8G"
GYRO_RANGE = "500DPS"  # Options: "250DPS", "500DPS", "1000DPS", "2000DPS"

# =============================================================================
# STABILIZATION SETTINGS
# =============================================================================

# Enable/disable stabilization
STABILIZATION_ENABLED = True

# Compensation gain (0-1): how aggressively to counteract movement
# Higher = more stable but may feel robotic
STABILIZATION_GAIN = 0.7

# Smoothing factor (0-1): higher = smoother but more lag
STABILIZATION_SMOOTHING = 0.3

# =============================================================================
# AUTO-FRAMING SETTINGS
# =============================================================================

# Enable/disable subject tracking
FRAMING_ENABLED = True

# Detection model: "haar", "dnn", or "yolo"
# - "haar": Fast but less accurate
# - "dnn": Better accuracy, requires model files
# - "yolo": Best accuracy, requires YOLO weights
DETECTION_MODEL = "haar"

# Tracking smoothing (0-1)
TRACKING_SMOOTHING = 0.15

# Composition style: "center" or "rule_of_thirds"
COMPOSITION_STYLE = "center"

# Headroom ratio (space above subject's head)
HEADROOM_RATIO = 0.15

# =============================================================================
# CONTROL KEYS
# =============================================================================

CAPTURE_KEY = ord('c')  # Press to take photo
QUIT_KEY = ord('q')     # Press to exit
TOGGLE_STABILIZATION = ord('s')
TOGGLE_TRACKING = ord('t')

# =============================================================================
# PHOTO SETTINGS
# =============================================================================

PHOTO_OUTPUT_DIR = "./photos"
PHOTO_QUALITY = 95  # JPEG quality (0-100)
