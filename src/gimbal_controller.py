#!/usr/bin/env python3
"""
Pi Gimbal Stabilizer - Main Controller

A camera gimbal stabilizer with auto-framing and IMU-based stabilization
running entirely on Raspberry Pi.

Features:
- Real-time face/body detection for subject tracking
- Auto-framing: keeps subject centered with proper composition
- IMU-based stabilization: compensates for camera shake
- 2-axis servo control (pitch/yaw) via PCA9685
- Photo capture with optimal framing

Author: Si6gma
License: MIT
"""

import cv2
import numpy as np
import time
import logging
import threading
from typing import Tuple, Optional, List
from dataclasses import dataclass
from collections import deque

from servo_driver import ServoDriver
from imu_sensor import IMUSensor
from stabilizer import Stabilizer
from auto_framing import AutoFramer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Camera settings
CAMERA_INDEX = 0          # 0 for Pi Camera, 1 for USB
FRAME_WIDTH = 1280        # Higher res for better detection
FRAME_HEIGHT = 720
FPS = 30

# Servo configuration (PCA9685)
SERVO_PITCH_CHANNEL = 0
SERVO_YAW_CHANNEL = 1
PITCH_MIN = 0
PITCH_MAX = 180
YAW_MIN = 0
YAW_MAX = 180
PITCH_CENTER = 90
YAW_CENTER = 90

# Stabilization settings
STABILIZATION_ENABLED = True
STABILIZATION_GAIN = 0.7  # How much to compensate for shake (0-1)

# Auto-framing settings
FRAMING_ENABLED = True
FACE_DETECTION_MODEL = "yolo"  # "haar", "dnn", or "yolo"
TRACKING_SMOOTHING = 0.15      # Lower = smoother but more lag

# Photo capture
PHOTO_OUTPUT_DIR = "./photos"
CAPTURE_KEY = ord('c')  # Press 'c' to capture
QUIT_KEY = ord('q')

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GimbalState:
    """Current state of the gimbal system."""
    pitch_angle: float = 90.0
    yaw_angle: float = 90.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    target_pitch: float = 90.0
    target_yaw: float = 90.0
    stabilization_active: bool = False
    tracking_active: bool = False
    subject_detected: bool = False


class GimbalController:
    """
    Main controller for the Pi Gimbal Stabilizer.
    
    Combines IMU stabilization with auto-framing subject tracking
    to create smooth, professional-looking footage.
    """
    
    def __init__(self):
        self.state = GimbalState()
        self.running = False
        
        # Initialize hardware
        logger.info("Initializing hardware...")
        self.servo = ServoDriver(
            pitch_channel=SERVO_PITCH_CHANNEL,
            yaw_channel=SERVO_YAW_CHANNEL,
            pitch_range=(PITCH_MIN, PITCH_MAX),
            yaw_range=(YAW_MIN, YAW_MAX)
        )
        
        self.imu = IMUSensor()
        self.stabilizer = Stabilizer(
            gain=STABILIZATION_GAIN,
            smoothing=0.3
        )
        self.framer = AutoFramer(
            smoothing=TRACKING_SMOOTHING,
            model_type=FACE_DETECTION_MODEL
        )
        
        # Initialize camera
        self.cap = None
        self._init_camera()
        
        # Movement smoothing
        self.pitch_buffer = deque(maxlen=5)
        self.yaw_buffer = deque(maxlen=5)
        
        logger.info("Gimbal controller initialized")
    
    def _init_camera(self):
        """Initialize camera with optimal settings."""
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # Check if Pi Camera
        if not self.cap.isOpened():
            logger.info("Trying Pi Camera module...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera initialized: {actual_width}x{actual_height}")
    
    def start(self):
        """Start the gimbal controller."""
        self.running = True
        
        # Center servos
        self.servo.set_position(PITCH_CENTER, YAW_CENTER)
        time.sleep(0.5)
        
        # Start IMU sampling in background
        self.imu.start()
        
        logger.info("Gimbal started - Press 'c' to capture, 'q' to quit")
        self._main_loop()
    
    def _main_loop(self):
        """Main control loop."""
        last_time = time.time()
        
        while self.running:
            # Calculate dt
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Frame capture failed")
                continue
            
            # Flip for mirror effect (optional)
            frame = cv2.flip(frame, 1)
            
            # Get IMU data
            imu_data = self.imu.get_reading()
            
            # Detect and track subject
            subject_data = self.framer.process_frame(frame)
            
            # Calculate desired angles based on tracking
            if subject_data.detected and FRAMING_ENABLED:
                # Auto-framing: move gimbal to center subject
                frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                target = self.framer.calculate_framing(
                    subject_data.bbox, 
                    frame.shape[:2]
                )
                self.state.target_pitch = target[0]
                self.state.target_yaw = target[1]
                self.state.tracking_active = True
            else:
                # No subject - return to center slowly
                self.state.target_pitch = PITCH_CENTER
                self.state.target_yaw = YAW_CENTER
                self.state.tracking_active = False
            
            # Apply stabilization if enabled
            if STABILIZATION_ENABLED and imu_data.valid:
                # Compensate for detected motion
                compensation = self.stabilizer.calculate_compensation(
                    imu_data.gyro,
                    imu_data.accel,
                    dt
                )
                self.state.pitch_rate = compensation[0]
                self.state.yaw_rate = compensation[1]
                self.state.stabilization_active = True
            else:
                self.state.pitch_rate = 0
                self.state.yaw_rate = 0
                self.state.stabilization_active = False
            
            # Update gimbal angles
            self._update_gimbal_position(dt)
            
            # Send to servos
            self.servo.set_position(
                self.state.pitch_angle,
                self.state.yaw_angle
            )
            
            # Draw overlays
            display_frame = self._draw_overlays(frame, subject_data)
            
            # Show preview
            cv2.imshow("Pi Gimbal Stabilizer", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == QUIT_KEY:
                self.running = False
            elif key == CAPTURE_KEY:
                self._capture_photo(frame)
        
        self.shutdown()
    
    def _update_gimbal_position(self, dt: float):
        """
        Update gimbal position combining tracking and stabilization.
        
        This blends the auto-framing target with IMU compensation
        to achieve smooth, stable footage.
        """
        # Smooth tracking movement
        tracking_pitch = self.state.pitch_angle + (
            self.state.target_pitch - self.state.pitch_angle
        ) * TRACKING_SMOOTHING
        
        tracking_yaw = self.state.yaw_angle + (
            self.state.target_yaw - self.state.yaw_angle
        ) * TRACKING_SMOOTHING
        
        # Add stabilization compensation
        if self.state.stabilization_active:
            # Compensate for rotation (inverted - move opposite to shake)
            tracking_pitch -= self.state.pitch_rate * dt * 50
            tracking_yaw -= self.state.yaw_rate * dt * 50
        
        # Apply limits
        self.state.pitch_angle = float(np.clip(
            tracking_pitch, PITCH_MIN, PITCH_MAX
        ))
        self.state.yaw_angle = float(np.clip(
            tracking_yaw, YAW_MIN, YAW_MAX
        ))
    
    def _draw_overlays(self, frame: np.ndarray, subject_data) -> np.ndarray:
        """Draw visual overlays on the preview."""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw rule of thirds grid (composition guide)
        third_x = w // 3
        third_y = h // 3
        color_grid = (50, 50, 50)
        cv2.line(display, (third_x, 0), (third_x, h), color_grid, 1)
        cv2.line(display, (2 * third_x, 0), (2 * third_x, h), color_grid, 1)
        cv2.line(display, (0, third_y), (w, third_y), color_grid, 1)
        cv2.line(display, (0, 2 * third_y), (w, 2 * third_y), color_grid, 1)
        
        # Draw subject bounding box
        if subject_data.detected:
            x, y, w_box, h_box = subject_data.bbox
            cv2.rectangle(display, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            
            # Draw target center
            cx, cy = subject_data.center
            cv2.circle(display, (cx, cy), 5, (0, 255, 0), -1)
            
            # Draw label
            label = f"Subject: {subject_data.confidence:.2f}"
            cv2.putText(display, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw gimbal position info
        status_lines = [
            f"Pitch: {self.state.pitch_angle:.1f}°",
            f"Yaw: {self.state.yaw_angle:.1f}°",
            f"Stabilization: {'ON' if self.state.stabilization_active else 'OFF'}",
            f"Tracking: {'ON' if self.state.tracking_active else 'OFF'}",
            "[C]apture  [Q]uit"
        ]
        
        y_offset = 30
        for line in status_lines:
            cv2.putText(display, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return display
    
    def _capture_photo(self, frame: np.ndarray):
        """Capture a photo with timestamp."""
        import os
        from datetime import datetime
        
        os.makedirs(PHOTO_OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PHOTO_OUTPUT_DIR}/capture_{timestamp}.jpg"
        
        cv2.imwrite(filename, frame)
        logger.info(f"Photo captured: {filename}")
        
        # Visual feedback
        cv2.putText(frame, "PHOTO CAPTURED!", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Pi Gimbal Stabilizer", frame)
        cv2.waitKey(100)
    
    def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self.running = False
        self.imu.stop()
        self.servo.center()
        time.sleep(0.3)
        self.servo.disable()
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")


def main():
    """Entry point."""
    controller = GimbalController()
    try:
        controller.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
