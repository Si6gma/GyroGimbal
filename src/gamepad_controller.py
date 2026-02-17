#!/usr/bin/env python3
"""
Gamepad Controller for Pi Gimbal Stabilizer

Supports Xbox, PlayStation, and generic USB gamepads.
Uses pygame for cross-platform compatibility.

Controls:
    Left Stick        - Manual pan/tilt
    Right Stick       - Fine adjustment
    A/Cross Button    - Capture photo
    B/Circle Button   - Center gimbal
    X/Square Button   - Toggle stabilization
    Y/Triangle Button - Toggle tracking
    LB/L1             - Decrease speed
    RB/R1             - Increase speed
    D-Pad Up/Down     - Adjust pitch limits
    D-Pad Left/Right  - Adjust yaw limits
    Start             - Exit

Usage:
    python gamepad_controller.py
"""

import logging
import threading
import time
import numpy as np
import cv2
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available. Gamepad control disabled.")

from gimbal_controller import GimbalController
from servo_driver import ServoDriver
from imu_sensor import IMUSensor
from stabilizer import Stabilizer
from auto_framing import AutoFramer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GamepadController:
    """
    Gamepad controller for manual gimbal operation.
    
    Supports Xbox, PlayStation, and generic USB controllers.
    """
    
    # Button mappings (Xbox/Generic)
    BTN_A = 0           # Cross (PS)
    BTN_B = 1           # Circle (PS)
    BTN_X = 2           # Square (PS)
    BTN_Y = 3           # Triangle (PS)
    BTN_LB = 4          # L1 (PS)
    BTN_RB = 5          # R1 (PS)
    BTN_BACK = 6        # Select/Share
    BTN_START = 7       # Start/Options
    BTN_LS = 8          # L3
    BTN_RS = 9          # R3
    
    # Axis mappings
    AXIS_LX = 0         # Left stick X
    AXIS_LY = 1         # Left stick Y
    AXIS_RX = 2         # Right stick X (or trigger on some controllers)
    AXIS_RY = 3         # Right stick Y
    AXIS_LT = 4         # Left trigger
    AXIS_RT = 5         # Right trigger
    
    # Deadzone for analog sticks
    DEADZONE = 0.15
    
    def __init__(self):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is required for gamepad control")
        
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        # Find controller
        self.joystick = None
        self._init_controller()
        
        # Gimbal components
        self.servo = ServoDriver()
        self.imu = IMUSensor()
        self.stabilizer = Stabilizer()
        self.framer = AutoFramer()
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Control state
        self.running = False
        self.manual_mode = False
        self.stabilization_enabled = True
        self.tracking_enabled = False
        
        # Current position
        self.current_pitch = 90.0
        self.current_yaw = 90.0
        
        # Speed multiplier
        self.speed = 1.0
        self.speed_levels = [0.3, 0.5, 1.0, 1.5, 2.0]
        self.speed_index = 2
        
        # Button debounce
        self.last_button_time = {}
        self.debounce_ms = 300
        
        # Start IMU
        self.imu.start()
        
        logger.info("Gamepad controller initialized")
    
    def _init_controller(self):
        """Initialize gamepad connection."""
        controller_count = pygame.joystick.get_count()
        
        if controller_count == 0:
            logger.error("No gamepad detected! Please connect a controller.")
            raise RuntimeError("No gamepad found")
        
        # Use first controller
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        name = self.joystick.get_name()
        axes = self.joystick.get_numaxes()
        buttons = self.joystick.get_numbuttons()
        
        logger.info(f"Connected: {name}")
        logger.info(f"Axes: {axes}, Buttons: {buttons}")
        
        # Auto-detect controller type for mapping adjustments
        if 'xbox' in name.lower() or 'x-box' in name.lower():
            self.controller_type = 'xbox'
        elif 'playstation' in name.lower() or 'dualshock' in name.lower() or 'dualsense' in name.lower():
            self.controller_type = 'playstation'
        else:
            self.controller_type = 'generic'
        
        logger.info(f"Controller type: {self.controller_type}")
    
    def _apply_deadzone(self, value):
        """Apply deadzone to analog input."""
        if abs(value) < self.DEADZONE:
            return 0.0
        # Rescale after deadzone
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.DEADZONE) / (1 - self.DEADZONE)
    
    def _is_button_pressed(self, button_id):
        """Check if button is pressed with debounce."""
        if not self.joystick.get_button(button_id):
            return False
        
        now = time.time() * 1000
        if button_id in self.last_button_time:
            if now - self.last_button_time[button_id] < self.debounce_ms:
                return False
        
        self.last_button_time[button_id] = now
        return True
    
    def start(self):
        """Start the gamepad control loop."""
        self.running = True
        
        print("\n" + "="*50)
        print("GAMEPAD CONTROLS")
        print("="*50)
        print("Left Stick     - Pan/Tilt")
        print("Right Stick    - Fine adjustment")
        print("A/Cross        - Capture photo")
        print("B/Circle       - Center gimbal")
        print("X/Square       - Toggle stabilization")
        print("Y/Triangle     - Toggle tracking")
        print("LB/L1          - Decrease speed")
        print("RB/R1          - Increase speed")
        print("Start          - Exit")
        print("="*50 + "\n")
        
        print("Press START to exit\n")
        
        # Center on start
        self.servo.set_position(90, 90)
        
        # Main loop
        last_time = time.time()
        
        while self.running:
            # Calculate dt
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Process events
            pygame.event.pump()
            
            # Check for exit
            if self._is_button_pressed(self.BTN_START):
                logger.info("Exit requested")
                break
            
            # Handle buttons
            self._handle_buttons()
            
            # Handle analog sticks
            self._handle_analog(dt)
            
            # Small delay to prevent CPU spinning
            time.sleep(0.01)
        
        self.shutdown()
    
    def _handle_buttons(self):
        """Handle button presses."""
        # Capture photo
        if self._is_button_pressed(self.BTN_A):
            self._capture_photo()
        
        # Center gimbal
        if self._is_button_pressed(self.BTN_B):
            self._center_gimbal()
        
        # Toggle stabilization
        if self._is_button_pressed(self.BTN_X):
            self.stabilization_enabled = not self.stabilization_enabled
            logger.info(f"Stabilization: {'ON' if self.stabilization_enabled else 'OFF'}")
        
        # Toggle tracking
        if self._is_button_pressed(self.BTN_Y):
            self.tracking_enabled = not self.tracking_enabled
            self.manual_mode = False
            logger.info(f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}")
        
        # Decrease speed
        if self._is_button_pressed(self.BTN_LB):
            if self.speed_index > 0:
                self.speed_index -= 1
                self.speed = self.speed_levels[self.speed_index]
                logger.info(f"Speed: {self.speed}x")
        
        # Increase speed
        if self._is_button_pressed(self.BTN_RB):
            if self.speed_index < len(self.speed_levels) - 1:
                self.speed_index += 1
                self.speed = self.speed_levels[self.speed_index]
                logger.info(f"Speed: {self.speed}x")
    
    def _handle_analog(self, dt):
        """Handle analog stick inputs."""
        # Read left stick (primary control)
        lx = self._apply_deadzone(self.joystick.get_axis(self.AXIS_LX))
        ly = self._apply_deadzone(self.joystick.get_axis(self.AXIS_LY))
        
        # Read right stick (fine adjustment)
        rx = self._apply_deadzone(self.joystick.get_axis(self.AXIS_RX))
        ry = self._apply_deadzone(self.joystick.get_axis(self.AXIS_RY))
        
        # Combine inputs (right stick has less influence)
        x = lx + rx * 0.3
        y = ly + ry * 0.3
        
        # If there's input, switch to manual mode
        if abs(x) > 0.01 or abs(y) > 0.01:
            self.manual_mode = True
            self.tracking_enabled = False
            
            # Calculate target velocities (deg/s)
            max_speed = 90.0  # Max degrees per second
            yaw_vel = x * max_speed * self.speed
            pitch_vel = -y * max_speed * self.speed  # Inverted Y
            
            # Update positions
            self.current_yaw += yaw_vel * dt
            self.current_pitch += pitch_vel * dt
            
            # Clamp
            self.current_yaw = np.clip(self.current_yaw, 0, 180)
            self.current_pitch = np.clip(self.current_pitch, 0, 180)
            
            # Send to servos
            self.servo.set_position(self.current_pitch, self.current_yaw)
            
        elif self.manual_mode and not self.tracking_enabled:
            # Return to auto mode after a delay
            # For now, just stay at current position
            pass
    
    def _capture_photo(self):
        """Capture a photo."""
        import os
        from datetime import datetime
        
        os.makedirs('./photos', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./photos/gamepad_capture_{timestamp}.jpg"
        
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(filename, frame)
            logger.info(f"Photo captured: {filename}")
        else:
            logger.error("Failed to capture photo")
    
    def _center_gimbal(self):
        """Center the gimbal."""
        logger.info("Centering gimbal")
        self.current_pitch = 90.0
        self.current_yaw = 90.0
        self.servo.set_position_smooth(90, 90, duration=0.5)
        self.manual_mode = False
    
    def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self.running = False
        self.imu.stop()
        self.servo.center()
        time.sleep(0.3)
        self.servo.disable()
        self.cap.release()
        pygame.quit()
        logger.info("Shutdown complete")


class GamepadWithPreview(GamepadController):
    """Gamepad controller with video preview window."""
    
    def __init__(self):
        super().__init__()
        self.show_preview = True
    
    def start(self):
        """Start with video preview."""
        import cv2
        
        self.running = True
        
        print("\n" + "="*50)
        print("GAMEPAD CONTROLS (with preview)")
        print("="*50)
        print("See previous output for controls")
        print("="*50 + "\n")
        
        # Center on start
        self.servo.set_position(90, 90)
        
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Process pygame events
            pygame.event.pump()
            
            # Check exit
            if self._is_button_pressed(self.BTN_START):
                break
            
            # Handle controls
            self._handle_buttons()
            self._handle_analog(dt)
            
            # Get and display frame
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                
                # Draw status overlay
                self._draw_overlay(frame)
                
                cv2.imshow("Gamepad Control", frame)
                
                # Also check for 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.001)
        
        cv2.destroyAllWindows()
        self.shutdown()
    
    def _draw_overlay(self, frame):
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]
        
        # Status bar background
        cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 0), -1)
        
        # Build status text
        mode = "MANUAL" if self.manual_mode else ("TRACKING" if self.tracking_enabled else "CENTER")
        status = f"Mode: {mode} | Speed: {self.speed}x | Stab: {'ON' if self.stabilization_enabled else 'OFF'}"
        
        cv2.putText(frame, status, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw crosshair at center
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)


def main():
    """Main entry point."""
    try:
        # Try with preview
        controller = GamepadWithPreview()
        controller.start()
    except Exception as e:
        logger.error(f"Error: {e}")
        # Fallback to no preview
        try:
            controller = GamepadController()
            controller.start()
        except Exception as e2:
            logger.error(f"Fatal error: {e2}")
            raise


if __name__ == '__main__':
    main()
