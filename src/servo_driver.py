#!/usr/bin/env python3
"""
Servo Driver - PCA9685 PWM Controller

Controls pitch and yaw servos using the PCA9685 16-channel PWM driver
connected via I2C to the Raspberry Pi.

Hardware:
    - PCA9685 PWM Driver (I2C address 0x40 default)
    - 2x MG996R or DS3218 servos (high torque for camera gimbal)
    - External 5V power supply for servos (separate from Pi power)

Wiring:
    PCA9685 VCC -> Pi 3.3V or 5V
    PCA9685 GND -> Pi GND
    PCA9685 SDA -> Pi GPIO 2 (SDA)
    PCA9685 SCL -> Pi GPIO 3 (SCL)
    PCA9685 V+  -> External 5V power supply
    PCA9685 GND -> External power supply GND (common ground)
    
    Servo Pitch -> PCA9685 Channel 0
    Servo Yaw   -> PCA9685 Channel 1
"""

import logging
import time
from typing import Tuple, Optional

try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    logging.warning("Adafruit libraries not available. Running in simulation mode.")

logger = logging.getLogger(__name__)

# Servo pulse width limits (in microseconds)
# Adjust these based on your specific servos
SERVO_MIN_PULSE = 500   # 0 degrees
SERVO_MAX_PULSE = 2500  # 180 degrees
SERVO_FREQ = 50         # 50Hz standard for servos


class ServoDriver:
    """
    PCA9685-based servo controller for 2-axis gimbal.
    
    Features:
    - Smooth motion with configurable speed limits
    - Position feedback (for servos with feedback wire)
    - Emergency stop
    """
    
    def __init__(
        self,
        pitch_channel: int = 0,
        yaw_channel: int = 1,
        pitch_range: Tuple[int, int] = (0, 180),
        yaw_range: Tuple[int, int] = (0, 180),
        i2c_address: int = 0x40
    ):
        self.pitch_range = pitch_range
        self.yaw_range = yaw_range
        self.current_pitch = 90.0
        self.current_yaw = 90.0
        
        if not HARDWARE_AVAILABLE:
            logger.warning("Running in SIMULATION mode - no actual servo control")
            self.pca = None
            self.pitch_servo = None
            self.yaw_servo = None
            return
        
        # Initialize I2C and PCA9685
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c, address=i2c_address)
            self.pca.frequency = SERVO_FREQ
            
            # Initialize servos
            self.pitch_servo = servo.Servo(
                self.pca.channels[pitch_channel],
                min_pulse=SERVO_MIN_PULSE,
                max_pulse=SERVO_MAX_PULSE
            )
            self.yaw_servo = servo.Servo(
                self.pca.channels[yaw_channel],
                min_pulse=SERVO_MIN_PULSE,
                max_pulse=SERVO_MAX_PULSE
            )
            
            logger.info(f"PCA9685 initialized at address 0x{i2c_address:02X}")
            logger.info(f"Pitch servo on channel {pitch_channel}")
            logger.info(f"Yaw servo on channel {yaw_channel}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PCA9685: {e}")
            raise
    
    def set_position(self, pitch: float, yaw: float):
        """
        Set servo positions in degrees.
        
        Args:
            pitch: Pitch angle (0-180 degrees)
            yaw: Yaw angle (0-180 degrees)
        """
        # Clamp to limits
        pitch = max(self.pitch_range[0], min(self.pitch_range[1], pitch))
        yaw = max(self.yaw_range[0], min(self.yaw_range[1], yaw))
        
        self.current_pitch = pitch
        self.current_yaw = yaw
        
        if HARDWARE_AVAILABLE and self.pitch_servo and self.yaw_servo:
            self.pitch_servo.angle = pitch
            self.yaw_servo.angle = yaw
        else:
            # Simulation mode - just log
            pass  # logger.debug(f"Simulated: pitch={pitch:.1f}, yaw={yaw:.1f}")
    
    def set_position_smooth(
        self, 
        pitch: float, 
        yaw: float, 
        duration: float = 0.5
    ):
        """
        Move servos smoothly to target position over specified duration.
        
        Args:
            pitch: Target pitch angle
            yaw: Target yaw angle
            duration: Time to complete movement (seconds)
        """
        start_pitch = self.current_pitch
        start_yaw = self.current_yaw
        
        steps = int(duration * 60)  # 60 updates per second
        
        for i in range(steps + 1):
            t = i / steps
            # Easing function (ease-in-out cubic)
            t = t * t * (3 - 2 * t)
            
            p = start_pitch + (pitch - start_pitch) * t
            y = start_yaw + (yaw - start_yaw) * t
            
            self.set_position(p, y)
            time.sleep(duration / steps)
    
    def center(self):
        """Move servos to center position (90, 90)."""
        logger.info("Centering servos")
        self.set_position_smooth(90, 90, duration=0.5)
    
    def disable(self):
        """Disable PWM output (servos go limp)."""
        if HARDWARE_AVAILABLE and self.pca:
            # Set duty cycle to 0
            self.pca.channels[0].duty_cycle = 0
            self.pca.channels[1].duty_cycle = 0
            logger.info("Servos disabled")
    
    def emergency_stop(self):
        """Immediate stop - disable servos."""
        logger.warning("EMERGENCY STOP triggered")
        self.disable()
    
    def get_position(self) -> Tuple[float, float]:
        """Get current servo positions."""
        return (self.current_pitch, self.current_yaw)
    
    def __del__(self):
        """Cleanup on destruction."""
        if HARDWARE_AVAILABLE and self.pca:
            self.pca.deinit()
