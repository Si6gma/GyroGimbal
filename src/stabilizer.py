#!/usr/bin/env python3
"""
Stabilizer Module - Sensor Fusion and Compensation

Implements sensor fusion algorithms to combine IMU data with visual tracking
to produce smooth, stabilized camera movements.

Features:
    - Complementary filter for orientation estimation
    - PID controller for stabilization
    - Motion prediction for subject tracking
    - Jerk limiting for smooth servo movements
"""

import numpy as np
import logging
import time
from typing import Tuple, Optional
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PIDConfig:
    """PID controller configuration."""
    kp: float = 1.0  # Proportional gain
    ki: float = 0.1  # Integral gain
    kd: float = 0.5  # Derivative gain
    integral_limit: float = 10.0  # Anti-windup
    output_limit: float = 50.0    # Maximum output


class PIDController:
    """Simple PID controller with anti-windup."""
    
    def __init__(self, config: PIDConfig):
        self.config = config
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
    
    def update(self, error: float, dt: float) -> float:
        """
        Update PID controller.
        
        Args:
            error: Current error (setpoint - measured)
            dt: Time step in seconds
            
        Returns:
            Control output
        """
        if dt <= 0:
            return 0.0
        
        # Proportional term
        p = self.config.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(
            self.integral, 
            -self.config.integral_limit, 
            self.config.integral_limit
        )
        i = self.config.ki * self.integral
        
        # Derivative term
        d = self.config.kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        # Calculate output
        output = p + i + d
        output = np.clip(output, -self.config.output_limit, self.config.output_limit)
        
        return output


class Stabilizer:
    """
    Main stabilization engine.
    
    Combines IMU data with optional visual feedback to calculate
    compensation angles for the gimbal.
    """
    
    def __init__(
        self,
        gain: float = 0.7,
        smoothing: float = 0.3,
        use_complementary_filter: bool = True
    ):
        self.gain = gain  # How aggressively to compensate
        self.smoothing = smoothing
        self.use_complementary_filter = use_complementary_filter
        
        # Orientation estimation
        self.roll = 0.0   # Around X axis (side tilt)
        self.pitch = 0.0  # Around Y axis (forward/back tilt)
        self.yaw = 0.0    # Around Z axis (rotation)
        
        # Complementary filter coefficient
        # Higher = trust gyro more, lower = trust accel more
        self.alpha = 0.98
        
        # PID controllers for each axis
        self.pitch_pid = PIDController(PIDConfig(
            kp=2.0, ki=0.2, kd=1.0, output_limit=100
        ))
        self.yaw_pid = PIDController(PIDConfig(
            kp=2.0, ki=0.2, kd=1.0, output_limit=100
        ))
        
        # History for smoothing
        self.pitch_history = deque(maxlen=10)
        self.yaw_history = deque(maxlen=10)
        
        # Previous values for rate calculation
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_time = None
        
        logger.info("Stabilizer initialized")
    
    def calculate_compensation(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        dt: float
    ) -> Tuple[float, float]:
        """
        Calculate stabilization compensation based on IMU data.
        
        Args:
            gyro: Gyroscope readings [x, y, z] in rad/s
            accel: Accelerometer readings [x, y, z] in g
            dt: Time step in seconds
            
        Returns:
            Tuple of (pitch_compensation, yaw_compensation) in deg/s
        """
        if dt <= 0:
            return (0.0, 0.0)
        
        # Extract gyro rates
        gyro_x, gyro_y, gyro_z = gyro
        
        # Calculate angles from accelerometer
        accel_pitch = np.arctan2(accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
        accel_roll = np.arctan2(accel[1], accel[2])
        
        # Complementary filter for orientation
        if self.use_complementary_filter:
            # Integrate gyro
            self.pitch += gyro_y * dt
            self.roll += gyro_x * dt
            self.yaw += gyro_z * dt
            
            # Correct drift with accelerometer (only for pitch/roll)
            self.pitch = self.alpha * self.pitch + (1 - self.alpha) * accel_pitch
            self.roll = self.alpha * self.roll + (1 - self.alpha) * accel_roll
        
        # Calculate desired compensation
        # We want to move opposite to the detected rotation
        # to keep the camera steady
        
        # Method 1: Direct rate compensation
        pitch_rate_comp = -gyro_y * self.gain  # rad/s
        yaw_rate_comp = -gyro_z * self.gain    # rad/s
        
        # Convert to deg/s
        pitch_rate_comp = np.degrees(pitch_rate_comp)
        yaw_rate_comp = np.degrees(yaw_rate_comp)
        
        # Method 2: Add PID control on position error (if we have a reference)
        # For now, we just use rate compensation
        
        # Smooth the output
        self.pitch_history.append(pitch_rate_comp)
        self.yaw_history.append(yaw_rate_comp)
        
        smoothed_pitch = np.mean(self.pitch_history)
        smoothed_yaw = np.mean(self.yaw_history)
        
        return (smoothed_pitch, smoothed_yaw)
    
    def calculate_tracking_compensation(
        self,
        subject_position: Tuple[float, float],
        frame_center: Tuple[float, float],
        dt: float
    ) -> Tuple[float, float]:
        """
        Calculate compensation to keep subject centered.
        
        Args:
            subject_position: (x, y) pixel coordinates of subject
            frame_center: (x, y) pixel coordinates of frame center
            dt: Time step
            
        Returns:
            Tuple of (pitch_adjustment, yaw_adjustment) in degrees
        """
        # Calculate error in pixels
        error_x = subject_position[0] - frame_center[0]
        error_y = subject_position[1] - frame_center[1]
        
        # Convert pixel error to angular error
        # Assuming camera FOV is known (e.g., 60° horizontal)
        fov_horizontal = 60.0  # degrees
        fov_vertical = 45.0    # degrees
        
        # Scale factors (degrees per pixel)
        scale_x = fov_horizontal / 1280  # Assuming 1280px width
        scale_y = fov_vertical / 720     # Assuming 720px height
        
        yaw_error = error_x * scale_x
        pitch_error = error_y * scale_y
        
        # PID control
        yaw_adjustment = self.yaw_pid.update(yaw_error, dt)
        pitch_adjustment = self.pitch_pid.update(pitch_error, dt)
        
        return (pitch_adjustment, yaw_adjustment)
    
    def blend_tracking_stabilization(
        self,
        tracking_angles: Tuple[float, float],
        stabilization_angles: Tuple[float, float],
        tracking_weight: float = 0.5
    ) -> Tuple[float, float]:
        """
        Blend tracking and stabilization commands.
        
        Args:
            tracking_angles: (pitch, yaw) from subject tracking
            stabilization_angles: (pitch, yaw) from IMU stabilization
            tracking_weight: How much to prioritize tracking (0-1)
                             1.0 = only tracking, 0.0 = only stabilization
                             
        Returns:
            Blended (pitch, yaw) angles
        """
        stab_weight = 1.0 - tracking_weight
        
        # For stabilization, we add the compensation
        # For tracking, we set the target position
        # This is a simplified blend - in practice you'd want more sophisticated fusion
        
        pitch = tracking_angles[0] + stabilization_angles[0] * stab_weight
        yaw = tracking_angles[1] + stabilization_angles[1] * stab_weight
        
        return (pitch, yaw)
    
    def apply_jerk_limiting(
        self,
        target_pitch: float,
        target_yaw: float,
        current_pitch: float,
        current_yaw: float,
        dt: float,
        max_jerk: float = 500.0  # deg/s^3
    ) -> Tuple[float, float]:
        """
        Apply jerk limiting for smooth servo movements.
        
        Jerk is the rate of change of acceleration. Limiting it
        prevents sudden, jarring movements.
        """
        if dt <= 0:
            return (current_pitch, current_yaw)
        
        # Calculate desired velocities
        desired_pitch_vel = (target_pitch - current_pitch) / dt
        desired_yaw_vel = (target_yaw - current_yaw) / dt
        
        # Calculate previous velocities (simplified - would need history)
        if self.prev_time is None:
            self.prev_pitch = current_pitch
            self.prev_yaw = current_yaw
            self.prev_time = time.time() - dt
        
        prev_pitch_vel = (current_pitch - self.prev_pitch) / dt
        prev_yaw_vel = (current_yaw - self.prev_yaw) / dt
        
        # Calculate acceleration
        pitch_accel = (desired_pitch_vel - prev_pitch_vel) / dt
        yaw_accel = (desired_yaw_vel - prev_yaw_vel) / dt
        
        # Limit jerk (change in acceleration)
        max_accel_change = max_jerk * dt
        pitch_accel = np.clip(pitch_accel, -max_accel_change, max_accel_change)
        yaw_accel = np.clip(yaw_accel, -max_accel_change, max_accel_change)
        
        # Calculate limited velocity
        limited_pitch_vel = prev_pitch_vel + pitch_accel * dt
        limited_yaw_vel = prev_yaw_vel + yaw_accel * dt
        
        # Update stored values
        self.prev_pitch = current_pitch
        self.prev_yaw = current_yaw
        self.prev_time = time.time()
        
        # Calculate new positions
        new_pitch = current_pitch + limited_pitch_vel * dt
        new_yaw = current_yaw + limited_yaw_vel * dt
        
        return (new_pitch, new_yaw)
    
    def reset(self):
        """Reset all internal states."""
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.pitch_history.clear()
        self.yaw_history.clear()
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_time = None
