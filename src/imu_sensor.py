#!/usr/bin/env python3
"""
IMU Sensor Module - MPU6050/MPU9250 Interface

Reads accelerometer and gyroscope data for stabilization.
Supports both MPU6050 (6-DOF) and MPU9250 (9-DOF with magnetometer).

Hardware:
    - MPU6050 or MPU9250 (I2C)
    - Connected to Raspberry Pi I2C bus

Wiring:
    VCC -> Pi 3.3V or 5V (check sensor specs)
    GND -> Pi GND
    SDA -> Pi GPIO 2 (SDA)
    SCL -> Pi GPIO 3 (SCL)
    INT -> Pi GPIO (optional, for interrupt-driven sampling)
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque
import numpy as np

try:
    import board
    import busio
    import adafruit_mpu6050
    # Verify we have actual hardware pins available
    HARDWARE_AVAILABLE = hasattr(board, 'SCL') and hasattr(board, 'SDA')
    if not HARDWARE_AVAILABLE:
        logging.warning("Board pins not available. Running in simulation mode.")
except ImportError:
    HARDWARE_AVAILABLE = False
    logging.warning("IMU libraries not available. Running in simulation mode.")

logger = logging.getLogger(__name__)

# Default I2C address
MPU6050_ADDR = 0x68
MPU9250_ADDR = 0x68

# Sensor ranges
ACCEL_RANGE_2G = 0
ACCEL_RANGE_4G = 1
ACCEL_RANGE_8G = 2
GYRO_RANGE_250DPS = 0
GYRO_RANGE_500DPS = 1
GYRO_RANGE_1000DPS = 2


@dataclass
class IMUData:
    """Container for IMU sensor readings."""
    accel: np.ndarray = None  # m/s^2
    gyro: np.ndarray = None   # rad/s
    mag: Optional[np.ndarray] = None  # uT (MPU9250 only)
    temperature: float = 0.0         # Celsius
    timestamp: float = 0.0
    valid: bool = False
    
    def __post_init__(self):
        if self.accel is None:
            self.accel = np.zeros(3)
        if self.gyro is None:
            self.gyro = np.zeros(3)


class IMUSensor:
    """
    MPU6050/MPU9250 IMU interface with background sampling.
    
    Features:
    - Background thread for high-frequency sampling
    - Built-in filtering (low-pass)
    - Gyro calibration on startup
    """
    
    def __init__(
        self,
        i2c_address: int = MPU6050_ADDR,
        sample_rate: int = 100,  # Hz
        accel_range: int = ACCEL_RANGE_2G,
        gyro_range: int = GYRO_RANGE_500DPS
    ):
        self.i2c_address = i2c_address
        self.sample_rate = sample_rate
        self.sample_period = 1.0 / sample_rate
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_data = IMUData()
        
        # Calibration offsets
        self._gyro_offset = np.zeros(3)
        self._accel_offset = np.zeros(3)
        
        # Filtering
        self._accel_filter_alpha = 0.2
        self._gyro_filter_alpha = 0.3
        self._filtered_accel = np.zeros(3)
        self._filtered_gyro = np.zeros(3)
        
        if not HARDWARE_AVAILABLE:
            logger.warning("IMU running in SIMULATION mode")
            self._sensor = None
            return
        
        # Initialize hardware
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = adafruit_mpu6050.MPU6050(i2c, address=i2c_address)
            
            # Configure ranges
            # Note: adafruit_mpu6050 library handles this differently
            # These settings may need adjustment based on library version
            
            logger.info(f"MPU6050 initialized at 0x{i2c_address:02X}")
            logger.info(f"Accelerometer range: ±2G")
            logger.info(f"Gyroscope range: ±500°/s")
            
        except Exception as e:
            logger.error(f"Failed to initialize IMU: {e}")
            self._sensor = None
    
    def start(self):
        """Start background sampling thread."""
        if self._running:
            return
        
        if HARDWARE_AVAILABLE and self._sensor:
            # Calibrate gyro
            self._calibrate_gyro()
        
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info(f"IMU sampling started at {self.sample_rate}Hz")
    
    def stop(self):
        """Stop background sampling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("IMU sampling stopped")
    
    def _calibrate_gyro(self, samples: int = 500):
        """
        Calibrate gyroscope by averaging stationary readings.
        Must be called with sensor stationary!
        """
        if not HARDWARE_AVAILABLE or not self._sensor:
            return
        
        logger.info("Calibrating gyroscope... keep sensor still!")
        time.sleep(1)
        
        gyro_sum = np.zeros(3)
        
        for _ in range(samples):
            gyro = self._read_raw_gyro()
            gyro_sum += gyro
            time.sleep(0.001)
        
        self._gyro_offset = gyro_sum / samples
        logger.info(f"Gyro offset: {self._gyro_offset}")
    
    def _read_raw_accel(self) -> np.ndarray:
        """Read raw accelerometer data."""
        if not HARDWARE_AVAILABLE or not self._sensor:
            return np.zeros(3)
        
        # Convert from m/s^2 to g, then to array
        accel = np.array([
            self._sensor.acceleration[0] / 9.80665,
            self._sensor.acceleration[1] / 9.80665,
            self._sensor.acceleration[2] / 9.80665
        ])
        return accel
    
    def _read_raw_gyro(self) -> np.ndarray:
        """Read raw gyroscope data in rad/s."""
        if not HARDWARE_AVAILABLE or not self._sensor:
            return np.zeros(3)
        
        # Convert from deg/s to rad/s
        gyro = np.array([
            np.radians(self._sensor.gyro[0]),
            np.radians(self._sensor.gyro[1]),
            np.radians(self._sensor.gyro[2])
        ])
        return gyro
    
    def _sample_loop(self):
        """Background sampling loop."""
        while self._running:
            loop_start = time.time()
            
            # Read sensor
            accel = self._read_raw_accel()
            gyro = self._read_raw_gyro()
            
            # Apply calibration
            gyro -= self._gyro_offset
            
            # Apply low-pass filtering
            self._filtered_accel = (
                self._accel_filter_alpha * accel + 
                (1 - self._accel_filter_alpha) * self._filtered_accel
            )
            self._filtered_gyro = (
                self._gyro_filter_alpha * gyro + 
                (1 - self._gyro_filter_alpha) * self._filtered_gyro
            )
            
            # Create data packet
            data = IMUData(
                accel=self._filtered_accel.copy(),
                gyro=self._filtered_gyro.copy(),
                temperature=self._read_temperature(),
                timestamp=time.time(),
                valid=True
            )
            
            # Update latest data (thread-safe)
            with self._lock:
                self._latest_data = data
            
            # Maintain sample rate
            elapsed = time.time() - loop_start
            sleep_time = self.sample_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _read_temperature(self) -> float:
        """Read sensor temperature."""
        if not HARDWARE_AVAILABLE or not self._sensor:
            return 0.0
        try:
            return self._sensor.temperature
        except:
            return 0.0
    
    def get_reading(self) -> IMUData:
        """Get the latest IMU reading (thread-safe)."""
        with self._lock:
            return self._latest_data
    
    def get_angles(self) -> Tuple[float, float, float]:
        """
        Estimate roll and pitch from accelerometer.
        Note: This is only valid when sensor is not accelerating!
        """
        data = self.get_reading()
        if not data.valid:
            return (0.0, 0.0, 0.0)
        
        ax, ay, az = data.accel
        
        # Roll (rotation around X axis)
        roll = np.arctan2(ay, az)
        
        # Pitch (rotation around Y axis)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        
        # Convert to degrees
        return (
            np.degrees(roll),
            np.degrees(pitch),
            0.0  # Yaw can't be determined from accel alone
        )
    
    def simulate_motion(self, pitch_rate: float, yaw_rate: float):
        """
        For testing without hardware - simulate rotational motion.
        
        Args:
            pitch_rate: Pitch angular velocity in rad/s
            yaw_rate: Yaw angular velocity in rad/s
        """
        if HARDWARE_AVAILABLE:
            return
        
        data = IMUData(
            accel=np.array([0, 0, 1.0]),
            gyro=np.array([0, pitch_rate, yaw_rate]),
            timestamp=time.time(),
            valid=True
        )
        
        with self._lock:
            self._latest_data = data
