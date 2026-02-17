# Pi Gimbal Stabilizer

A 2-axis camera gimbal stabilizer with auto-framing and IMU-based stabilization, running entirely on Raspberry Pi.

## Features

- **IMU-Based Stabilization**: Uses MPU6050/MPU9250 gyroscope and accelerometer to compensate for camera shake
- **Auto-Framing**: Detects people and automatically keeps them centered with proper composition
- **Real-Time Subject Tracking**: Face and body detection with smooth servo following
- **Photo Capture**: Capture perfectly framed photos with single keypress
- **2-Axis Control**: Pitch and yaw servo control via PCA9685 PWM driver
- **Web Control Interface**: Control from any device on your network with live video stream
- **Gamepad Support**: Use Xbox/PlayStation controllers for smooth manual control
- **All-in-One Raspberry Pi**: No Arduino required - everything runs on the Pi

## Hardware Requirements

| Component | Specification | Purpose |
|-----------|--------------|---------|
| Raspberry Pi | 3B+ or 4 (4 recommended) | Main controller |
| Camera | Pi Camera Module V2 or USB webcam | Video capture |
| Servo Driver | PCA9685 16-channel PWM (I2C) | Servo control |
| Servos | 2x MG996R or DS3218 (high torque) | Gimbal movement |
| IMU | MPU6050 or MPU9250 (I2C) | Motion sensing |
| Power Supply | 5V 4A (separate from Pi) | Servo power |
| Gimbal Frame | 3D printed or purchased | Camera mount |
| Gamepad (optional) | Xbox, PlayStation, or USB | Manual control |

## Wiring

```
PCA9685 (Servo Driver):
    VCC  -> Pi 3.3V or 5V
    GND  -> Pi GND
    SDA  -> Pi GPIO 2 (Pin 3)
    SCL  -> Pi GPIO 3 (Pin 5)
    V+   -> External 5V 4A power supply
    GND  -> External PSU GND (common with Pi)
    
    Servo Pitch -> Channel 0
    Servo Yaw   -> Channel 1

MPU6050/MPU9250 (IMU):
    VCC  -> Pi 3.3V
    GND  -> Pi GND
    SDA  -> Pi GPIO 2 (Pin 3) [shared with PCA9685]
    SCL  -> Pi GPIO 3 (Pin 5) [shared with PCA9685]
    
Pi Camera:
    Connect to CSI port on Pi

OR USB Camera:
    Connect to USB port

Gamepad (optional):
    Connect via USB or Bluetooth
```

## Installation

### 1. Enable I2C on Raspberry Pi

```bash
sudo raspi-config
# Navigate to: Interface Options -> I2C -> Enable
sudo reboot
```

Verify I2C is working:
```bash
sudo apt-get install i2c-tools
sudo i2cdetect -y 1
```
You should see devices at `0x40` (PCA9685) and `0x68` (MPU6050).

### 2. Install Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y python3-pip python3-opencv libcamera-dev

# Clone repository
git clone https://github.com/Si6gma/LaserTurret.git
cd LaserTurret

# Install Python packages
pip3 install -r requirements.txt
```

### 3. Configure Hardware

Edit `config.py` to match your setup:

```python
# If using USB camera instead of Pi Camera
CAMERA_INDEX = 1

# Adjust servo limits if needed
PITCH_MIN = 0
PITCH_MAX = 180

# Fine-tune stabilization
STABILIZATION_GAIN = 0.7  # Increase if footage is still shaky
```

## Usage

### Option 1: Web Control (Recommended)

Control the gimbal from any device on your network:

```bash
cd src
sudo python3 web_server.py
```

Then open a browser on any device and go to:
```
http://[raspberry-pi-ip]:5000
```

Find your Pi's IP:
```bash
hostname -I
```

#### Web Interface Features:
- **Live Video Stream**: MJPEG feed from the camera
- **Virtual Joystick**: Click/touch and drag to pan/tilt manually
- **Mode Toggles**: Turn stabilization and tracking on/off
- **Photo Capture**: Click to save photos
- **Center Button**: Return to center position
- **Mobile Friendly**: Works on phones and tablets

![Web Interface](assets/Dashboard.png)

### Option 2: Gamepad Control

Use an Xbox, PlayStation, or USB gamepad for smooth manual control:

```bash
cd src
sudo python3 gamepad_controller.py
```

#### Gamepad Controls:

| Control | Action |
|---------|--------|
| **Left Stick** | Pan/Tilt gimbal |
| **Right Stick** | Fine adjustment |
| **A / Cross** | Capture photo |
| **B / Circle** | Center gimbal |
| **X / Square** | Toggle stabilization |
| **Y / Triangle** | Toggle tracking |
| **LB / L1** | Decrease speed |
| **RB / R1** | Increase speed |
| **Start** | Exit |

The analog sticks provide much smoother control than keyboard or web joystick.

### Option 3: Standalone (No Interface)

Run the gimbal without any control interface:

```bash
cd src
sudo python3 gimbal_controller.py
```

Controls:
| Key | Action |
|-----|--------|
| `c` | Capture photo |
| `s` | Toggle stabilization |
| `t` | Toggle tracking |
| `q` | Quit |

## Photo Output

Photos are saved to `./photos/` with timestamp:
```
photos/
  capture_20240217_143022.jpg
  web_capture_20240217_143156.jpg
  gamepad_capture_20240217_143220.jpg
```

## How It Works

### Stabilization Algorithm

1. **IMU Sampling**: MPU6050 samples gyroscope and accelerometer at 100Hz
2. **Sensor Fusion**: Complementary filter combines gyro (fast response) and accel (drift-free)
3. **Motion Compensation**: Detected rotation is inverted and applied to servos
4. **Smoothing**: Low-pass filter and jerk limiting prevent servo jitter

### Auto-Framing

1. **Detection**: OpenCV Haar cascades detect faces/bodies
2. **Tracking**: Kalman filter predicts subject movement
3. **Composition**: Calculates optimal gimbal angles using rule of thirds
4. **Smoothing**: Exponential moving average prevents hunting

### Web Interface Architecture

```
┌─────────────┐      WebSocket/HTTP      ┌─────────────┐
│   Browser   │  <-------------------->  │  Flask API  │
│  (Phone/PC) │     MJPEG Stream         │   (Pi)      │
└─────────────┘                          └──────┬──────┘
                                                │
                    ┌─────────────┬─────────────┼─────────────┐
                    │             │             │             │
               ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
               │  Camera │   │  IMU    │   │ PCA9685 │   │  GPIO   │
               │ Module  │   │MPU6050  │   │(Servos) │   │ (LEDs)  │
               └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

### Gamepad Control Flow

```
Gamepad Input → Pygame Events → Position Update → PCA9685 → Servos
     ↑                                                    ↓
     └──────────── Visual Feedback ← Camera ←─────────────┘
```

## Project Structure

```
LaserTurret/
├── src/
│   ├── gimbal_controller.py   # Main standalone application
│   ├── web_server.py           # Flask web interface
│   ├── gamepad_controller.py   # Gamepad control
│   ├── servo_driver.py         # PCA9685 servo control
│   ├── imu_sensor.py           # MPU6050 interface
│   ├── stabilizer.py           # Sensor fusion & PID
│   └── auto_framing.py         # Subject detection & framing
├── config.py                   # Hardware configuration
├── requirements.txt            # Python dependencies
└── README.md
```

## Tuning

### Stabilization Too Aggressive

```python
# config.py
STABILIZATION_GAIN = 0.4  # Reduce compensation
STABILIZATION_SMOOTHING = 0.5  # More smoothing
```

### Tracking Too Slow

```python
# config.py
TRACKING_SMOOTHING = 0.3  # Faster response
```

### Servo Jitter

1. Check power supply - servos need stable 5V
2. Increase smoothing values
3. Check I2C cable length (keep under 30cm)
4. Add capacitors near servo power input

## Troubleshooting

### "No I2C devices found"
```bash
# Check I2C is enabled
sudo raspi-config

# Check wiring - common ground between Pi and servos
# Verify device addresses
sudo i2cdetect -y 1
```

### "Camera not detected"
```bash
# For Pi Camera
sudo raspi-config  # Enable camera interface
# For USB camera
check CAMERA_INDEX in config.py (try 0, 1, 2...)
```

### Servos not moving
1. Check external power supply is connected
2. Verify PCA9685 V+ is powered (not just VCC)
3. Check servo channel numbers in config
4. Test with `servo_driver.py` directly

### Web interface not accessible
```bash
# Check Flask is listening on all interfaces
# In web_server.py, app.run() should have host='0.0.0.0'

# Check firewall
sudo ufw allow 5000

# Check Pi's IP
hostname -I
```

### Gamepad not detected
```bash
# List connected controllers
python3 -c "import pygame; pygame.init(); print([pygame.joystick.Joystick(i).get_name() for i in range(pygame.joystick.get_count())])"

# For Bluetooth controllers, pair first
bluetoothctl
```

### Poor stabilization
1. Calibrate IMU on startup (keep still for 2 seconds)
2. Adjust STABILIZATION_GAIN
3. Check IMU is firmly mounted to camera
4. Reduce mechanical backlash in gimbal

## License

MIT License - See LICENSE file

---

*Built for smooth footage and perfect composition.*
