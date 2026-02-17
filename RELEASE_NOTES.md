# Pi Gimbal Stabilizer v1.0.0

A 2-axis camera gimbal stabilizer with IMU-based stabilization and auto-framing, running entirely on Raspberry Pi.

## Features

- IMU-Based Stabilization — Uses MPU6050/MPU9250 to compensate for camera shake in real-time
- Auto-Framing — Detects faces/bodies and keeps subjects centered with proper composition
- Web Control Interface — Control from any device on your network with live video stream
- Gamepad Support — Use Xbox/PlayStation controllers for smooth manual control
- Photo Capture — Save perfectly framed photos with single click/button
- 100+ Unit Tests — Full test coverage with CI/CD pipeline (Python 3.9/3.10/3.11)

## Quick Start

```bash
git clone https://github.com/Si6gma/GyroGimbal.git
cd GyroGimbal
pip install -r requirements.txt

# Run web interface
cd src && sudo python3 web_server.py
```

Then open `http://[pi-ip]:5000` in your browser.

## Hardware Requirements

- Raspberry Pi 3B+ or 4
- Pi Camera V2 or USB webcam
- PCA9685 servo driver (I2C)
- 2x MG996R or DS3218 high-torque servos
- MPU6050 or MPU9250 IMU (I2C)
- 5V 4A external power supply for servos

## What's Included

- ~2,800 lines of Python code
- 100 unit tests across 6 test files
- CI/CD pipeline testing Python 3.9/3.10/3.11
- Full documentation and troubleshooting guide

## Documentation

See [README.md](https://github.com/Si6gma/GyroGimbal/blob/main/README.md) for complete setup instructions, wiring diagrams, and tuning guide.

## License

MIT License
