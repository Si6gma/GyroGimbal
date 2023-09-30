# LaserAI

Work In Progress

This project uses OpenCV to detect faces in a webcam feed, and two servos controlled by an Arduino to move a laser pointer to follow the detected face.

## Components

- A webcam for capturing video feed.
- An Arduino board for controlling the servos.
- Two servos for moving the laser pointer in two axes.
- A laser pointer.

## Software Dependencies

- Python 3
- OpenCV Python library
- PySerial library for Python-Arduino communication
- Arduino IDE

## How It Works

1. The Python script uses OpenCV to capture video from the webcam and detect faces in each frame.
2. The coordinates of the detected face are transformed from the webcam frame's coordinate system to the servo's coordinate system.
3. The transformed coordinates are sent to the Arduino via serial communication.
4. The Arduino moves the servos to point the laser at the received coordinates.
