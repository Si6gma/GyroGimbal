# LaserTurret

**Work In Progress**

This project uses OpenCV to detect faces in a webcam feed, and two servos(motors) controlled by an Arduino to move a laser pointer to follow the detected face.

_Basically it points a laser into someone's eye_

## Coordination System (WIP)

The two servos control the pitch (x-axis), and yaw (y-axis)

Using arctan (tan inverse), we can figure out the two angles `α - pitch, θ - yaw` to control the laser.

The laser is at the origin for all the diagrams below:
![Coordination Plan](/Coordination%20Plan.png)

`center_x` and `center_y` in `facialRecog.py` are the x and y coordinates

`Z coordinate` is the distance between the object and the laser, which is not measurable due to technical limitations, therefore will require manual calibration beforehand.

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

- Enable the Virtual Environment by inputting the following command
  _On Windows_:

```
.\env\Scripts\activate
```

_On macOS/Linus_:

```
source env/bin/activate
```

## How It Works

1. The Python script uses OpenCV to capture video from the webcam and detect faces in each frame.
2. The coordinates of the detected face are transformed from the webcam frame's coordinate system to the servo's coordinate system.
3. The transformed coordinates are sent to the Arduino via serial communication.
4. The Arduino moves the servos to point the laser at the received coordinates.
