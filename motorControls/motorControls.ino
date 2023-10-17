#include <Servo.h>
#include "Arduino_LED_Matrix.h"
#include "fonts.h"

Servo myservo;           // create servo object to control a servo
ArduinoLEDMatrix matrix; // create LED Matrix object

byte frame[8][12] = {
    {0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0},
    {0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0},
    {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0},
    {0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

void setup()
{
    myservo.attach(9);    // attaches the servo on pin
    Serial.begin(115200); // starts the serial communication
    matrix.begin();       // starts the LED Matrix

    // Display heart on startup
    display_frame();
}

void loop()
{
    if (Serial.available() > 1)
    {
        String yaw_angle = Serial.readString(); // reads the data from the serial port
        int angle = yaw_angle.toInt();          // convert string to integer
        myservo.write(angle);                   // rotates the servo motor according to the received angle

        // Display angle on LED Matrix
        clear_frame();
        add_to_frame(angle);
        display_frame();
        Serial.println(angle);
    }
}

void clear_frame()
{
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 12; col++)
        {
            frame[row][col] = 0;
        }
    }
}

void display_frame()
{
    matrix.renderBitmap(frame, 8, 12);
}

void add_to_frame(int angle)
{
    String angleStr = String(angle);
    for (int i = angleStr.length() - 1; i >= max(4 - angleStr.length(), -1); i--)
    {
        char c = angleStr.charAt(i);
        int index = c - 'A' + (c >= 'A' && c <= 'Z') * 10;
        for (int row = sizeof(frame) / sizeof(frame[1]) - 1; row >= max(8 - angleStr.length(), -1); row--)
        {
            uint32_t temp = fonts[index][row] << (7 - i);
            for (int col = sizeof(frame[1]) / sizeof(frame[1][1]) - 1; col >= max(12 - angleStr.length(), -1); col--)
            {
                frame[row][col] |= (temp >> (11 - col)) & HIGH;
            }
        }
    }
}
