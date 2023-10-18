#include <ArduinoR4DigitDisplay.h>

#include "Arduino_LED_Matrix.h"

ArduinoLEDMatrix matrix;
// The setup function runs once when you press reset or power the board

void setup() {
  Serial.begin(115200);
  matrix.begin();
}

uint8_t frame[8][12] = {
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};
// The loop function runs over and over again forever
void loop() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    // Read the incoming data and convert it to an integer
    int receivedNum = Serial.parseInt();
    
    // Print the received number to the Serial Monitor
    // Serial.print("Received number: ");
    // Serial.println(receivedNum);
      drawNumber(receivedNum,frame);
  matrix.renderBitmap(frame, 8, 12);
  }
}
