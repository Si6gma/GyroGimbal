import serial
import time
import random

# Create a serial connection
ser = serial.Serial('/dev/cu.usbmodemDC5475C3BB642', 115200)

# Give some time for the connection to establish
time.sleep(2)

try:
    while True:
        # Generate a random number between 1 and 100
        rand_num = random.randint(1, 100)
        
        # Convert the number to a string and encode it
        data = str(rand_num).encode()
        
        # Send the data
        ser.write(data)
        
        # Wait for a second before sending the next number
        time.sleep(.5)

except KeyboardInterrupt:
    print("Stopped")

finally:
    # Close the serial connection
    ser.close()
