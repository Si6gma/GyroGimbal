import serial
import time

serialCom = serial.Serial('/dev/cu.usbmodemDC5475C3BB642',115200)
serialCom.timeout = 1

while True:
    i = input("Input(On/Off) : ").strip()
    if i == 'done':
        print('Finished Program')
        break
    serialCom.write(i.encode())
    time.sleep(.5)
    print(serialCom.readline().decode('ascii'))

serialCom.close()