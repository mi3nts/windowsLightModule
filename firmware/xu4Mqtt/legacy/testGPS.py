import serial

# Configure the serial port
ser = serial.Serial('COM5', 9600, timeout=1)

# Read data from the serial port
try:
    while True:
        if ser.in_waiting > 0:  # Check if there is data waiting to be read
            data = ser.readline().decode('utf-8').rstrip()
            print(data)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()