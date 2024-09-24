import serial
import datetime
from mintsXU4 import mintsSensorReader as mSR
from mintsXU4 import mintsDefinitions as mD
import time
import pynmea2
from collections import OrderedDict
import traceback

dataFolder = mD.dataFolder
gpsPorts = mD.gpsPorts
baudRate = 9600

def main():
    reader = pynmea2.NMEAStreamReader()
    ser = serial.Serial(
        port=gpsPorts[0],
        baudrate=baudRate,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0
    )

    lastGPRMC = time.time()
    lastGPGGA = time.time()
    delta = 2
    print("connected to: " + ser.portstr)

    line = []
    try:
        while True:
            try:
                for c in ser.read():
                    line.append(chr(c))
                    if chr(c) == '\n':
                        dataString = (''.join(line))
                        dateTime = datetime.datetime.now()
                        print(dataString)
                        if dataString.startswith("$GPGGA") and mSR.getDeltaTime(lastGPGGA, delta):
                            mSR.GPSGPGGA2Write(dataString, dateTime)
                            lastGPGGA = time.time()
                        if dataString.startswith("$GPRMC") and mSR.getDeltaTime(lastGPRMC, delta):
                            mSR.GPSGPRMC2Write(dataString, dateTime)
                            lastGPRMC = time.time()
                        line = []
                        break
            except Exception as e:
                print("Exception occurred: Incomplete String Read")
                traceback.print_exc()
                line = []
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping...")
    except Exception as e:
        print("An unexpected error occurred:")
        traceback.print_exc()
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    print("=============")
    print("    MINTS    ")
    print("=============")
    print("Monitoring GPS Sensor on port: {0}".format(gpsPorts[0]) + " with baudrate " + str(baudRate))
    main()




