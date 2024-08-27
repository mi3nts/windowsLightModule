#
import serial
import datetime
from mintsXU4 import mintsSensorReader as mSR
from mintsXU4 import mintsDefinitions as mD
import sys
import traceback

dataFolder  = mD.dataFolder
nanoPorts   = mD.nanoPorts
baudRate    = 9600

def main(portNum):
    if(len(nanoPorts)>0):

        ser = serial.Serial(
        port= nanoPorts[portNum],\
        baudrate=baudRate,\
        parity  =serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
        timeout=0)

        print(" ")
        print("Connected to: " + ser.portstr)
        print(" ")

        #this will store the line
        line = []

        while True:
            try:
            # if (True):
                for c in ser.read():
                    line.append(chr(c))
                    if chr(c) == '~':
                        dataString     = (''.join(line))
                        dataStringPost = dataString.replace('~', '')
                        print("================")
                        print(dataStringPost)
                        mSR.dataSplit(dataStringPost,datetime.datetime.now())
                        line = []
                        break
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
    portNum = int(sys.argv[1])
    print("Number of Arduino Nano devices: {0}".format(len(nanoPorts)))
    print("Monitoring Arduino Nano on port: {0}".format(nanoPorts[portNum]) + " with baudrate " + str(baudRate))
    main(portNum)

