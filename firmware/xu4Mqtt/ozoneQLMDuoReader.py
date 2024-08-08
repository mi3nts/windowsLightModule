

#
import serial
import datetime
from mintsXU4 import mintsSensorReader as mSR
from mintsXU4 import mintsDefinitions as mD
import time
import sys

dataFolderReference    = mD.dataFolderReference
ozonePorts             = mD.ozonePort

def ozoneReader(portIn,baudRateIn,ser):

    menuSetUp = False
    print(" ")
    print("Connected to: " + ser.portstr)
    print(" ")
    line = []
    ser.write(str.encode('x'))
    while True:
        try:
            for c in ser.read():
                line.append(chr(c))
                
                if chr(c) == '\n' and not(menuSetUp):
                    dataString = ''.join(line)
                    dataString     = (''.join(line)).replace("\n","").replace("\r","")
                    
                    print("Entering Menu")
                    ser.write(str.encode('m'))
                    time.sleep(2)
                    print("Setting Frequency to 10 Seconds")
                    ser.write(str.encode('a'))
                    time.sleep(2)
                    ser.write(str.encode('1'))
                    time.sleep(2)

                    print("Setting Ozone Units to ppb")
                    ser.write(str.encode('u'))
                    time.sleep(2)
                    ser.write(str.encode('0'))
                    time.sleep(2)

                    print("Setting Temperature Units to C")
                    ser.write(str.encode('c'))
                    time.sleep(2)
                    ser.write(str.encode('1'))
                    time.sleep(2)

                    print("Setting Pressure Units to mbar")
                    ser.write(str.encode('o'))
                    time.sleep(2)
                    ser.write(str.encode('1'))
                    time.sleep(2)

                    print("Exiting Menu")
                    ser.write(str.encode('x'))
                    time.sleep(2)
                    menuSetUp = True
                    line = []

                if chr(c) == '\n' and (menuSetUp):
                    dataString = ''.join(line)
                    dataString     = (''.join(line)).replace("\n","").replace("\r","")
                    print(dataString)
                    dateTime = datetime.datetime.now()

					# The Output shouldnt have any letters
                    if(not(any(c.isalpha() for c in dataString))):
                        mSR.TB108LWrite(dataString,dateTime)
                    line = []
        except:
            print("Incomplete read. Error Detected: {0}")
            line = []

def QLMReader(portIn,baudRateIn,ser):

        #this will store the line
        line = []

        while True:
            try:
                for c in ser.read():
                    line.append(chr(c))

                    if chr(c) == '\n':
                        dataString     = (''.join(line))
                        dataStringPost = dataString.replace('\n', '')
                        currentDateTime = datetime.datetime.now()
                        if dataString.find('START Watchdog Reset;')>0:
                            dataStringPost = dataStringPost.replace('START Watchdog Reset;', '')
                            mSR.QLMRAD001Write(dataStringPost,currentDateTime)
                            mSR.QLMRAD001Write("-100",currentDateTime)
                        else:
                            print("================")
                            print(dataStringPost)
                            mSR.QLMRAD001Write(dataStringPost,currentDateTime)
                        line = []
                        break

            except:
                print("Incomplete String Read")
                line = []
        ser.close()

def setUpSerialPort(portIn,baudRateIn):
    print("Settin up serial port for Port:" + str(portIn)+ " and Baud Rate "  + str(baudRateIn))
    ser = serial.Serial(
    port= portIn,\
    baudrate=baudRateIn,\
	parity  =serial.PARITY_NONE,\
	stopbits=serial.STOPBITS_ONE,\
	bytesize=serial.EIGHTBITS,\
    timeout=0)
    print("Connected to: " + ser.portstr)
    return ser;



def main(portIn):
    baudRates = [2400 , 115200] 
    startTime  = time.time()
    checkTime  = 30 

    for baudRateIn in baudRates:
        ser = setUpSerialPort(portIn,baudRateIn)
        while True:
            try:
                for c in ser.read():
                    # print(c)
                    if chr(c) == '\n':
                        print("Port Found @ baud rate " + str(baudRateIn))
                        if baudRateIn == 2400:
                            ozoneReader(portIn,baudRateIn,ser)
                        if baudRateIn == 115200:
                            QLMReader(portIn,baudRateIn,ser)    
                        break

                if time.time()-startTime>checkTime:
                    startTime = time.time()
                    break
            except:
                print("Incomplete read. Error Detected: {0}")

        print("No Port Found on Baud Rate: " + str(baudRateIn))
        ser.close()


if __name__ == "__main__":
    print("=============")
    print("    MINTS    ")
    print("=============")
    portNum = int(sys.argv[1])
    print("Number of Ozone/QLM  devices: {0}".format(len(ozonePorts)))
    main(ozonePorts[portNum])

