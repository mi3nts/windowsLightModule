from datetime import timezone
import time
import os
import datetime
import netifaces as ni
from collections import OrderedDict
import netifaces as ni
from requests import get
import socket 

from mintsXU4 import mintsSensorReader as mSR
from mintsXU4 import mintsDefinitions  as mD

dataFolder = mD.dataFolder


def main():

    sensorName = "IP"
    dateTimeNow = datetime.datetime.now()
    print("Gaining Public and Private IPs")

    # Getting the public IP address
    publicIp = get('https://api.ipify.org').text

    # Getting the local IP address for Windows
    hostname = socket.gethostname()
    localIp  = socket.gethostbyname(hostname)

    # Creating the sensor dictionary
    sensorDictionary = OrderedDict([
        ("dateTime", dateTimeNow.strftime('%Y-%m-%d %H:%M:%S.%f')),
        ("publicIp", publicIp),
        ("localIp", localIp)
    ])


    mSR.sensorFinisherIP(dateTimeNow,sensorName,sensorDictionary)

if __name__ == "__main__":
    print("=============")
    print("    MINTS    ")
    print("=============")
    main()