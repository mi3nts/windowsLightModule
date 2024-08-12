from datetime import timezone
import time
import os
import datetime
import numpy as np
import pickle
from skimage import io, color
import cv2

from mintsXU4 import mintsSkyCamReader as mSCR
from mintsXU4 import mintsSensorReader as mSR
from mintsXU4 import mintsDefinitions as mD

loopInterval = 300 # 5 minutes
dataFolder = mD.dataFolder

def delayMints(timeSpent,loopIntervalIn):
    loopIntervalReal = loopIntervalIn ;
    if(loopIntervalReal>timeSpent):
        waitTime = loopIntervalReal - timeSpent;
        time.sleep(waitTime);
    return time.time();


def main():

    sensorName = "SKYCAM004"
    dateTimeNow = datetime.datetime.now()
    subFolder     = mSR.getWritePathSnaps(sensorName,dateTimeNow)

    # onboardCapture = True
    try:
        while True:
            try:
                startTime = time.time()
                currentImage,imagePath =  mSCR.getSnapShotXU4(subFolder)
                print("Sleeping for " + str(loopInterval) + " Seconds")
                startTime = delayMints(time.time() - startTime,loopInterval)
            except Exception as e:
                # Handle any exception
                print(f"An unexpected error occurred: {e}")

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
    



if __name__ == "__main__":
   main()
