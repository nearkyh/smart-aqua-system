import os
from time import time
import numpy as np
import cv2


LEFT_PATH = "capture/left/{:06d}.jpg"
RIGHT_PATH = "capture/right/{:06d}.jpg"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960

# TODO: Use more stable identifiers
left = cv2.VideoCapture(0)
right = cv2.VideoCapture(1)

# Increase the resolution
left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

frameId = 0
# Frame time variable
prevTime = 0
countFrame = 0
checkFrame = 0

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    curTime = time()
    sec = curTime - prevTime
    prevTime = curTime
    frameRate = "FPS %0.1f" % (1 / (sec))

    if not (left.grab() and right.grab()):
        print("No more frames")
        break

    _, leftFrame = left.retrieve()
    _, rightFrame = right.retrieve()

    if countFrame == 0:
        try:
            cv2.imwrite(LEFT_PATH.format(frameId), leftFrame)
            cv2.imwrite(RIGHT_PATH.format(frameId), rightFrame)
        except Exception:
            pass

    countFrame += 1
    checkFrame = 5
    if countFrame > checkFrame:
        countFrame = 0

    cv2.imshow('left', leftFrame)
    cv2.imshow('right', rightFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frameId += 1

left.release()
right.release()
cv2.destroyAllWindows()