import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

filename = "capture/cam0/cam0_image{:06d}.jpg"
filename2 = "capture/cam1/cam1_image{:06d}.jpg"

frameId = 0
while(True) :
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    cv2.imshow('cam0', frame)
    cv2.imshow('cam1', frame2)
    horizon_frame = cv2.hconcat([frame, frame2])
    cv2.imshow('Test', cv2.resize(horizon_frame, (640 * 2, 480)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # cv2.imwrite(filename.format(frameId), frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
        # cv2.imwrite(filename2.format(frameId), frame2, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
        break
    frameId += 1

cap.release()
cap2.release()
cv2.destroyAllWindows()
