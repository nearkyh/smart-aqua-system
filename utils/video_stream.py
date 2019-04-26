import numpy as np
import cv2
import time


def get_frame(video1='rtsp://192.168.0.129:8091/artik_cam1.mp4',
              video2='rtsp://192.168.0.67:8091/artik_cam2.mp4'):
        # Opencv, Video capture
        leftCam = cv2.VideoCapture(video1)
        rightCam = cv2.VideoCapture(video2)

        # Frame time variable
        prevTime = 0

        while True:
            if not leftCam.grab() or not rightCam.grab():
                print("No more frames")
                break

            # ================
            #   Frame rate
            # ================
            curTime = time.time()
            sec = curTime - prevTime
            prevTime = curTime
            frameRate = "FPS %0.1f" % (1 / (sec))

            # ===============================
            #   Stereo camera calibration
            # ===============================
            _, leftFrame = leftCam.read()
            _, rightFrame = rightCam.read()

            # ================================================
            #   Adjusting video resolution (width, height)
            # ================================================
            # vertical_frame = cv2.vconcat([fixedLeft, fixedRight])
            horizon_frame = cv2.hconcat([leftFrame, rightFrame])

            im_encode = cv2.imencode('.jpg', horizon_frame)[1]
            stringData = im_encode.tostring()
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')
