import numpy as np
import cv2


class VideoRecorder:

    def __init__(self, fileName, width, height):
        self.fileName = fileName
        self.width = width
        self.height = height

    def recording(self):
        recording_video = "rec_{}.avi".format(self.fileName)
        fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        rec_frameRate = 10.0

        return cv2.VideoWriter(recording_video, fcc, rec_frameRate, (self.width, self.height))

    def output(self, frame, recording):
        recording.write(frame)
