import cv2
import numpy as np

from utils.object_detection import ObjectDetection
from utils.video_recorder import VideoRecorder
from utils.frame_rate import FrameRate


fish_name = 'flowerhorn'
objectDetection = ObjectDetection(model='rfcn_resnet101_{}'.format(fish_name),
                                  labels='{}_label_map.pbtxt'.format(fish_name),
                                  num_classes=1)
videoRec = VideoRecorder(fileName='{}'.format(fish_name),
                         width=1280,
                         height=720)

if __name__ == '__main__':

    # input_cam = 2
    input_video = '{}.mp4'.format(fish_name)
    cap = cv2.VideoCapture(input_video)
    rec = videoRec.recording()

    frameRate = FrameRate()

    while True:
        _, frame = cap.read()

        objectDetection.run(image_np=frame, display=True)

        frameRate.putText(frame=frame, text=frameRate.fps())

        cv2.imshow('Detector [{}]'.format(fish_name), frame)

        videoRec.output(frame=frame, recording=rec)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


