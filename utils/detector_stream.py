import numpy as np
import cv2
import time

from utils.object_detection import ObjectDetection


class RecVideo:

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


class ImageRotate:

    def __init__(self, image, degrees):
        self.image = image
        self.degrees = degrees

    def rotate(self):
        if self.degrees == 90:
            dst = cv2.transpose(self.image)
            dst = cv2.flip(dst, 1)
            return dst
        elif self.degrees == 180:
            dst = cv2.flip(self.image, -1)
            return dst
        elif self.degrees == 270:
            dst = cv2.transpose(self.image)
            dst = cv2.flip(dst, 0)
            return dst
        else:
            pass


def get_frame(video):
    try:
        object_detection = ObjectDetection()
    except Exception as e:
        print("[error code] Import utils\n")

    # Opencv, Video capture
    leftCap = cv2.VideoCapture(video)

    # Frame time variable
    prevTime = 0

    leftWidth = int(leftCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    leftHeight = int(leftCap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Increase the resolution
    # leftWidth = 1280
    # leftHeight = 720
    # set_resolution(frame=leftCap,
    #                width=leftWidth,
    #                height=leftHeight)

    recVideo_leftCam = RecVideo(fileName='monitoring',
                                 width=leftWidth,
                                 height=leftHeight)
    recording_leftCam = recVideo_leftCam.recording()

    while True:
        try:
            if not leftCap.grab():
                print("No more frames")
                break

            _, leftFrame = leftCap.read()

            # leftFrame = rotate(leftFrame, 90)

            # Frame rate
            curTime = time.time()
            sec = curTime - prevTime
            prevTime = curTime
            frameRate = "FPS %0.1f" % (1 / (sec))

            try:
                # Run detector
                leftCam_boxes, leftCam_scores, leftCam_classes, leftCam_category_index = object_detection.run(image_np=leftFrame)

                # Data processing
                _, leftCam_object_point, leftCam_x_min, leftCam_y_min, leftCam_x_max, leftCam_y_max, _ = object_detection.data_processing(
                    image_np=leftFrame,
                    boxes=leftCam_boxes,
                    scores=leftCam_scores,
                    classes=leftCam_classes,
                    category_index=leftCam_category_index,
                    point_buff=object_detection.leftCam_point_buff)

            except Exception as e:
                print("[error code] TF Object Detection\n", e)
                pass

            recVideo_leftCam.output(frame=leftFrame,
                                     recording=recording_leftCam)

            im_encode = cv2.imencode('.jpg', leftFrame)[1]
            stringData = im_encode.tostring()
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')

        except Exception:
            pass
