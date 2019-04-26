"""
    Left Camera is Front Camera
    Right Camera is Side Camera
"""
import numpy as np
import cv2
import pandas as pd
import os
import sys
import time
from datetime import datetime
from datetime import timedelta

from utils.object_detection import ObjectDetection
from utils.frame_rate import FrameRate
from utils.camera_calibration import CameraCalibration
from utils.abnormal_behavior_detection import AbnormalBehaviorDetection
from utils.abnormal_behavior_detection import RangeOfAbnormalBehaviorDetection
from utils.db_connector import DBConnector


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


class ResizeImage:

    def __init__(self, width, height, cropUpper, cropBottom, cropLeft, cropRight):
        self.width = width
        self.height = height
        self.cropU = cropUpper
        self.cropB = cropBottom
        self.cropL = cropLeft
        self.cropR = cropRight

        self.fixedWidth = self.width - (self.cropL + self.cropR)
        self.fixedHeight = self.height - (self.cropU + self.cropB)

    def resize(self, image):
        return image[
               self.cropU:self.height - self.cropB,
               self.cropL:self.width - self.cropR]


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


if __name__ == '__main__':

    # Define class
    try:
        object_detection = ObjectDetection(model='rfcn_resnet101_aquarium_fish_v2_22751',
                                           labels='aquarium_fish_v2_label_map.pbtxt',
                                           num_classes=3)
        frameRate = FrameRate()
        ABDetection = AbnormalBehaviorDetection()
        mysql_connector = DBConnector()
    except Exception as e:
        print("[error code] Import utils object\n", e)
        pass

    # Frame time variable
    prevTime = 0

    # Counting for Save coordinates(x, y)
    countFrame = 0
    checkFrame = 0
    endTime = datetime.now() + timedelta(days=1)
    fileNum = 0

    # Queue for pattern-finding
    # TODO: Try change data length of dir(latest_behavior_pattern).
    patternArr_size = 6 * (60) * (30)   # Input minutes, ex) (30)min == 6fps * (60)sec * (30)
                                        # Testing,       ex) 10sec = 6fps * (10) * (1)
    patternArr = [] * patternArr_size

    # TODO: Try change data length for check_behavior_pattern_2st.
    queue_size_of_speed = (60) * (30)   # Input minutes, ex) (30)min == (60)sec * (30)
                                        # Testing,       ex) 10sec = (10) * (1)
    queue_of_speed = [] * queue_size_of_speed

    input_leftCam = None
    input_rightCam = None

    # Check the input device
    check_calibration = False
    try:
        if sys.argv[1] == 'camera':
            # Input camera definition
            input_leftCam = 0
            input_rightCam = 1
            check_calibration = True

        elif sys.argv[1] == 'video':
            # Input video definition
            input_leftCam = 'frontCam.avi'
            input_rightCam = 'sideCam.avi'
            check_calibration = False

        elif sys.argv[1] == 'realsense':
            # Input realSense definition
            input_leftCam = 2
            input_rightCam = 5
            check_calibration = False

        else:
            print("\n"
                  "  ***************************************  \n\n"
                  "    python detector.py [input device]\n\n"
                  "    ex)\n\n"
                  "      1. Using USB CAMERA\n"
                  "        python detector.py camera\n\n"
                  "      2. Using VIDEO\n"
                  "        python detector.py video\n\n"
                  "      3. Using RealSense\n"
                  "        python detector.py realsense\n\n"
                  "  ***************************************  \n")
            exit()

    except Exception as e:
        print("\n"
              "  ***************************************  \n\n"
              "    python detector.py [input device]\n\n"
              "    ex)\n\n"
              "      1. Using USB CAMERA\n"
              "        python detector.py camera\n\n"
              "      2. Using VIDEO\n"
              "        python detector.py video\n\n"
              "      3. Using RealSense\n"
              "        python detector.py realsense\n\n"
              "  ***************************************  \n")
        exit()

    # Opencv, Video capture
    leftCam = cv2.VideoCapture(input_leftCam)
    rightCam = cv2.VideoCapture(input_rightCam)

    if check_calibration == True:
        camWidth = 1280
        camHeight = 720
        cropWidth = 960
        camera_calibration = CameraCalibration(cap1=leftCam,
                                               cap2=rightCam,
                                               camWidth=camWidth,
                                               camHeight=camHeight,
                                               cropWidth=cropWidth)

        camera_calibration.set_stereoMatcher()

        '''
            The distortion in the leftCam and rightCam edges prevents a good calibration,
            so discard the edges
        '''
        camera_calibration.set_resolution(cap=leftCam)
        camera_calibration.set_resolution(cap=rightCam)

    elif check_calibration == False:
        leftWidth, leftHeight = int(leftCam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(leftCam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rightWidth, rightHeight = int(rightCam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(rightCam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        camWidth = leftWidth
        camHeight = leftHeight

    # TODO: Try applying the size of the part you want to crop from the image.
    if check_calibration == True:
        leftResize = ResizeImage(width=camera_calibration.cropWidth,
                                 height=camera_calibration.camHeight,
                                 cropUpper=0,
                                 cropBottom=0,
                                 cropLeft=0,
                                 cropRight=0)

        rightResize = ResizeImage(width=camera_calibration.cropWidth,
                                  height=camera_calibration.camHeight,
                                  cropUpper=0,
                                  cropBottom=0,
                                  cropLeft=0,
                                  cropRight=0)

    elif check_calibration == False:
        leftResize = ResizeImage(width=camWidth,
                                 height=camHeight,
                                 cropUpper=0,
                                 cropBottom=0,
                                 cropLeft=0,
                                 cropRight=0)

        rightResize = ResizeImage(width=camWidth,
                                  height=camHeight,
                                  cropUpper=0,
                                  cropBottom=0,
                                  cropLeft=0,
                                  cropRight=0)

    leftCam_fixedWidth = leftResize.fixedWidth
    leftCam_fixedHeight = leftResize.fixedHeight
    rightCam_fixedWidth = rightResize.fixedWidth
    rightCam_fixedHeight = rightResize.fixedHeight

    # recVideo_leftCam = RecVideo(fileName='frontCam',
    #                             width=leftCam_fixedWidth,
    #                             height=leftCam_fixedHeight)
    # recording_leftCam = recVideo_leftCam.recording()
    #
    # recVideo_rightCam = RecVideo(fileName='sideCam',
    #                              width=rightCam_fixedWidth,
    #                              height=rightCam_fixedHeight)
    # recording_rightCam = recVideo_rightCam.recording()

    recVideo_Cam = RecVideo(fileName='Cam',
                            width=leftCam_fixedWidth+rightCam_fixedWidth,
                            height=rightCam_fixedHeight)
    recording_Cam = recVideo_Cam.recording()

    while True:
        if not leftCam.grab() or not rightCam.grab():
            print("\n"
                  "  *******************************************  \n\n"
                  "    [error code] Please check the input device\n\n"
                  "  *******************************************  \n")
            break

        _, leftFrame = leftCam.read()
        _, rightFrame = rightCam.read()

        if input_leftCam == 'flowerhorn.mp4':
            leftFrame = ImageRotate(image=leftFrame, degrees=90).rotate()
            rightFrame = ImageRotate(image=rightFrame, degrees=90).rotate()
        elif input_leftCam == 'jinjurin.mp4':
            leftFrame = ImageRotate(image=leftFrame, degrees=90).rotate()
            rightFrame = ImageRotate(image=rightFrame, degrees=90).rotate()

        # Frame rate
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = "FPS %0.1f" % (1 / (sec))

        # Show text in upper left corner
        frameRate.putText(frame=leftFrame, text='Front Camera ' + fps)
        frameRate.putText(frame=rightFrame, text='Side Camera ' + fps)

        try:
            if check_calibration == True:
                leftFrame, rightFrame = camera_calibration.stereo_calibration(frame1=leftFrame,
                                                                              frame2=rightFrame)

                disparity, coordinates_of_3D = camera_calibration.depth_map_creator(frame1=leftFrame,
                                                                                    frame2=rightFrame)

            elif check_calibration == False:
                pass

        except Exception as e:
            print("[error code] Stereo calibration, Depth map creator\n", e)
            pass

        try:
            # Resize image to match the size of the fish-tank
            if check_calibration == True:
                leftFrame = leftResize.resize(image=leftFrame)
                rightFrame = rightResize.resize(image=rightFrame)

            elif check_calibration == False:
                leftFrame = leftResize.resize(image=leftFrame)
                rightFrame = rightResize.resize(image=rightFrame)

        except Exception as e:
            print("[error code] Resize Image\n", e)
            pass

        try:
            # Run detector
            leftCam_boxes, leftCam_scores, leftCam_classes, leftCam_category_index = object_detection.run(image_np=leftFrame,
                                                                                                          display=True)
            rightCam_boxes, rightCam_scores, rightCam_classes, rightCam_category_index = object_detection.run(image_np=rightFrame,
                                                                                                              display=True)

            # Data processing
            _, leftCam_object_point, leftCam_x_min, leftCam_y_min, leftCam_x_max, leftCam_y_max, _ = object_detection.data_processing(
                image_np=leftFrame,
                boxes=leftCam_boxes,
                scores=leftCam_scores,
                classes=leftCam_classes,
                category_index=leftCam_category_index,
                point_buff=object_detection.leftCam_point_buff)
            _, rightCam_object_point, rightCam_x_min, rightCam_y_min, rightCam_x_max, rightCam_y_max, _ = object_detection.data_processing(
                image_np=rightFrame,
                boxes=rightCam_boxes,
                scores=rightCam_scores,
                classes=rightCam_classes,
                category_index=rightCam_category_index,
                point_buff=object_detection.rightCam_point_buff)

        except Exception as e:
            print("[error code] TF Object Detection\n", e)
            pass

        try:
            if check_calibration == True:
                Pl, Pr = camera_calibration.triangulation(leftCam_point=leftCam_object_point,
                                                          rightCam_point=rightCam_object_point)
                # print("Distance: {}".format(round(Pr[2] / 300, 2)))

            elif check_calibration == False:
                pass

        except Exception as e:
            # print("[error code] Triangulation\n", e)
            pass

        try:
            '''
            Method 1
                3D positions (x, y and depth) were assigned combining
                the horizontal coordinates of the left camera (coordinates_x and coordinates_y) and
                the vertical coordinate of the right camera (depth using x-axis)
                leftCam_object_point[0]: coordinates_x
                leftCam_object_point[1]: coordinates_y
                rightCam_object_point[0]: depth
            '''
            current_time = datetime.now()
            year = current_time.year
            month = current_time.month
            day = current_time.day
            hour = current_time.hour
            minute = current_time.minute
            second = current_time.second
            timestamp = '{}.{}.{}.{}.{}.{}'.format(year, month, day, hour, minute, second)
            pointValue = [
                (leftCam_object_point[0],
                 leftCam_object_point[1],
                 rightCam_object_point[0],
                 timestamp,
                 leftCam_fixedWidth,
                 leftCam_fixedHeight,
                 rightCam_fixedWidth,
                 rightCam_fixedHeight)
            ]

            '''
            Method 2
                Using stereo camera calibration and depth map
                xs: coordinates_x
                ys: coordinates_y
                zs: depth
            '''
            # xs = coordinates_of_3D[leftCam_object_point[0], leftCam_object_point[1], 0]
            # ys = coordinates_of_3D[leftCam_object_point[0], leftCam_object_point[1], 1]
            # zs = coordinates_of_3D[leftCam_object_point[0], leftCam_object_point[1], 2]
            # pointValue = [
            #     (xs,
            #      ys,
            #      zs,
            #      time.time(),
            #      leftCam_fixedWidth,
            #      leftCam_fixedHeight,
            #      rightCam_fixedWidth,
            #      rightCam_fixedHeight)
            #     ]

            # Save data to a csv file
            column_name = ['coordinates_x', 'coordinates_y', 'depth', 'timestamp', 'frontCam_w', 'frontCam_h', 'sideCam_w', 'sideCam_h']
            behavior_pattern_save_dir = 'save_behavior_pattern'
            if not os.path.isdir(behavior_pattern_save_dir):
                os.mkdir(behavior_pattern_save_dir)

            save_file = behavior_pattern_save_dir + '/save_{}.csv'.format(str(fileNum).rjust(4, '0'))
            if os.path.exists(save_file) == True:
                with open(save_file, 'a') as f:
                    xml_df = pd.DataFrame(pointValue, columns=column_name)
                    xml_df.to_csv(f, header=False, index=None)
            else:
                xml_df = pd.DataFrame(columns=column_name)
                xml_df.to_csv(save_file, index=None)
                with open(save_file, 'a') as f:
                    xml_df = pd.DataFrame(pointValue, columns=column_name)
                    xml_df.to_csv(f, header=False, index=None)

            # Save in the last 30minutes behavior pattern data
            patternArr.append(pointValue)
            patternArr = patternArr[-patternArr_size:]

            # DB Connection
            try:
                pass
                """conn = mysql_connector.connect_mysql()
                curs = conn.cursor()
                mysql_connector.insert_data(curs=curs,
                                            conn=conn,
                                            coordinates_x=leftCam_object_point[0],
                                            coordinates_y=leftCam_object_point[1],
                                            depth=rightCam_object_point[0],
                                            timestamp=time.time(),
                                            frontCam_w=leftCam_fixedWidth,
                                            frontCam_h=leftCam_fixedHeight,
                                            sideCam_w=rightCam_fixedWidth,
                                            sideCam_h=rightCam_fixedHeight)"""
            except Exception as e:
                print("[error code] MySQL DB\n", e)
                pass

        # When it can not be detected even from one frame
        except Exception as e:
            print("[error code] Data Array\n", e)
            pass

        # ============================================================================
        #   1st pattern : If object stay on the edge of the screen for a long time
        #   and abnormal behavior detection area
        # ============================================================================
        # TODO: Try change data length of abnormal behavior point.
        abnormal_behavior_size_x = 120
        abnormal_behavior_size_y = 120
        RangeOfABD = RangeOfAbnormalBehaviorDetection(leftFrame=leftFrame,
                                                      rightFrame=rightFrame,
                                                      range_x=abnormal_behavior_size_x,
                                                      range_y=abnormal_behavior_size_y,
                                                      leftCam_object_point=leftCam_object_point,
                                                      rightCam_object_point=rightCam_object_point)

        try:
            # Range of Abnormal Behavior Detection
            RangeOfABD.line2()

            pattern1st_Arr = patternArr
            check_abnormal_behavior_list = []

            # The front of the Smart-Aquarium
            # Left & Upper
            check_abnormal_behavior = ABDetection.pattern_1st(x1=0, x2=abnormal_behavior_size_x,
                                                              y1=0, y2=abnormal_behavior_size_y,
                                                              z1=0, z2=abnormal_behavior_size_x,
                                                              patternArr_size=patternArr_size,
                                                              patternArr=pattern1st_Arr)
            check_abnormal_behavior_list.append(check_abnormal_behavior)

            # Right & Upper
            check_abnormal_behavior = ABDetection.pattern_1st(x1=leftCam_fixedWidth - abnormal_behavior_size_x, x2=leftCam_fixedWidth,
                                                              y1=0, y2=abnormal_behavior_size_y,
                                                              z1=0, z2=abnormal_behavior_size_x,
                                                              patternArr_size=patternArr_size,
                                                              patternArr=pattern1st_Arr)
            check_abnormal_behavior_list.append(check_abnormal_behavior)

            # Left & Bottom
            check_abnormal_behavior = ABDetection.pattern_1st(x1=0, x2=abnormal_behavior_size_x,
                                                              y1=leftCam_fixedHeight - abnormal_behavior_size_y, y2=leftCam_fixedHeight,
                                                              z1=0, z2=abnormal_behavior_size_x,
                                                              patternArr_size=patternArr_size,
                                                              patternArr=pattern1st_Arr)
            check_abnormal_behavior_list.append(check_abnormal_behavior)

            # Right & Bottom
            check_abnormal_behavior = ABDetection.pattern_1st(x1=leftCam_fixedWidth - abnormal_behavior_size_x, x2=leftCam_fixedWidth,
                                                              y1=leftCam_fixedHeight - abnormal_behavior_size_y, y2=leftCam_fixedHeight,
                                                              z1=0, z2=abnormal_behavior_size_x,
                                                              patternArr_size=patternArr_size,
                                                              patternArr=pattern1st_Arr)
            check_abnormal_behavior_list.append(check_abnormal_behavior)

            # The back of the Smart-Aquarium
            # Left & Upper
            check_abnormal_behavior = ABDetection.pattern_1st(x1=0, x2=abnormal_behavior_size_x,
                                                              y1=0, y2=abnormal_behavior_size_y,
                                                              z1=leftCam_fixedWidth - abnormal_behavior_size_x, z2=leftCam_fixedWidth,
                                                              patternArr_size=patternArr_size,
                                                              patternArr=pattern1st_Arr)
            check_abnormal_behavior_list.append(check_abnormal_behavior)

            # Right & Upper
            check_abnormal_behavior = ABDetection.pattern_1st(x1=leftCam_fixedWidth - abnormal_behavior_size_x, x2=leftCam_fixedWidth,
                                                              y1=0, y2=abnormal_behavior_size_y,
                                                              z1=abnormal_behavior_size_x, z2=leftCam_fixedWidth,
                                                              patternArr_size=patternArr_size,
                                                              patternArr=pattern1st_Arr)
            check_abnormal_behavior_list.append(check_abnormal_behavior)

            # Left & Bottom
            check_abnormal_behavior = ABDetection.pattern_1st(x1=0, x2=abnormal_behavior_size_x,
                                                              y1=leftCam_fixedHeight - abnormal_behavior_size_y, y2=leftCam_fixedHeight,
                                                              z1=leftCam_fixedWidth - abnormal_behavior_size_x, z2=leftCam_fixedWidth,
                                                              patternArr_size=patternArr_size,
                                                              patternArr=pattern1st_Arr)
            check_abnormal_behavior_list.append(check_abnormal_behavior)

            # Right & Bottom
            check_abnormal_behavior = ABDetection.pattern_1st(x1=leftCam_fixedWidth - abnormal_behavior_size_x, x2=leftCam_fixedWidth,
                                                              y1=leftCam_fixedHeight - abnormal_behavior_size_y, y2=leftCam_fixedHeight,
                                                              z1=leftCam_fixedWidth - abnormal_behavior_size_x, z2=leftCam_fixedWidth,
                                                              patternArr_size=patternArr_size,
                                                              patternArr=pattern1st_Arr)
            check_abnormal_behavior_list.append(check_abnormal_behavior)

            for i in check_abnormal_behavior_list:
                if i == 'Detect abnormal behavior':
                    pattern1st_Arr.clear()
                    ABDetection.display(num_pattern=1,
                                        leftFrame=leftFrame,
                                        rightFrame=rightFrame,
                                        leftCam_w=leftCam_fixedWidth,
                                        leftCam_h=leftCam_fixedHeight,
                                        rightCam_w=rightCam_fixedWidth,
                                        rightCam_h=rightCam_fixedHeight)
            check_abnormal_behavior_list.clear()

        # When it can not be detected even from one frame
        except Exception as e:
            # print("[error code] 1st pattern", e)
            # Range of Abnormal Behavior Detection
            RangeOfABD.line()
            pass

        # ==================================================================
        #   2st pattern : If the movement is noticeably slower or faster
        # ==================================================================
        try:
            pattern2st_Arr = patternArr
            for i in range(len(pattern2st_Arr)):
                '''
                    pattern2st_Arr[i][0][0], coordinates_x
                    pattern2st_Arr[i][0][1], coordinates_y
                    pattern2st_Arr[i][0][2], depth
                    pattern2st_Arr[i][0][3], timestamp
                '''
                lastData = len(pattern2st_Arr) - 1
                oneSecondPreviousData = lastData - int(float("{0:.1f}".format(1/sec)))
                speed = ABDetection.speed_of_three_dimensional(resolution_x=leftCam_fixedWidth,
                                                               resolution_y=leftCam_fixedHeight,
                                                               resolution_z=rightCam_fixedWidth,
                                                               coordinates_x1=(pattern2st_Arr[oneSecondPreviousData][0])[0],
                                                               coordinates_x2=(pattern2st_Arr[lastData][0])[0],
                                                               coordinates_y1=(pattern2st_Arr[oneSecondPreviousData][0])[1],
                                                               coordinates_y2=(pattern2st_Arr[lastData][0])[1],
                                                               depth1=(pattern2st_Arr[oneSecondPreviousData][0])[2],
                                                               depth2=(pattern2st_Arr[lastData][0])[2],
                                                               time1=(pattern2st_Arr[oneSecondPreviousData][0])[3],
                                                               time2=(pattern2st_Arr[lastData][0])[3])
            cv2.putText(leftFrame, '{0:.2f}mm/s'.format(speed), (leftCam_x_max, leftCam_y_max), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.putText(rightFrame, '{0:.2f}mm/s'.format(speed), (rightCam_x_max, rightCam_y_max), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            if countFrame == 0:
                check_abnormal_behavior = ABDetection.pattern_2st(speed=speed,
                                                                  queue_size_of_speed=queue_size_of_speed,
                                                                  queue_of_speed=queue_of_speed)
                if check_abnormal_behavior == 'Detect abnormal behavior':
                    queue_of_speed.clear()
                    ABDetection.display(num_pattern=2,
                                        leftFrame=leftFrame,
                                        rightFrame=rightFrame,
                                        leftCam_w=leftCam_fixedWidth,
                                        leftCam_h=leftCam_fixedHeight,
                                        rightCam_w=rightCam_fixedWidth,
                                        rightCam_h=rightCam_fixedHeight)

        # When it can not be detected even from one frame
        except Exception as e:
            # print("[error code] 2st pattern\n", e)
            pass

        # ================================================
        #   3st pattern : If detect white spot disease
        # ================================================
        try:
            leftCam_check_white_spot_disease = ''
            rightCam_check_white_spot_disease = ''

            if (leftCam_x_min != None) and (leftCam_x_max != None) and (leftCam_x_min != None) and (leftCam_y_max != None):
                cropLeft_object = leftFrame[leftCam_y_min:leftCam_y_max, leftCam_x_min:leftCam_x_max]
                h = leftCam_y_max-leftCam_y_min
                w = leftCam_x_max-leftCam_x_min
                resizeLeft = cv2.resize(cropLeft_object, (w * 2, h * 2))
                leftCam_check_white_spot_disease = ABDetection.pattern_3st(frame=resizeLeft,
                                                                           title='frontCam')

            if (rightCam_x_min != None) and (rightCam_x_max != None) and (rightCam_y_min != None) and (rightCam_y_max != None):
                cropRight_object = rightFrame[rightCam_y_min:rightCam_y_max, rightCam_x_min:rightCam_x_max]
                h = rightCam_y_max-rightCam_y_min
                w = rightCam_x_max-rightCam_x_min
                resizeRight = cv2.resize(cropRight_object, (w * 2, h * 2))
                rightCam_check_white_spot_disease = ABDetection.pattern_3st(frame=resizeRight,
                                                                            title='sideCam')

            if (leftCam_check_white_spot_disease == 'Detect white spot disease') or (rightCam_check_white_spot_disease == 'Detect white spot disease'):
                ABDetection.display(num_pattern=3,
                                    leftFrame=leftFrame,
                                    rightFrame=rightFrame,
                                    leftCam_w=leftCam_fixedWidth,
                                    leftCam_h=leftCam_fixedHeight,
                                    rightCam_w=rightCam_fixedWidth,
                                    rightCam_h=rightCam_fixedHeight)

        # When it can not be detected even from one frame
        except Exception as e:
            # print("[error code] 3st pattern\n", e)
            pass

        countFrame += 1
        if object_detection.model[:16] == 'ssd_inception_v2':
            checkFrame = int(float("{0:.1f}".format(1/sec)))
        elif object_detection.model[:14] == 'rfcn_resnet101':
            checkFrame = int(float("{0:.1f}".format(1/sec)))
        if countFrame > checkFrame:
            countFrame = 0
        if datetime.now() >= endTime:
            fileNum += 1
            endTime = datetime.now() + timedelta(days=1)

        # Adjusting video resolution (width, height)
        # vertical_frame = cv2.vconcat([leftFrame, rightFrame])
        horizon_frame = cv2.hconcat([leftFrame, rightFrame])
        cv2.imshow('Detector', cv2.resize(horizon_frame, (720 * 2, 720)))

        if check_calibration == True:
            depth_frame = disparity / camera_calibration.DEPTH_VISUALIZATION_SCALE
            cv2.imshow('Depth', cv2.resize(depth_frame, (640, 480)))

        # recVideo_leftCam.output(frame=leftFrame,
        #                         recording=recording_leftCam)
        # recVideo_rightCam.output(frame=rightFrame,
        #                          recording=recording_rightCam)
        recVideo_Cam.output(frame=horizon_frame,
                            recording=recording_Cam)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    leftCam.release()
    rightCam.release()
    cv2.destroyAllWindows()
