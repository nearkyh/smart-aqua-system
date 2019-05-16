import numpy as np
import cv2
import pandas as pd
import os
import sys
from time import time
from datetime import datetime, timedelta
from random import randrange

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, QTime, QDateTime, QRect, Qt
from PyQt5 import uic

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt

from utils.object_detection import ObjectDetection
from utils.frame_rate import FrameRate
from utils.video_recorder import VideoRecorder
from utils.camera_calibration import CameraCalibration
from utils.abnormal_behavior_detection import AbnormalBehaviorDetection
from utils.abnormal_behavior_detection import RangeOfAbnormalBehaviorDetection
from utils.db_connector import DBConnector
from utils.visualization_3D import Visualization3D


form_class = uic.loadUiType("main_window.ui")[0]


class MyWindow(QMainWindow, form_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Smart Aqua System")

        # Set Visualization Widget
        self.graphicsView = gl.GLViewWidget(self.tab_visualization)
        self.graphicsView.setGeometry(QRect(260, 40, 500, 500))
        self.graphicsView.setObjectName("graphicsView")

        # Init 3D Plotting
        self.axisX = 640
        self.axisY = 640
        self.axisZ = 480.0
        self.initScale = {'x': int(self.axisX / 20), 'y': int(self.axisY / 20), 'z': 10}
        self.mapScale = {'x': 1, 'y': 1, 'z': 1}
        self.mapTranslate = {'dx': -int(self.axisX / 2) + 5, 'dy': -int(self.axisY / 2) + 5, 'dz': 5}
        self.cameraPosition = self.axisX * 2.5
        self.init_3D_plotting()

        # Temp 3D Data
        self.coordinates_x = 0
        self.coordinates_y = 0
        self.depth = 0
        self.timestamp = None

        # Create a timer
        self.timer = QTimer()
        # Set timer timeout callback function
        self.timer.timeout.connect(self.main_viewer)

        # Set click function for camera control
        self.btn_camera_connect.clicked.connect(self.camera_connect)
        self.btn_camera_disconnect.clicked.connect(self.camera_disconnect)
        self.btn_matching.clicked.connect(self.matching_frame_size)

        # Set click function for 3D Visualization
        self.btn_3D_data_path.clicked.connect(self.load_3D_data)
        self.btn_load_visualization.clicked.connect(self.load_visualization)

        # Set click function for Fish Detection
        self.btn_model_path.clicked.connect(self.set_model)
        self.btn_label_path.clicked.connect(self.set_label)
        self.btn_load_object_detection.clicked.connect(self.load_object_detection)

        # Set date/time edit
        self.dateTimeEdit_start.setDisplayFormat("yy/MM/dd hh:mm:ss")
        self.dateTimeEdit_start.setDateTime(QDateTime(2019, 1, 1, 0, 0, 0))
        # self.dateTimeEdit_start.setDateTime(QDateTime.currentDateTime())
        self.dateTimeEdit_end.setDisplayFormat("yy/MM/dd hh:mm:ss")
        self.dateTimeEdit_end.setDateTime(QDateTime.currentDateTime())

        # Set model for fish detection
        self.model = 'rfcn_resnet101_aquarium_fish_v2_22751'
        self.label_model_path.setText("  " + self.model)
        # self.label_model_path.setFont(QFont('Arial', 10))
        self.label = 'aquarium_fish_v2_label_map.pbtxt'
        self.label_label_path.setText("  " + self.label)
        self.num_classes = 3
        self.spinBox_num_class.setValue(self.num_classes)
        self.spinBox_num_class.valueChanged.connect(self.set_num_classes)

        # Define class
        self.object_detection = ObjectDetection(model=self.model,
                                                labels=self.label,
                                                num_classes=self.num_classes)
        self.frameRate = FrameRate()
        self.ABDetection = AbnormalBehaviorDetection()
        self.vis3D = Visualization3D()

        # Default input size of camera
        self.lCamWidth = 1280   # X
        self.lCamHeight = 720   # Y
        self.rCamWidth = 1280   # Depth
        self.rCamHeight = 720

        # Resize frame to match the size of the fish-tank
        self.check_calibration = False
        self.lCropUpper, self.rCropUpper = 0, 0
        self.lCropBottom, self.rCropBottom = 0, 0
        self.lCropLeft, self.rCropLeft = 0, 0
        self.lCropRight, self.rCropRight = 0, 0

        # Frame time variable
        self.prevTime = 0

        # Init speed data
        self.speed = 0

        # Counting for Save coordinates(x, y)
        self.countFrame = 0
        self.checkFrame = 0
        self.endTime = datetime.now() + timedelta(days=1)
        self.fileNum = str(datetime.now().year) + "%02d" %datetime.now().month + "%02d" %datetime.now().day

        # Queue for pattern-finding
        # TODO: Try change data length of dir(latest_behavior_pattern).
        self.patternArr_size = 6 * (60) * (30)  # Input minutes, ex) (30)min == 6fps * (60)sec * (30)
                                                # Testing,       ex) 10sec = 6fps * (10) * (1)
        self.patternArr = [] * self.patternArr_size

        # TODO: Try change data length for check_behavior_pattern_2st.
        self.queue_size_of_speed = (60) * (30)  # Input minutes, ex) (30)min == (60)sec * (30)
                                                # Testing,       ex) 10sec = (10) * (1)
        self.queue_of_speed = [] * self.queue_size_of_speed


    def main_viewer(self):
        try:
            # Read frame
            ret, leftFrame = self.leftCam.read()
            ret, rightFrame = self.rightCam.read()

            # Frame rate
            curTime = time()
            sec = curTime - self.prevTime
            self.prevTime = curTime
            fps = "FPS %0.1f" % (1 / (sec))

            # Show text in upper left corner
            self.frameRate.putText(frame=leftFrame, text='Front Camera ' + fps)
            self.frameRate.putText(frame=rightFrame, text='Side Camera ' + fps)

            # RGB format
            leftFrame = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2RGB)
            rightFrame = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2RGB)

            # Resize frame to match the size of the fish-tank
            matchLeftFrame = leftFrame[
                              self.lCropUpper : self.lCamHeight - self.lCropBottom,
                              self.lCropLeft : self.lCamWidth - self.lCropRight]
            matchRightFrame = rightFrame[
                               self.rCropUpper : self.rCamHeight - self.rCropBottom,
                               self.rCropLeft : self.rCamWidth - self.rCropRight]

            # Run fish detectorf
            l_boxes, l_scores, l_classes, l_category_index = self.object_detection.run(
                image_np=matchLeftFrame,
                display=True)
            r_boxes, r_scores, r_classes, r_category_index = self.object_detection.run(
                image_np=matchRightFrame,
                display=True)

            # Data processing
            _, l_object_point, l_x_min, l_y_min, l_x_max, l_y_max, _ = self.object_detection.data_processing(
                image_np=leftFrame,
                boxes=l_boxes,
                scores=l_scores,
                classes=l_classes,
                category_index=l_category_index,
                point_buff=self.object_detection.leftCam_point_buff)
            _, r_object_point, r_x_min, r_y_min, r_x_max, r_y_max, _ = self.object_detection.data_processing(
                image_np=rightFrame,
                boxes=r_boxes,
                scores=r_scores,
                classes=r_classes,
                category_index=r_category_index,
                point_buff=self.object_detection.rightCam_point_buff)

            try:
                '''
                Method 1
                    3D positions (x, y and depth) were assigned combining
                    the horizontal coordinates of the left camera (coordinates_x and coordinates_y) and
                    the vertical coordinate of the right camera (depth using x-axis)
                    l_object_point[0]: coordinates_x
                    l_object_point[1]: coordinates_y
                    r_object_point[0]: depth
                '''
                timestamp = time()
                dataValue = [
                    (l_object_point[0],
                     l_object_point[1],
                     r_object_point[0],
                     timestamp,
                     self.lCamWidth,
                     self.lCamHeight,
                     self.rCamWidth,
                     self.rCamHeight)
                ]
                # Save in the last 30minutes behavior pattern data
                self.patternArr.append(dataValue)
                self.patternArr = self.patternArr[-self.patternArr_size:]

            # When it can not be detected even from one frame
            except Exception as e:
                print("[error code] not detected\n", e)
                pass

            # ============================================================================
            #   1st pattern : If object stay on the edge of the screen for a long time
            #   and abnormal behavior detection area
            # ============================================================================
            try:
                # TODO: Try change data length of abnormal behavior point.
                abnormal_behavior_size_x = 120
                abnormal_behavior_size_y = 120
                RangeOfABD = RangeOfAbnormalBehaviorDetection(
                    leftFrame=leftFrame,
                    rightFrame=rightFrame,
                    range_x=abnormal_behavior_size_x,
                    range_y=abnormal_behavior_size_y,
                    leftCam_object_point=l_object_point,
                    rightCam_object_point=r_object_point)
            except Exception as e:
                print("[error code] init RangeOfABD\n", e)
                pass

            try:
                # Range of Abnormal Behavior Detection
                RangeOfABD.line2()

                pattern1st_Arr = self.patternArr
                check_abnormal_behavior_list = []

                # The front of the Smart-Aquarium
                # Left & Upper
                check_abnormal_behavior = self.ABDetection.pattern_1st(
                    x1=0, x2=abnormal_behavior_size_x,
                    y1=0, y2=abnormal_behavior_size_y,
                    z1=0, z2=abnormal_behavior_size_x,
                    patternArr_size=self.patternArr_size,
                    patternArr=pattern1st_Arr)
                check_abnormal_behavior_list.append(check_abnormal_behavior)

                # Right & Upper
                check_abnormal_behavior = self.ABDetection.pattern_1st(
                    x1=self.lCamWidth - abnormal_behavior_size_x, x2=self.lCamWidth,
                    y1=0, y2=abnormal_behavior_size_y,
                    z1=0, z2=abnormal_behavior_size_x,
                    patternArr_size=self.patternArr_size,
                    patternArr=pattern1st_Arr)
                check_abnormal_behavior_list.append(check_abnormal_behavior)

                # Left & Bottom
                check_abnormal_behavior = self.ABDetection.pattern_1st(
                    x1=0, x2=abnormal_behavior_size_x,
                    y1=self.lCamHeight - abnormal_behavior_size_y, y2=self.lCamHeight,
                    z1=0, z2=abnormal_behavior_size_x,
                    patternArr_size=self.patternArr_size,
                    patternArr=pattern1st_Arr)
                check_abnormal_behavior_list.append(check_abnormal_behavior)

                # Right & Bottom
                check_abnormal_behavior = self.ABDetection.pattern_1st(
                    x1=self.lCamWidth - abnormal_behavior_size_x, x2=self.lCamWidth,
                    y1=self.lCamHeight - abnormal_behavior_size_y, y2=self.lCamHeight,
                    z1=0, z2=abnormal_behavior_size_x,
                    patternArr_size=self.patternArr_size,
                    patternArr=pattern1st_Arr)
                check_abnormal_behavior_list.append(check_abnormal_behavior)

                # The back of the Smart-Aquarium
                # Left & Upper
                check_abnormal_behavior = self.ABDetection.pattern_1st(
                    x1=0, x2=abnormal_behavior_size_x,
                    y1=0, y2=abnormal_behavior_size_y,
                    z1=self.lCamWidth - abnormal_behavior_size_x, z2=self.lCamWidth,
                    patternArr_size=self.patternArr_size,
                    patternArr=pattern1st_Arr)
                check_abnormal_behavior_list.append(check_abnormal_behavior)

                # Right & Upper
                check_abnormal_behavior = self.ABDetection.pattern_1st(
                    x1=self.lCamWidth - abnormal_behavior_size_x, x2=self.lCamWidth,
                    y1=0, y2=abnormal_behavior_size_y,
                    z1=abnormal_behavior_size_x, z2=self.lCamWidth,
                    patternArr_size=self.patternArr_size,
                    patternArr=pattern1st_Arr)
                check_abnormal_behavior_list.append(check_abnormal_behavior)

                # Left & Bottom
                check_abnormal_behavior = self.ABDetection.pattern_1st(
                    x1=0, x2=abnormal_behavior_size_x,
                    y1=self.lCamHeight - abnormal_behavior_size_y, y2=self.lCamHeight,
                    z1=self.lCamWidth - abnormal_behavior_size_x, z2=self.lCamWidth,
                    patternArr_size=self.patternArr_size,
                    patternArr=pattern1st_Arr)
                check_abnormal_behavior_list.append(check_abnormal_behavior)

                # Right & Bottom
                check_abnormal_behavior = self.ABDetection.pattern_1st(
                    x1=self.lCamWidth - abnormal_behavior_size_x, x2=self.lCamWidth,
                    y1=self.lCamHeight - abnormal_behavior_size_y, y2=self.lCamHeight,
                    z1=self.lCamWidth - abnormal_behavior_size_x, z2=self.lCamWidth,
                    patternArr_size=self.patternArr_size,
                    patternArr=pattern1st_Arr)
                check_abnormal_behavior_list.append(check_abnormal_behavior)

                for i in check_abnormal_behavior_list:
                    if i == 'Detect abnormal behavior':
                        pattern1st_Arr.clear()
                        self.ABDetection.display(
                            num_pattern=1,
                            leftFrame=leftFrame,
                            rightFrame=rightFrame,
                            leftCam_w=self.lCamWidth,
                            leftCam_h=self.lCamHeight,
                            rightCam_w=self.rCamWidth,
                            rightCam_h=self.rCamHeight)
                check_abnormal_behavior_list.clear()

            # When it can not be detected even from one frame
            except Exception as e:
                print("[error code] 1st pattern\n", e)
                # Range of Abnormal Behavior Detection
                RangeOfABD.line()
                pass

            # ==================================================================
            #   2st pattern : If the movement is noticeably slower or faster
            # ==================================================================
            try:
                pattern2st_Arr = self.patternArr
                for i in range(len(pattern2st_Arr)):
                    '''
                        pattern2st_Arr[i][0][0], coordinates_x
                        pattern2st_Arr[i][0][1], coordinates_y
                        pattern2st_Arr[i][0][2], depth
                        pattern2st_Arr[i][0][3], timestamp
                    '''
                    lastData = len(pattern2st_Arr) - 1
                    oneSecondPreviousData = lastData - int(float("{0:.1f}".format(1 / sec)))
                    speed = self.ABDetection.speed_of_three_dimensional(
                        resolution_x=self.lCamWidth,
                        resolution_y=self.lCamHeight,
                        resolution_z=self.rCamWidth,
                        coordinates_x1=(pattern2st_Arr[oneSecondPreviousData][0])[0],
                        coordinates_x2=(pattern2st_Arr[lastData][0])[0],
                        coordinates_y1=(pattern2st_Arr[oneSecondPreviousData][0])[1],
                        coordinates_y2=(pattern2st_Arr[lastData][0])[1],
                        depth1=(pattern2st_Arr[oneSecondPreviousData][0])[2],
                        depth2=(pattern2st_Arr[lastData][0])[2],
                        time1=(pattern2st_Arr[oneSecondPreviousData][0])[3],
                        time2=(pattern2st_Arr[lastData][0])[3])
                    self.speed = speed
                cv2.putText(leftFrame, '{0:.2f}mm/s'.format(speed), (l_x_max, l_y_max), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                cv2.putText(rightFrame, '{0:.2f}mm/s'.format(speed), (r_x_max, r_y_max), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

                if self.countFrame == 0:
                    check_abnormal_behavior = self.ABDetection.pattern_2st(
                        speed=speed,
                        queue_size_of_speed=self.queue_size_of_speed,
                        queue_of_speed=self.queue_of_speed)
                    if check_abnormal_behavior == 'Detect abnormal behavior':
                        self.queue_of_speed.clear()
                        self.ABDetection.display(
                            num_pattern=2,
                            leftFrame=leftFrame,
                            rightFrame=rightFrame,
                            leftCam_w=self.lCamWidth,
                            leftCam_h=self.lCamHeight,
                            rightCam_w=self.rCamWidth,
                            rightCam_h=self.rCamHeight)

            # When it can not be detected even from one frame
            except Exception as e:
                print("[error code] 2st pattern\n", e)
                pass

            # ================================================
            #   3st pattern : If detect white spot disease
            # ================================================
            try:
                leftCam_check_white_spot_disease = ''
                rightCam_check_white_spot_disease = ''

                if (l_x_min != None) and (l_x_max != None) and (l_x_min != None) and (l_y_max != None):
                    cropLeft_object = leftFrame[l_y_min:l_y_max, l_x_min:l_x_max]
                    h = l_y_max - l_y_min
                    w = l_x_max - l_x_min
                    resizeLeft = cv2.resize(cropLeft_object, (w * 2, h * 2))
                    leftCam_check_white_spot_disease = self.ABDetection.pattern_3st(
                        frame=resizeLeft,
                        title='frontCam')

                if (r_x_min != None) and (r_x_max != None) and (r_y_min != None) and (r_y_max != None):
                    cropRight_object = rightFrame[r_y_min:r_y_max, r_x_min:r_x_max]
                    h = r_y_max - r_y_min
                    w = r_x_max - r_x_min
                    resizeRight = cv2.resize(cropRight_object, (w * 2, h * 2))
                    rightCam_check_white_spot_disease = self.ABDetection.pattern_3st(
                        frame=resizeRight,
                        title='sideCam')

                if (leftCam_check_white_spot_disease == 'Detect white spot disease') or (rightCam_check_white_spot_disease == 'Detect white spot disease'):
                    self.ABDetection.display(
                        num_pattern=3,
                        leftFrame=leftFrame,
                        rightFrame=rightFrame,
                        leftCam_w=self.lCamWidth,
                        leftCam_h=self.lCamHeight,
                        rightCam_w=self.rCamWidth,
                        rightCam_h=self.rCamHeight)

            # When it can not be detected even from one frame
            except Exception as e:
                print("[error code] 3st pattern\n", e)
                pass

            self.countFrame += 1
            if self.object_detection.model[:16] == 'ssd_inception_v2':
                checkFrame = int(float("{0:.1f}".format(1 / sec)))
            elif self.object_detection.model[:14] == 'rfcn_resnet101':
                checkFrame = int(float("{0:.1f}".format(1 / sec)))
            if self.countFrame > checkFrame:
                self.countFrame = 0
            if datetime.now() >= self.endTime:
                self.fileNum += 1
                self.endTime = datetime.now() + timedelta(days=1)

            try:
                # Save data to a csv file
                current_time = datetime.now()
                year = current_time.year
                month = current_time.month
                day = current_time.day
                hour = current_time.hour
                minute = current_time.minute
                second = current_time.second
                _timestamp = '{}.{}.{}.{}.{}.{}'.format(year, month, day, hour, minute, second)
                speed =  "%0.4f" % (self.speed)
                saveValue = [
                    (l_object_point[0],
                     l_object_point[1],
                     r_object_point[0],
                     speed,
                     _timestamp,
                     self.lCamWidth,
                     self.lCamHeight,
                     self.rCamWidth,
                     self.rCamHeight)
                ]
                column_name = ['coordinates_x', 
                               'coordinates_y', 
                               'depth', 
                               'speed', 
                               'timestamp', 
                               'frontCam_w', 
                               'frontCam_h', 
                               'sideCam_w', 
                               'sideCam_h']
                save_pattern_dir = 'save_pattern'
                if not os.path.isdir(save_pattern_dir):
                    os.mkdir(save_pattern_dir)
                save_file = save_pattern_dir + '/{}.csv'.format(str(self.fileNum))
                if os.path.exists(save_file) == True:
                    with open(save_file, 'a') as f:
                        xml_df = pd.DataFrame(saveValue, columns=column_name)
                        xml_df.to_csv(f, header=False, index=None)
                else:
                    xml_df = pd.DataFrame(columns=column_name)
                    xml_df.to_csv(save_file, index=None)
                    with open(save_file, 'a') as f:
                        xml_df = pd.DataFrame(saveValue, columns=column_name)
                        xml_df.to_csv(f, header=False, index=None)

            except Exception as e:
                print("[error code] save data\n", e)
                pass

            # Fix frame size (640, 480)
            fixLeftFrame = cv2.resize(matchLeftFrame, (640, 480))
            fixRightFrame = cv2.resize(matchRightFrame, (640, 480))
            # Get image info
            height, width, channel = fixLeftFrame.shape
            step = channel * width
            height2, width2, channel2 = fixRightFrame.shape
            step2 = channel2 * width2
            # Create QImage from image
            qImg = QImage(fixLeftFrame.data,
                          width, height, step,
                          QImage.Format_RGB888)
            qImg2 = QImage(fixRightFrame.data,
                          width2, height2, step2,
                          QImage.Format_RGB888)
            # Show image in camera_widget
            self.widget_camera.setPixmap(QPixmap.fromImage(qImg))
            self.widget_camera_2.setPixmap(QPixmap.fromImage(qImg2))

        except Exception as e:
            print("[main_viewer]\n", e)
            pass


    def camera_connect(self):
        if not self.timer.isActive():
            # Create camera capture
            self.leftCam = cv2.VideoCapture('frontCam.avi')
            self.rightCam = cv2.VideoCapture('sideCam.avi')

            # Resize frame to match the size of the fish-tank
            if self.check_calibration == True:
                cropWidth = 960
                camera_calibration = CameraCalibration(cap1=self.leftCam,
                                                       cap2=self.rightCam,
                                                       camWidth=self.lCamWidth,
                                                       camHeight=self.lCamHeight,
                                                       cropWidth=cropWidth)
                camera_calibration.set_stereoMatcher()
                '''
                    The distortion in the leftCam and rightCam edges prevents a good calibration,
                    so discard the edges
                '''
                camera_calibration.set_resolution(cap=self.leftCam)
                camera_calibration.set_resolution(cap=self.rightCam)
            elif self.check_calibration == False:
                self.lCamWidth = int(self.leftCam.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.lCamHeight = int(self.leftCam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.rCamWidth = int(self.rightCam.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.rCamHeight = int(self.rightCam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.lCamWidth = self.lCamWidth - (self.lCropLeft + self.lCropRight)
            self.lCamHeight = self.lCamHeight - (self.lCropUpper + self.lCropBottom)
            self.rCamWidth = self.rCamWidth - (self.rCropLeft + self.rCropRight)
            self.rCamHeight = self.rCamHeight - (self.rCropUpper + self.rCropBottom)

            # Start timer
            self.timer.start(2)
            # Update btn_camera_connect text
            self.btn_camera_connect.setText("Running . . .")


    def camera_disconnect(self):
        if self.timer.isActive():
            self.widget_camera.setPixmap(QPixmap.fromImage(QImage()))
            self.widget_camera_2.setPixmap(QPixmap.fromImage(QImage()))
            # Stop timer
            self.timer.stop()
            # Release camera capture
            self.leftCam.release()
            self.rightCam.release()
            self.btn_camera_connect.setText("Connect")


    def matching_frame_size(self):
        try:
            self.lCropUpper = int(self.textEdit_cropU.toPlainText())
            self.lCropBottom = int(self.textEdit_cropB.toPlainText())
            self.lCropLeft = int(self.textEdit_cropL.toPlainText())
            self.lCropRight = int(self.textEdit_cropR.toPlainText())
            self.rCropUpper = int(self.textEdit_cropU_2.toPlainText())
            self.rCropBottom = int(self.textEdit_cropB_2.toPlainText())
            self.rCropLeft = int(self.textEdit_cropL_2.toPlainText())
            self.rCropRight = int(self.textEdit_cropR_2.toPlainText())
        except:
            pass


    def set_model(self):
        try:
            modelPath = QFileDialog.getOpenFileName(None,
                                                    caption='Load Your Model',
                                                    directory='./object_detection/saved_models')
            modelName = modelPath[0].split('/')[-2]
            fileFormat = modelPath[0].split('/')[-1].split('.')[-1]
            if fileFormat == 'pb':
                self.model = modelName
                self.label_model_path.setText(self.model)
                self.label_model_path.setText('  ' + self.model)
            elif fileFormat == '':
                pass
            else:
                QMessageBox.about(None, "Error", "Please select a model.")

        except Exception as e:
            print("[set_model] \n", e)
            pass


    def set_label(self):
        try:
            labelPath = QFileDialog.getOpenFileName(None,
                                                    caption='Load Your Label',
                                                    directory='./object_detection/data')
            labelName = labelPath[0].split('/')[-1]
            fileFormat = labelName.split('.')[-1]
            if fileFormat == 'pbtxt':
                self.label = labelName
                self.label_label_path.setText(self.label)
                self.label_label_path.setText("  " + self.label)
            elif fileFormat == '':
                pass
            else:
                QMessageBox.about(None, "Error", "Please select a label file.")

        except Exception as e:
            print("[set_label] \n", e)
            pass


    def set_num_classes(self):
        self.num_classes = self.spinBox_num_class.value()


    def load_object_detection(self):
        self.object_detection = ObjectDetection(model=self.model,
                                                labels=self.label,
                                                num_classes=self.num_classes)


    def init_3D_plotting(self):
        try:
            self.graphicsView.show()
            self.graphicsView.setBackgroundColor('k')
            self.graphicsView.setCameraPosition(distance=self.cameraPosition,
                                                elevation=None,
                                                azimuth=None)

            # Add a grid to the view
            glg = gl.GLGridItem()
            glg.scale(x=self.initScale['x'],
                      y=self.initScale['y'],
                      z=self.initScale['z'])
            glg.setDepthValue(10) # draw grid after surfaces since they may be translucent
            self.graphicsView.addItem(glg)

            # 3D Plotting
            rectNumDataX = int(self.axisX * 4)
            rectNumDataY = int(self.axisY * 4)
            rectNumDataZ = int(self.axisZ * 4)
            rectNumData = rectNumDataX+rectNumDataY+rectNumDataZ
            rectColorWhite = (1.0, 1.0, 1.0, 0.1)
            rectColorRed = (1.0, 0.0, 0.0, 0.1)
            pos = np.empty((rectNumData, 3))
            size = np.empty((rectNumData))
            color = np.empty((rectNumData, 4))
            numDataX = int(self.axisX)
            numDataY = int(self.axisY)
            numDataZ = int(self.axisZ)
            tempNumData = 0
            for i in range(numDataX):
                pos[i] = (i, 0, 0); size[i] = 10; color[i] =  rectColorRed
            tempNumData += numDataX
            for i in range(numDataY):
                pos[i+tempNumData] = (0, i, 0); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorRed
            tempNumData += numDataY
            for i in range(numDataZ):
                pos[i+tempNumData] = (0, 0, i); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorRed
            tempNumData += numDataZ
            for i in range(numDataY):
                pos[i+tempNumData] = (self.axisX, i, 0); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorWhite
            tempNumData += numDataY
            for i in range(numDataZ):
                pos[i+tempNumData] = (self.axisX, 0, i); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorWhite
            tempNumData += numDataZ
            for i in range(numDataX):
                pos[i+tempNumData] = (i, self.axisY, 0); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorWhite
            tempNumData += numDataX
            for i in range(numDataZ):
                pos[i+tempNumData] = (0, self.axisY, i); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorWhite
            tempNumData += numDataZ
            for i in range(numDataX):
                pos[i+tempNumData] = (i, 0, self.axisZ); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorWhite
            tempNumData += numDataX
            for i in range(numDataY):
                pos[i+tempNumData] = (0, i, self.axisZ); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorWhite
            tempNumData += numDataY
            for i in range(numDataZ):
                pos[i+tempNumData] = (self.axisX, self.axisY, i); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorWhite
            tempNumData += numDataZ
            for i in range(numDataY):
                pos[i+tempNumData] = (self.axisX, i, self.axisZ); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorWhite
            tempNumData += numDataY
            for i in range(numDataX):
                pos[i+tempNumData] = (i, self.axisY, self.axisZ); size[i+tempNumData] = 10; color[i+tempNumData] = rectColorWhite

            gsp = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
            gsp.scale(
                x=self.mapScale['x'],
                y=self.mapScale['y'],
                z=self.mapScale['z'])
            gsp.translate(dx=self.mapTranslate['dx'],
                dy=self.mapTranslate['dy'],
                dz=self.mapTranslate['dz'])
            self.graphicsView.addItem(gsp)

        except Exception as e:
            print("[init_3D_surface] \n", e)
            pass


    def set_3D_plotting(self, dataPath):
        self.vis3D.read_data(data_path=dataPath)

        start_time = self.vis3D.timestamp[0]
        start_time = start_time.split('.')
        end_time = self.vis3D.timestamp[-1]
        end_time = end_time.split('.')
        self.dateTimeEdit_start.setDateTime(QDateTime(
            int(start_time[0]),
            int(start_time[1]),
            int(start_time[2]),
            int(start_time[3]),
            int(start_time[4]),
            int(start_time[5])))
        self.dateTimeEdit_end.setDateTime(QDateTime(
            int(end_time[0]),
            int(end_time[1]),
            int(end_time[2]),
            int(end_time[3]),
            int(end_time[4]),
            int(end_time[5])))

        # 3D Plotting
        self.axisX = int(self.vis3D.frontCam_w[0])
        self.axisY = int(self.vis3D.sideCam_w[0])
        self.axisZ = float(self.vis3D.frontCam_h[0])

        self.update_3D_plotting(
            x=self.vis3D.coordinates_x,
            y=self.vis3D.coordinates_y,
            depth=self.vis3D.depth)


    def update_3D_plotting(self, x, y, depth):
        numData = len(x)
        pos = np.empty((numData, 3))
        size = np.empty((numData))
        color = np.empty((numData, 4))

        for i in range(numData):
            # Real(3D Visualization) to OpenGL(3D Visualization)
            # pos[i] = (x[i],
            #           self.axisY - y[i],
            #           self.axisZ - depth[i])
            pos[i] = (self.axisX - depth[i],
                      x[i],
                      self.axisZ - y[i])
            size[i] = 10
            color[i] = (0.0, 1.0, 0.0, 1.0)

        gsp = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
        gsp.scale(x=self.mapScale['x'],
                  y=self.mapScale['y'],
                  z=self.mapScale['z'])
        gsp.translate(dx=self.mapTranslate['dx'],
                      dy=self.mapTranslate['dy'],
                      dz=self.mapTranslate['dz'])
        self.graphicsView.addItem(gsp)


    def load_3D_data(self):
        try:
            dataPath = QFileDialog.getOpenFileName(None,
                                                   caption='Load Your Data',
                                                   directory='./save_pattern')
            dataName = dataPath[0].split('/')[-1]
            self.label_3D_data_path.setText('  ' + dataName)
            fileFormat = dataName.split('.')[-1]
            if fileFormat == 'csv':
                self.set_3D_plotting(dataPath[0])
                self.set_data_table()
            elif fileFormat == '':
                pass
            else:
                QMessageBox.about(None, "Error", "Please select a csv file.")

        except Exception as e:
            print("[load_3D_data] \n", e)
            pass


    def load_visualization(self, dataPath):
        try:
            if self.label_3D_data_path.text() == '  ' + 'Choose data':
                QMessageBox.about(None, "Error", "Please select a csv file.")
            elif self.label_3D_data_path.text() == '  ':
                QMessageBox.about(None, "Error", "Please select a csv file.")

            del self.graphicsView.items[2:]

            # test_time = self.dateTimeEdit_start.dateTime().toPyDateTime()
            start_time = self.dateTimeEdit_start.dateTime()
            start_time = str(start_time.toString('yy.MM.dd.hh.mm.ss'))
            end_time = self.dateTimeEdit_end.dateTime()
            end_time = str(end_time.toString('yy.MM.dd.hh.mm.ss'))
            list_x, list_y, list_depth = self.vis3D.set_time_zone(
                start_time=start_time,
                end_time=end_time,
                timestamp=self.vis3D.timestamp,
                x=self.vis3D.coordinates_x,
                y=self.vis3D.coordinates_y,
                depth=self.vis3D.depth)

            self.update_3D_plotting(
                x=list_x,
                y=list_y,
                depth=list_depth)

        except Exception as e:
            print("[load_visualization] \n", e)
            pass


    def set_data_table(self):
        column_headers = ['x\n(frontW)', 'y\n(frontH)', 'depth\n(sideW)',
                          'speed\n(mm/s)', 'timestamp\n(y.M.d.h.m.s)',
                          'frontCam_w', 'frontCam_h', 'sideCam_w', 'sideCam_h']
        coordinates_x = self.vis3D.coordinates_x
        coordinates_y = self.vis3D.coordinates_y
        depth = self.vis3D.depth
        speed = self.vis3D.speed
        timestamp = self.vis3D.timestamp
        frontCam_w = self.vis3D.frontCam_w
        frontCam_h = self.vis3D.frontCam_h
        sideCam_w = self.vis3D.sideCam_w
        sideCam_h = self.vis3D.sideCam_h

        self.tableWidget_dataList.setRowCount(len(coordinates_x))
        self.tableWidget_dataList.setColumnCount(len(column_headers))
        self.tableWidget_dataList.setHorizontalHeaderLabels(column_headers)

        for idx in range(len(coordinates_x)):
            self.tableWidget_dataList.setItem(idx, 0, QTableWidgetItem(str(coordinates_x[idx])))
            self.tableWidget_dataList.setItem(idx, 1, QTableWidgetItem(str(coordinates_y[idx])))
            self.tableWidget_dataList.setItem(idx, 2, QTableWidgetItem(str(depth[idx])))
            self.tableWidget_dataList.setItem(idx, 3, QTableWidgetItem(str(speed[idx])))
            self.tableWidget_dataList.setItem(idx, 4, QTableWidgetItem(str(timestamp[idx])))
            self.tableWidget_dataList.setItem(idx, 5, QTableWidgetItem(str(frontCam_w[idx])))
            self.tableWidget_dataList.setItem(idx, 6, QTableWidgetItem(str(frontCam_h[idx])))
            self.tableWidget_dataList.setItem(idx, 7, QTableWidgetItem(str(sideCam_w[idx])))
            self.tableWidget_dataList.setItem(idx, 8, QTableWidgetItem(str(sideCam_h[idx])))

        self.tableWidget_dataList.resizeColumnsToContents()
        self.tableWidget_dataList.resizeRowsToContents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
