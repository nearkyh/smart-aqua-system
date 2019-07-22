import numpy as np
import cv2
import pandas as pd
import os
import sys
from time import time
from datetime import datetime, timedelta

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QAction
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtCore import QTimer, QDateTime, QRect
from PyQt5 import uic

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from utils.object_detection import ObjectDetection
from utils.visualization_3D import Visualization3D
from utils.frame_rate import FrameRate
from utils.video_recorder import VideoRecorder
from utils.camera_calibration import CameraCalibration
from utils.abnormal_behavior_detection import AbnormalBehaviorDetection
from utils.abnormal_behavior_detection import RangeOfAbnormalBehaviorDetection
from utils.db_connector import DBConnector


form_class = uic.loadUiType("main_window.ui")[0]


class MyWindow(QMainWindow, form_class):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Smart Aqua System")

        # 관상어 이동패턴에 대해 3D 시각화를 위한 위젯 생성
        self.graphicsView = gl.GLViewWidget(self.tab_visualization)
        self.graphicsView.setGeometry(QRect(260, 40, 500, 500))
        self.graphicsView.setObjectName("graphicsView")

        # 3D Scatter 변수 초기화
        self.axisX = 640
        self.axisY = 640
        self.axisZ = 480.0
        self.initScale = {'x': int(self.axisX / 20), 'y': int(self.axisY / 20), 'z': 10}
        self.mapScale = {'x': 1, 'y': 1, 'z': 1}
        self.mapTranslate = {'dx': -int(self.axisX / 2) + 5, 'dy': -int(self.axisY / 2) + 5, 'dz': 5}
        self.cameraPosition = self.axisX * 2.5
        self.init_3D_scatter()
        # 3D Scatter 입력 데이터 초기화
        self.pos = None
        self.scatter_size = None
        self.scatter_color = None
        self.line_size = None
        self.line_color = None

        # 타이머 생성
        self.timer = QTimer()
        # 타임아웃 콜백 설정
        self.timer.timeout.connect(self.detector)

        # 카메라 컨트롤 기능 설정
        self.frontCamIndex = 2
        self.sideCamIndex = 5
        self.btn_camera_connect.clicked.connect(self.camera_connect)
        self.btn_camera_disconnect.clicked.connect(self.camera_disconnect)
        self.btn_matching.clicked.connect(self.matching_frame_size)

        # 녹화 기능 설정
        self.btn_rec.clicked.connect(self.rec)
        self.btn_rec_stop.clicked.connect(self.rec_stop)

        # 비디오 컨트롤 기능 설정
        self.front_video = ''
        self.side_video = ''
        self.btn_front_video.clicked.connect(self.set_front_video)
        self.btn_side_video.clicked.connect(self.set_side_video)
        self.btn_video_connect.clicked.connect(self.video_connect)
        self.btn_video_disconnect.clicked.connect(self.video_disconnect)

        # 3D 시각화를 위한 기능 설정
        self.btn_load_csv_file.clicked.connect(self.load_csv_file)
        self.btn_load_section.clicked.connect(self.load_section)
        # 1) 한개의 특정 데이터(관상어 이동좌표)만 시각화를 하기 위한 기능 설정
        self.tableWidget_dataList.cellClicked.connect(self.clicked_data)
        self.check_row = None
        # 2) 여러개의 특정 데이터(관상어 이동좌표)를 시각화 하기 위한 기능 설정
        self.tableWidget_dataList.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.select_action = QAction("select", self.tableWidget_dataList)
        self.tableWidget_dataList.addAction(self.select_action)
        self.select_action.triggered.connect(self.selected_data)
        # 3) 1, 2번의 시각화 기능 리셋
        self.reset_action = QAction("reset", self.tableWidget_dataList)
        self.tableWidget_dataList.addAction(self.reset_action)
        self.reset_action.triggered.connect(self.selected_data_reset)
        # Smooth 적용 (Scatter Plot --> Line Plot)
        self.smooth_mode = 'scatter'    # Available modes('scatter', 'line')
        self.btn_apply_line.clicked.connect(self.apply_line)
        self.btn_apply_scatter.clicked.connect(self.apply_scatter)
        self.btn_apply_reset.clicked.connect(self.apply_reset)

        # 3D 시각화 이미지 캡쳐
        self.btn_capture.clicked.connect(self.capture_vis)

        # 관상어 인식을 위한 모델 및 라벨, 개체수에 대한 설정 기능
        self.btn_model_path.clicked.connect(self.set_model)
        self.btn_label_path.clicked.connect(self.set_label)
        self.btn_load_object_detection.clicked.connect(self.load_object_detection)

        # 시간/날짜 에디터 설정 기능
        self.dateTimeEdit_start.setDisplayFormat("yy/MM/dd hh:mm:ss")
        self.dateTimeEdit_start.setDateTime(QDateTime(2019, 1, 1, 0, 0, 0))
        # self.dateTimeEdit_start.setDateTime(QDateTime.currentDateTime())
        self.dateTimeEdit_end.setDisplayFormat("yy/MM/dd hh:mm:ss")
        self.dateTimeEdit_end.setDateTime(QDateTime.currentDateTime())

        # 관상어 인식 모델 및 라벨, 개체수에 대한 변수 초기화
        self.model = 'rfcn_resnet101_angelfish_40000'
        self.label_model_path.setText("  " + self.model)
        # self.label_model_path.setFont(QFont('Arial', 10))
        self.label = 'angelfish_label_map.pbtxt'
        self.label_label_path.setText("  " + self.label)
        self.num_classes = 1
        self.spinBox_num_class.setValue(self.num_classes)
        self.spinBox_num_class.valueChanged.connect(self.set_num_classes)

        # 임포트한 클래스 초기화
        self.object_detection = ObjectDetection(model=self.model,
                                                labels=self.label,
                                                num_classes=self.num_classes)
        self.frameRate = FrameRate()
        self.ABDetection = AbnormalBehaviorDetection()
        self.vis3D = Visualization3D()

        # 카메라 입력 사이즈 초기화
        self.lCamWidth = 1280   # X
        self.lCamHeight = 720   # Y
        self.rCamWidth = 1280   # Depth
        self.rCamHeight = 720

        # 비디오 녹화를 위한 변수 초기화
        self.videoRecorder = None
        self.recording = None

        # 수조 크기에 맞게 카메라 입력 프레임 리사이즈
        self.check_calibration = False
        self.lCropUpper, self.rCropUpper = 0, 0
        self.lCropBottom, self.rCropBottom = 0, 0
        self.lCropLeft, self.rCropLeft = 0, 0
        self.lCropRight, self.rCropRight = 0, 0

        # 프레임 측정을 위한 이전 시간 변수 초기화
        self.prevTime = 0
        # 속도 측정을 위한 변수 초기화
        self.speed = 0
        # 관상어 이동 좌표 저장을 위한 데이터 카운팅
        self.countFrame = 0
        self.checkFrame = 0
        self.endTime = datetime.now() + timedelta(days=1)
        self.fileNum = str(datetime.now().year) + "%02d" %datetime.now().month + "%02d" %datetime.now().day

        # 패턴 분석을 위한 데이터 열(Queue)
        # TODO: Try change data length of dir(latest_behavior_pattern).
        self.patternArr_size = 6 * (60) * (30)  # Input minutes, ex) (30)min == 6fps * (60)sec * (30)
                                                # Testing,       ex) 10sec = 6fps * (10) * (1)
        self.patternArr = [] * self.patternArr_size
        # TODO: Try change data length for check_behavior_pattern_2st.
        self.queue_size_of_speed = (60) * (30)  # Input minutes, ex) (30)min == (60)sec * (30)
                                                # Testing,       ex) 10sec = (10) * (1)
        self.queue_of_speed = [] * self.queue_size_of_speed


    def detector(self):
        try:
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

            # Run fish detector
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

            # Record Video
            if (self.videoRecorder and self.recording) != None:
                rec_frame = cv2.hconcat([fixLeftFrame, fixRightFrame])
                self.videoRecorder.output(frame=rec_frame,
                                          recording=self.recording)

        except Exception as e:
            print("[detector]\n", e)
            pass


    def camera_connect(self):
        try:
            if not self.timer.isActive():
                # Create camera capture
                self.leftCam = cv2.VideoCapture(self.frontCamIndex)
                self.rightCam = cv2.VideoCapture(self.sideCamIndex)

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
                self.btn_camera_connect.setText("Running...")

        except Exception as e:
            print("[camera_connect] \n", e)
            pass


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


    def rec(self):
        self.videoRecorder = VideoRecorder(fileName='video',
                                           width=self.lCamWidth + self.rCamWidth,
                                           height=self.lCamHeight)
        self.recording = self.videoRecorder.recording()
        self.btn_rec.setText("Recording...")


    def rec_stop(self):
        self.VideoRecorder = None
        self.recording = None
        self.btn_rec.setText("REC")


    def video_connect(self):
        try:
            if not self.timer.isActive():
                # Create camera capture
                self.leftCam = cv2.VideoCapture(self.front_video)
                self.rightCam = cv2.VideoCapture(self.side_video)

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
                self.btn_video_connect.setText("Running...")

        except Exception as e:
            print("[video_connect] \n", e)
            pass


    def video_disconnect(self):
        if self.timer.isActive():
            self.widget_camera.setPixmap(QPixmap.fromImage(QImage()))
            self.widget_camera_2.setPixmap(QPixmap.fromImage(QImage()))
            # Stop timer
            self.timer.stop()
            # Release camera capture
            self.leftCam.release()
            self.rightCam.release()
            self.btn_video_connect.setText("Connect")


    def set_front_video(self):
        try:
            videoPath = QFileDialog.getOpenFileName(None,
                                                    caption='Load Your Front Video',
                                                    directory='./')
            videoName = videoPath[0].split('/')[-1]
            fileFormat = videoPath[0].split('/')[-1].split('.')[-1]
            if (fileFormat == 'avi') or (fileFormat == 'mp4'):
                self.front_video = videoName
                self.label_front_video.setText('  ' + videoName)
            elif fileFormat == '':
                pass
            else:
                QMessageBox.about(None, "Error", "Please select a front video.")

        except Exception as e:
            print("[set_model] \n", e)
            pass


    def set_side_video(self):
        try:
            videoPath = QFileDialog.getOpenFileName(None,
                                                    caption='Load Your Side Video',
                                                    directory='./')
            videoName = videoPath[0].split('/')[-1]
            fileFormat = videoPath[0].split('/')[-1].split('.')[-1]
            if (fileFormat == 'avi') or (fileFormat == 'mp4'):
                self.side_video = videoName
                self.label_side_video.setText('  ' + videoName)
            elif fileFormat == '':
                pass
            else:
                QMessageBox.about(None, "Error", "Please select a side video.")

        except Exception as e:
            print("[set_model] \n", e)
            pass


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


    def init_3D_scatter(self):
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

            # 3D Scatter
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
            gsp.scale(x=self.mapScale['x'],
                      y=self.mapScale['y'],
                      z=self.mapScale['z'])
            gsp.translate(dx=self.mapTranslate['dx'],
                          dy=self.mapTranslate['dy'],
                          dz=self.mapTranslate['dz'])
            self.graphicsView.addItem(gsp)

        except Exception as e:
            print("[init_3D_scatter] \n", e)
            pass


    def set_3D_scatter(self, dataPath):
        try:
            self.vis3D.read_data(data_path=dataPath)
            # csv 파일 데이터에 대한 start time 및 end time 적용
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

            # 3D 좌표 사이즈 설정
            self.axisX = int(self.vis3D.frontCam_w[-1])
            self.axisY = int(self.vis3D.sideCam_w[-1])
            self.axisZ = float(self.vis3D.frontCam_h[-1])

            self.load_3D_scatter(x=self.vis3D.coordinates_x,
                                 y=self.vis3D.coordinates_y,
                                 z=self.vis3D.depth)

        except Exception as e:
            print("[set_3D_scatter] \n", e)
            pass


    def load_3D_scatter(self, x, y, z):
        try:
            numData = len(x)
            self.pos = np.empty((numData, 3))
            self.scatter_size = np.empty((numData))
            self.scatter_color = np.empty((numData, 4))
            for i in range(numData):
                # Real(3D 시각화 좌표) to OpenGL(3D 시각화 좌표)
                self.pos[i] = (self.axisX - z[i],
                               x[i],
                               self.axisZ - y[i])
                self.scatter_size[i] = 5
                self.scatter_color[i] = (0.0, 1.0, 0.0, 1.0)

            self.update_3D_scatter(pos=self.pos,
                                   size=self.scatter_size,
                                   color=self.scatter_color)

        except Exception as e:
            print("[load_3D_scatter] \n", e)
            pass


    def update_3D_scatter(self, pos, size, color):
        try:
            del self.graphicsView.items[2:]
            gsp = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
            gsp.scale(x=self.mapScale['x'],
                      y=self.mapScale['y'],
                      z=self.mapScale['z'])
            gsp.translate(dx=self.mapTranslate['dx'],
                          dy=self.mapTranslate['dy'],
                          dz=self.mapTranslate['dz'])
            self.graphicsView.addItem(gsp)

        except Exception as e:
            print("[update_3D_scatter] \n", e)
            pass


    def load_3D_line(self, x, y, z):
        try:
            numData = len(x)
            self.pos = np.empty((numData, 3))
            self.line_size = 2
            self.line_color = (0.0, 1.0, 0.0, 1.0)
            for i in range(numData):
                # Real(3D 시각화 좌표) to OpenGL(3D 시각화 좌표)
                self.pos[i] = (self.axisX - z[i],
                               x[i],
                               self.axisZ - y[i])
            self.update_3D_line(pos=self.pos,
                                width=self.line_size,
                                color=self.line_color)

        except Exception as e:
            print("[load_3D_line] \n", e)
            pass


    def update_3D_line(self, pos, width, color):
        try:
            del self.graphicsView.items[2:]
            gsp = gl.GLLinePlotItem(pos=pos, width=width, color=color, antialias=True)
            gsp.scale(x=self.mapScale['x'],
                      y=self.mapScale['y'],
                      z=self.mapScale['z'])
            gsp.translate(dx=self.mapTranslate['dx'],
                          dy=self.mapTranslate['dy'],
                          dz=self.mapTranslate['dz'])
            self.graphicsView.addItem(gsp)

        except Exception as e:
            print("[update_3D_line] \n", e)
            pass


    def apply_scatter(self):
        # Use scatter plot
        try:
            del self.graphicsView.items[2:]
            self.update_3D_scatter(pos=self.pos,
                                   size=self.scatter_size,
                                   color=self.scatter_color)
            self.smooth_mode = 'scatter'

        except Exception as e:
            print("[apply_scatter] \n", e)
            pass


    def apply_line(self):
        # Use line plot
        try:
            del self.graphicsView.items[2:]
            self.line_size = 2
            self.line_color = (0.0, 1.0, 0.0, 1.0)
            self.update_3D_line(pos=self.pos,
                                width=self.line_size,
                                color=self.line_color)
            self.smooth_mode = 'line'

        except Exception as e:
            print("[apply_line] \n", e)
            pass


    def apply_reset(self):
        try:
            if self.smooth_mode == 'scatter':
                self.load_3D_scatter(x=self.vis3D.coordinates_x,
                                     y=self.vis3D.coordinates_y,
                                     z=self.vis3D.depth)
            elif self.smooth_mode == 'line':
                self.load_3D_line(x=self.vis3D.coordinates_x,
                                  y=self.vis3D.coordinates_y,
                                  z=self.vis3D.depth)
            self.set_data_table(coordinates_x=self.vis3D.coordinates_x,
                                coordinates_y = self.vis3D.coordinates_y,
                                depth=self.vis3D.depth,
                                speed=self.vis3D.speed,
                                timestamp=self.vis3D.timestamp,
                                frontCam_w=self.vis3D.frontCam_w,
                                frontCam_h=self.vis3D.frontCam_h,
                                sideCam_w=self.vis3D.sideCam_w,
                                sideCam_h=self.vis3D.sideCam_h)

        except Exception as e:
            print("[apply_reset] \n", e)
            pass


    def load_csv_file(self):
        try:
            dataPath = QFileDialog.getOpenFileName(None,
                                                   caption='Load Your Data',
                                                   directory='./save_pattern')
            dataName = dataPath[0].split('/')[-1]
            self.label_3D_data_path.setText('  ' + dataName)
            fileFormat = dataName.split('.')[-1]
            if fileFormat == 'csv':
                self.set_3D_scatter(dataPath[0])
                self.set_data_table(coordinates_x=self.vis3D.coordinates_x,
                                    coordinates_y = self.vis3D.coordinates_y,
                                    depth=self.vis3D.depth,
                                    speed=self.vis3D.speed,
                                    timestamp=self.vis3D.timestamp,
                                    frontCam_w=self.vis3D.frontCam_w,
                                    frontCam_h=self.vis3D.frontCam_h,
                                    sideCam_w=self.vis3D.sideCam_w,
                                    sideCam_h=self.vis3D.sideCam_h)
            elif fileFormat == '':
                pass
            else:
                QMessageBox.about(None, "Error", "Please select a csv file.")

        except Exception as e:
            print("[load_csv_file] \n", e)
            pass


    def load_section(self, dataPath):
        # 특정 구간(시간)에 대한 관상어 3D 이동패턴 시각화
        try:
            if self.label_3D_data_path.text() == '  ' + 'Choose data':
                QMessageBox.about(None, "Error", "Please select a csv file.")
            elif self.label_3D_data_path.text() == '  ':
                QMessageBox.about(None, "Error", "Please select a csv file.")

            # test_time = self.dateTimeEdit_start.dateTime().toPyDateTime()
            start_time = self.dateTimeEdit_start.dateTime()
            start_time = str(start_time.toString('yy.MM.dd.hh.mm.ss'))
            end_time = self.dateTimeEdit_end.dateTime()
            end_time = str(end_time.toString('yy.MM.dd.hh.mm.ss'))

            list_x, list_y, list_depth, list_speed, list_timestamp, \
            list_frontW, list_frontH, list_sideW, list_sideH = self.vis3D.set_time_zone(
                start_time=start_time,
                end_time=end_time,
                x=self.vis3D.coordinates_x,
                y=self.vis3D.coordinates_y,
                depth=self.vis3D.depth,
                speed=self.vis3D.speed,
                timestamp=self.vis3D.timestamp,
                frontW=self.vis3D.frontCam_w,
                frontH=self.vis3D.frontCam_h,
                sideW=self.vis3D.sideCam_w,
                sideH=self.vis3D.sideCam_h)

            if self.smooth_mode == 'scatter':
                self.load_3D_scatter(x=list_x,
                                     y=list_y,
                                     z=list_depth)
            elif self.smooth_mode == 'line':
                self.load_3D_line(x=list_x,
                                  y=list_y,
                                  z=list_depth)

            self.set_data_table(coordinates_x=list_x,
                                coordinates_y = list_y,
                                depth=list_depth,
                                speed=list_speed,
                                timestamp=list_timestamp,
                                frontCam_w=list_frontW,
                                frontCam_h=list_frontH,
                                sideCam_w=list_sideW,
                                sideCam_h=list_sideH)

        except Exception as e:
            print("[load_section] \n", e)
            QMessageBox.about(None, "Error", "Empty data exists.\n"
                                             "Please select another time zone.")
            pass


    def set_data_table(self,
                       coordinates_x,
                       coordinates_y,
                       depth,
                       speed,
                       timestamp,
                       frontCam_w,
                       frontCam_h,
                       sideCam_w,
                       sideCam_h):
        column_headers = ['x\n(frontW)', 'y\n(frontH)', 'z\n(sideW)',
                          'speed\n(mm/s)', 'timestamp\n(y.M.d.h.m.s)',
                          'frontCam_w', 'frontCam_h', 'sideCam_w', 'sideCam_h']

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


    @pyqtSlot(int, int)
    def clicked_data(self, row, col):
        try:
            if col == 4:
                # 이전에 클릭한 데이터는 원래의 색상 및 크기로 변경
                self.scatter_size[self.check_row] = 5
                self.scatter_color[self.check_row] = (0.0, 1.0, 0.0, 1.0)
                # 클릭한 데이터 색상 및 크기 변경
                self.scatter_size[row] = 15
                self.scatter_color[row] = (0.0, 0.0, 1.0, 1.0)
                # row 인덱스 저장
                self.check_row = row

                self.update_3D_scatter(pos=self.pos,
                                       size=self.scatter_size,
                                       color=self.scatter_color)

        except Exception as e:
            print("[clicked_data] \n", e)
            pass


    @pyqtSlot()
    def selected_data(self):
        try:
            index = self.tableWidget_dataList.selectedIndexes()
            cell = set((idx.row(), idx.column()) for idx in index)
            self._pos = []
            for i in cell:
                row = i[0]
                col = i[1]
                # 타임스탬프 컬럼에서만 작동
                if col == 4:
                    # 선택한 데이터 색상 및 크기 변경
                    self.scatter_color[row] = (1.0, 0.0, 0.0, 1.0)
                    self.scatter_size[row] = 10

            if self.smooth_mode == 'scatter':
                self.update_3D_scatter(pos=self.pos,
                                       size=self.scatter_size,
                                       color=self.scatter_color)
            elif self.smooth_mode == 'line':
                self.update_3D_line(pos=self.pos,
                                    width=self.line_size,
                                    color=self.line_color)

        except Exception as e:
            print("[selected_data] \n", e)
            pass


    @pyqtSlot()
    def selected_data_reset(self):
        try:
            self.load_3D_scatter(x=self.vis3D.coordinates_x,
                                 y=self.vis3D.coordinates_y,
                                 z=self.vis3D.depth)
            self.set_data_table(coordinates_x=self.vis3D.coordinates_x,
                                coordinates_y=self.vis3D.coordinates_y,
                                depth=self.vis3D.depth,
                                speed=self.vis3D.speed,
                                timestamp=self.vis3D.timestamp,
                                frontCam_w=self.vis3D.frontCam_w,
                                frontCam_h=self.vis3D.frontCam_h,
                                sideCam_w=self.vis3D.sideCam_w,
                                sideCam_h=self.vis3D.sideCam_h)

        except Exception as e:
            print("[selected_data_reset] \n", e)
            pass


    def capture_vis(self):
        try:
            dataPath = QFileDialog.getSaveFileName(None,
                                                   caption='Capture image path',
                                                   directory='./capture_3D_img')
            filePath = dataPath[0]
            if filePath.split('.')[-1] == 'png':
                filePath = filePath.split('.')[0]
                self.graphicsView.grabFrameBuffer().save('{}.png'.format(filePath))

        except Exception as e:
            print("[capture_vis] \n", e)
            pass


if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
