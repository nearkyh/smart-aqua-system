import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import collections
import os
import sys
import time
import datetime
import math

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Model preparation
# inference_graph = 'rfcn_resnet101_aquarium_fish_29408'      # class 5, AngelFish, ClownFish, Discus, Jinjurin, Redhorn
inference_graph = 'rfcn_resnet101_aquarium_fish_v2_22751'   # class 3, AngelFish, ClownFish, Discus
MODEL_NAME = 'object_detection/inference_graph/{}'.format(inference_graph)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('object_detection/data', 'aquarium_fish_label_map.pbtxt')     # class 5, AngelFish, ClownFish, Discus, Jinjurin, Redhorn
PATH_TO_LABELS = os.path.join('object_detection/data', 'aquarium_fish_v2_label_map.pbtxt')  # class 3, AngelFish, ClownFish, Discus

# Aquarium-Fish
# NUM_CLASSES = 5     # AngelFish, ClownFish, Discus, Jinjurin, Redhorn
NUM_CLASSES = 3     # AngelFish, ClownFish, Discus

# Load a (frozen) TensorFlow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                        feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                       np.squeeze(boxes),
                                                       np.squeeze(classes).astype(np.int32),
                                                       np.squeeze(scores),
                                                       category_index,
                                                       use_normalized_coordinates=True,
                                                       min_score_thresh=.5,
                                                       line_thickness=4)
    return boxes, scores, classes, category_index


def data_processing_on_detect_objects(cam_index, image_np, width, height, point_buff_size, point_buff, boxes, scores, classes, category_index):
    count_objects = 0
    get_scores = np.squeeze(scores)
    objects_score = []
    get_category = np.array([category_index.get(i) for i in classes[0]])
    objects_category = np.array([])

    for i in range(100):
        if scores is None or get_scores[i] > .5:
            count_objects = count_objects + 1
            objects_score = np.append(objects_score, get_scores[i])
            objects_category = np.append(objects_category, get_category[i])

    '''
        x1,y1 ---
        |       |
        |       |
        --- x2,y2
        x_min = x1
        y_min = y1
        x_max = x2
        y_max = y2
    '''
    point = None
    x_min = None
    y_min = None
    x_max = None
    y_max = None

    for i in range(len(objects_score)):
        # Get boxes(y_min, x_min, y_max, x_max)
        get_boxes = np.squeeze(boxes)[i]
        x_min = int(get_boxes[1] * width)
        y_min = int(get_boxes[0] * height)
        x_max = int(get_boxes[3] * width)
        y_max = int(get_boxes[2] * height)

        # TensorFlow visualization coordinates
        coordinates_x = int((x_max + x_min) / 2)
        coordinates_y = int((y_max + y_min) / 2)

        point = (coordinates_x, coordinates_y)
        # cv2.circle(image_np, point, 2, (0, 255, 0), -1)
    point_buff.appendleft(point)

    # Loop over the set of tracked points
    for i in range(1, len(point_buff)):
        # If either of the tracked points are None, ignore
        # them
        if point_buff[i - 1] is None or point_buff[i] is None:
            continue
        # Otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(point_buff_size / float(i + 1)) * .5)
        # cv2.line(image_np, point_buff[i - 1], point_buff[i], (0, 255, 0), thickness)

    cv2.putText(image_np, cam_index, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    return objects_category, point, x_min, y_min, x_max, y_max, count_objects


def recoding_video(cap, width, height):
    recording_video = "rec_{}.avi".format(cap)
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    rec_frameRate = 10.0

    return cv2.VideoWriter(recording_video, fcc, rec_frameRate, (width, height))


def output_rec_video(frame, rec_camera):
    rec_camera.write(frame)


def get_resolution(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height


def set_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def crop_horizontal(image, camWidth, cropWidth):
    return image[:,
           int((camWidth - cropWidth) / 2):
           int(cropWidth + (camWidth - cropWidth) / 2)]


def triangulation(leftCam_point, rightCam_point, rotationMatrix, translationVector, leftProjection, rightProjection):
    R = rotationMatrix
    T = translationVector
    P1 = leftProjection
    P2 = rightProjection

    x1 = float(leftCam_point[0])
    y1 = float(leftCam_point[1])
    x2 = float(rightCam_point[0])
    y2 = float(rightCam_point[1])

    newMat = [R[0][0] * (x2 - T[0][0]) + R[0][1] * (y2 - T[1][0]) + R[0][2] * (0 - T[2][0]),
              R[1][0] * (x2 - T[0][0]) + R[1][1] * (y2 - T[1][0]) + R[1][2] * (0 - T[2][0]),
              R[2][0] * (x2 - T[0][0]) + R[2][1] * (y2 - T[1][0]) + R[2][2] * (0 - T[2][0])]
    equalTo = [[x1, 0, 0 - R[0][2]],
               [0, y1, 0 - R[1][2]],
               [0, 0, 1 - R[2][2]]]
    Pr = np.linalg.solve(equalTo, newMat)

    newMat2 = [R[0][0] * (x1 - T[0][0]) + R[0][1] * (y1 - T[1][0]) + R[0][2] * (0 - T[2][0]),
               R[1][0] * (x1 - T[0][0]) + R[1][1] * (y1 - T[1][0]) + R[1][2] * (0 - T[2][0]),
               R[2][0] * (x1 - T[0][0]) + R[2][1] * (y1 - T[1][0]) + R[2][2] * (0 - T[2][0])]
    equalTo2 = [[x2, 0, 0 - R[0][2]],
                [0, y2, 0 - R[1][2]],
                [0, 0, 1 - R[2][2]]]
    Pl = np.linalg.solve(equalTo2, newMat2)

    points_4D = np.array([])
    points_4D = cv2.triangulatePoints(P1, P2, np.array([Pr[0], Pr[1]]), np.array([Pl[0], Pl[1]]), points_4D)
    # print("points_4D", points_4D)

    return (Pl, Pr)


def display_detect_abnormal_behavior(num_pattern, leftFrame, rightFrame, leftCam_w, leftCam_h, rightCam_w, rightCam_h):
    text_size = 1
    display_color = ()
    cv_color_red = (0, 0, 255)
    cv_color_green = (0, 255, 0)
    cv_color_blue = (255, 0, 0)
    cv_color_black = (0, 0, 0)
    cv_color_white = (255, 255, 255)

    if num_pattern == 1:
        display_color = cv_color_red
        cv2.putText(leftFrame, 'Detect abnormal behavior!!!', (10, 40), cv2.FONT_HERSHEY_PLAIN, text_size, display_color)
        cv2.putText(rightFrame, 'Detect abnormal behavior!!!', (10, 40), cv2.FONT_HERSHEY_PLAIN, text_size, display_color)
    elif num_pattern == 2:
        display_color = cv_color_green
        cv2.putText(leftFrame, 'Detect abnormal behavior!!!', (10, 40), cv2.FONT_HERSHEY_PLAIN, text_size, display_color)
        cv2.putText(rightFrame, 'Detect abnormal behavior!!!', (10, 40), cv2.FONT_HERSHEY_PLAIN, text_size, display_color)
    elif num_pattern == 3:
        display_color = cv_color_white
        cv2.putText(leftFrame, 'Detect white spot disease!!!', (10, 40), cv2.FONT_HERSHEY_PLAIN, text_size, display_color)
        cv2.putText(rightFrame, 'Detect white spot disease!!!', (10, 40), cv2.FONT_HERSHEY_PLAIN, text_size, display_color)

    cv2.rectangle(leftFrame, (0, 0), (leftCam_w, leftCam_h), display_color, 20)
    cv2.rectangle(rightFrame, (0, 0), (rightCam_w, rightCam_h), display_color, 20)


def speed_of_three_dimensional(resolution_x, resolution_y, resolution_z, coordinates_x1, coordinates_x2, coordinates_y1, coordinates_y2, depth1, depth2, time1, time2):
    '''
        Moving speed of an object in a 3D(three-dimensional) space
    '''

    # Size of fish tank, 350*350*350(mm)
    total_length_x = 350
    total_length_y = 350
    total_length_z = 350

    # resolution_x(leftCam) : total_length_x = coordinates_x : real_length_x
    real_length_x1 = (total_length_x * coordinates_x1) / resolution_x
    real_length_x2 = (total_length_x * coordinates_x2) / resolution_x

    # resolution_y(leftCam) : total_length_y = coordinates_y : real_length_y
    real_length_y1 = (total_length_y * coordinates_y1) / resolution_y
    real_length_y2 = (total_length_y * coordinates_y2) / resolution_y

    # resolution_x(rightCam) : total_length_z = depth : real_length_z
    real_length_z1 = (total_length_z * depth1) / resolution_z
    real_length_z2 = (total_length_z * depth2) / resolution_z

    # Euclidean distance
    x = [real_length_x1, real_length_y1, real_length_z1]
    y = [real_length_x2, real_length_y2, real_length_z2]
    dist = math.sqrt(sum([(xi - yi) ** 2 for xi, yi in zip(x, y)]))
    # dist = math.sqrt((int(abs(real_length_x-_real_length_x)))^2 + (int(abs(real_length_y-_real_length_y)))^2 + (int(abs(real_length_z-_real_length_z)))^2)

    return dist/(time2-time1)


def check_behavior_pattern_1st(x1, x2, y1, y2, z1, z2, patternArr_size):
    latest_behavior_pattern = pd.read_csv('save_behavior_pattern/latest_behavior_pattern.csv')
    coordinates_x = latest_behavior_pattern['coordinates_x']
    coordinates_y = latest_behavior_pattern['coordinates_y']
    depth = latest_behavior_pattern['depth']

    count = 0
    for i in range(len(coordinates_x)):
        if (x1 < coordinates_x[i] < x2) and (y1 < coordinates_y[i] < y2) and (z1 < depth[i] < z2):
            count += 1

    # Detection of abnormal behavior when more than 90% of data are included in the area
    if int(count / patternArr_size * 100) >= 90:
        return 'Detect abnormal behavior'
    else:
        return 'Not detected'


def check_behavior_pattern_2st(speed, queue_size_of_speed, queue_of_speed):
    if speed:
        queue_of_speed.append(speed)
    list_of_speed = queue_of_speed[-queue_size_of_speed:]

    count = 0
    for i in list_of_speed:
        if i < 20:
            count += 1

    # Detect if there is no movement, consider noise data(99% or more)
    if int(count / queue_size_of_speed * 100) >= 99:

        return 'Detect abnormal behavior'


def erosionMsk(frame):
    kernel = np.ones((7, 7), np.uint8)
    frame = cv2.erode(frame, kernel, iterations=1)

    return frame


def dilationMsk(frame):
    kernel = np.ones((7, 7), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=1)

    return frame


def check_behavior_pattern_3st(frame, title):
    # Color, BRG
    # l_frame = np.array([0, 0, 0])
    # u_frame = np.array([255, 220, 220])

    # Threshold the image(BGR, HSV)
    # mask = cv2.inRange(frame, l_frame, u_frame)

    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 30

    # Filter by Area
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 50

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    cv_version = (cv2.__version__).split('.')
    if int(cv_version[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Erosion
    # erosion_frame = erosionMsk(mask)

    # Dilation
    # dilation_frame = dilationMsk(erosion_frame)

    keyPoints = detector.detect(frame)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
    frame = cv2.drawKeypoints(frame, keyPoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow(title, frame)

    if keyPoints :

        return 'Detect white spot disease'



if __name__ == '__main__':

    input_camera = False
    input_video = False
    input_realSense = False

    input_leftCam = None
    input_rightCam = None

    try:
        if sys.argv[1] == 'camera':
            input_camera = True
            # Input camera definition
            input_leftCam = 0
            input_rightCam = 1

        elif sys.argv[1] == 'video':
            input_video = True
            # Input video definition
            input_leftCam = 'test.mp4'
            input_rightCam = 'test.mp4'

        elif sys.argv[1] == 'realsense':
            input_realSense = True
            # Input realSense definition
            input_leftCam = 2
            input_rightCam = 5

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

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Check the input device
            input_device = None
            if input_camera == True:
                input_device = 'camera'
            elif input_video == True:
                input_device = 'video'
            elif input_realSense == True:
                input_device = 'video'

            # Opencv, Video capture
            leftCam = cv2.VideoCapture(input_leftCam)
            rightCam = cv2.VideoCapture(input_rightCam)

            # Frame time variable
            prevTime = 0

            # Counting for Save coordinates(x, y)
            countFrame = 0
            checkFrame = 0
            endTime = datetime.datetime.now() + datetime.timedelta(minutes=60)
            fileNum = 0

            # Queue for pattern-finding
            # TODO: Try change data length of dir(latest_behavior_pattern).
            patternArr_size = 6 * 60 * (1)   # Input minutes, ex) 30min == 6fps * 60sec * 30
            patternArr = [] * patternArr_size

            # TODO: Try change data length for check_behavior_pattern_2st.
            queue_size_of_speed = 6 * 60 * (1)    # Input minutes, ex) 30min == 6fps * 60sec * 30
            queue_of_speed = [] * queue_size_of_speed

            # Point(x, y) in queue
            point_buff_size = 64
            leftCam_pts_buff = collections.deque(maxlen=point_buff_size)
            rightCam_pts_buff = collections.deque(maxlen=point_buff_size)

            if input_device == 'camera':
                # ======================================
                #   Load compressed calibration data
                # ======================================
                REMAP_INTERPOLATION = cv2.INTER_LINEAR
                DEPTH_VISUALIZATION_SCALE = 2048
                STEREO_CALIBRATION_DATA = 'camera_calibration/calibration.npz'
                print("Syntax: {0} CALIBRATION_FILE".format(STEREO_CALIBRATION_DATA))
                calibration = np.load(STEREO_CALIBRATION_DATA, allow_pickle=False)

                # get imageSize
                imageSize = tuple(calibration["imageSize"])

                # getMatchingObjectAndImagePoints
                leftObjectPoints = calibration["leftObjectPoints"]
                leftImagePoints = calibration["leftImagePoints"]
                rightObjectPoints = calibration["rightObjectPoints"]
                rightImagePoints = calibration["rightImagePoints"]

                # calibrateCamera
                leftCameraMatrix = calibration["leftCameraMatrix"]
                leftDistortionCoefficients = calibration["leftDistortionCoefficients"]
                rightCameraMatrix = calibration["rightCameraMatrix"]
                rightDistortionCoefficients = calibration["rightDistortionCoefficients"]

                # stereoCalibrate
                rotationMatrix = calibration["rotationMatrix"]
                translationVector = calibration["translationVector"]

                # stereoRectify
                leftRectification = calibration["leftRectification"]
                rightRectification = calibration["rightRectification"]
                leftProjection = calibration["leftProjection"]
                rightProjection = calibration["rightProjection"]
                dispartityToDepthMap = calibration["dispartityToDepthMap"]
                leftROI = tuple(calibration["leftROI"])
                rightROI = tuple(calibration["rightROI"])

                # initUndistortRectifyMap
                leftMapX = calibration["leftMapX"]
                leftMapY = calibration["leftMapY"]
                rightMapX = calibration["rightMapX"]
                rightMapY = calibration["rightMapY"]

                '''
                    The distortion in the leftCam and rightCam edges prevents a good calibration,
                    so discard the edges
                '''
                camWidth = 1280
                camHeight = 720
                cropWidth = 960

                set_resolution(cap=leftCam,
                               width=cropWidth,
                               height=camHeight)
                set_resolution(cap=rightCam,
                               width=cropWidth,
                               height=camHeight)

                # TODO: Try applying brightness/contrast/gamma adjustments to the images.
                stereoMatcher = cv2.StereoBM_create()
                stereoMatcher.setMinDisparity(4)
                stereoMatcher.setNumDisparities(128)
                stereoMatcher.setBlockSize(21)
                stereoMatcher.setROI1(leftROI)
                stereoMatcher.setROI2(rightROI)
                stereoMatcher.setSpeckleRange(16)
                stereoMatcher.setSpeckleWindowSize(45)

            elif input_device == 'video':
                leftWidth, leftHeight = get_resolution(cap=leftCam)
                rightWidth, rightHeight = get_resolution(cap=rightCam)
                camWidth = leftWidth
                camHeight = leftHeight

            # =====================================================
            #   Resize image to match the size of the fish-tank
            # =====================================================
            # TODO: Try applying the size of the part you want to crop from the image.
            if input_device == 'camera':
                leftCam_cropSize_U = 0
                leftCam_cropSize_B = 0
                leftCam_cropSize_L = int((cropWidth - camHeight) / 2)
                leftCam_cropSize_R = int((cropWidth - camHeight) / 2)

                leftCam_fixedWidth = cropWidth - (leftCam_cropSize_L + leftCam_cropSize_R)
                leftCam_fixedHeight = camHeight - (leftCam_cropSize_U + leftCam_cropSize_B)

                rightCam_cropSize_U = 0
                rightCam_cropSize_B = 0
                rightCam_cropSize_L = int((cropWidth - camHeight) / 2)
                rightCam_cropSize_R = int((cropWidth - camHeight) / 2)

                rightCam_fixedWidth = cropWidth - (rightCam_cropSize_L + rightCam_cropSize_R)
                rightCam_fixedHeight = camHeight - (rightCam_cropSize_U + rightCam_cropSize_B)

            elif input_device == 'video':
                leftCam_cropSize_U = 0
                leftCam_cropSize_B = 0
                # leftCam_cropSize_L = int((camWidth - camHeight) / 2)
                # leftCam_cropSize_R = int((camWidth - camHeight) / 2)
                leftCam_cropSize_L = 30
                leftCam_cropSize_R = 30

                leftCam_fixedWidth = camWidth - (leftCam_cropSize_L + leftCam_cropSize_R)
                leftCam_fixedHeight = camHeight - (leftCam_cropSize_U + leftCam_cropSize_B)

                rightCam_cropSize_U = 0
                rightCam_cropSize_B = 0
                # rightCam_cropSize_L = int((camWidth - camHeight) / 2)
                # rightCam_cropSize_R = int((camWidth - camHeight) / 2)
                rightCam_cropSize_L = 30
                rightCam_cropSize_R = 30

                rightCam_fixedWidth = camWidth - (rightCam_cropSize_L + rightCam_cropSize_R)
                rightCam_fixedHeight = camHeight - (rightCam_cropSize_U + rightCam_cropSize_B)

            # =====================
            #   Recording video
            # =====================
            rec_leftCam = recoding_video(cap='leftCam',
                                         width=leftCam_fixedWidth,
                                         height=leftCam_fixedHeight)
            rec_rightCam = recoding_video(cap='rightCam',
                                          width=rightCam_fixedWidth,
                                          height=rightCam_fixedHeight)

            while True:
                try:
                    if not leftCam.grab() or not rightCam.grab():
                        print("No more frames")
                        break

                    _, leftFrame = leftCam.read()
                    _, rightFrame = rightCam.read()

                    # ================
                    #   Frame rate
                    # ================
                    curTime = time.time()
                    sec = curTime - prevTime
                    prevTime = curTime
                    frameRate = "FPS %0.1f" % (1 / (sec))

                    try:
                        if input_device == 'camera':
                            # ===============================
                            #   Stereo camera calibration
                            # ===============================
                            leftFrame = crop_horizontal(image=leftFrame,
                                                        camWidth=camWidth,
                                                        cropWidth=cropWidth)
                            leftHeight, leftWidth = leftFrame.shape[:2]

                            rightFrame = crop_horizontal(image=rightFrame,
                                                         camWidth=camWidth,
                                                         cropWidth=cropWidth)
                            rightHeight, rightWidth = rightFrame.shape[:2]

                            if (leftWidth, leftHeight) != imageSize:
                                print("Left camera has different size than the calibration data")
                                break
                            if (rightWidth, rightHeight) != imageSize:
                                print("Right camera has different size than the calibration data")
                                break

                            leftFrame = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
                            rightFrame = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

                            # =======================
                            #   Depth map creator
                            # =======================
                            grayLeft = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
                            grayRight = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
                            disparity = stereoMatcher.compute(grayLeft, grayRight)
                            coordinates_of_3D = cv2.reprojectImageTo3D(disparity, dispartityToDepthMap)

                        elif input_device == 'video':
                            pass

                    except Exception:
                        pass

                    try:
                        # =====================================================
                        #   Resize image to match the size of the fish-tank
                        # =====================================================
                        if input_device == 'camera':
                            leftFrame = leftFrame[leftCam_cropSize_U:camHeight-leftCam_cropSize_B, leftCam_cropSize_L:cropWidth-leftCam_cropSize_R]
                            rightFrame = rightFrame[rightCam_cropSize_U:camHeight-rightCam_cropSize_B, rightCam_cropSize_L:cropWidth-rightCam_cropSize_R]

                        elif input_device == 'video':
                            leftFrame = leftFrame[leftCam_cropSize_U:camHeight-leftCam_cropSize_B, leftCam_cropSize_L:camWidth-leftCam_cropSize_R]
                            rightFrame = rightFrame[rightCam_cropSize_U:camHeight-rightCam_cropSize_B, rightCam_cropSize_L:camWidth-rightCam_cropSize_R]

                    except Exception:
                        pass

                    # ============================
                    #   Aquarium-Fish detector
                    # ============================
                    leftCam_boxes, leftCam_scores, leftCam_classes, leftCam_category_index = detect_objects(
                        image_np=leftFrame,
                        sess=sess,
                        detection_graph=detection_graph)
                    rightCam_boxes, rightCam_scores, rightCam_classes, rightCam_category_index = detect_objects(
                        image_np=rightFrame,
                        sess=sess,
                        detection_graph=detection_graph)

                    # ==========================================
                    #   Aquarium-Fish detector data analysis
                    # ==========================================
                    leftCam_objects_category, leftCam_object_point, leftCam_x_min, leftCam_y_min, leftCam_x_max, leftCam_y_max, leftCam_count_objects = data_processing_on_detect_objects(
                        cam_index='Front Camera',
                        image_np=leftFrame,
                        width=leftCam_fixedWidth,
                        height=leftCam_fixedHeight,
                        point_buff_size=point_buff_size,
                        point_buff=leftCam_pts_buff,
                        boxes=leftCam_boxes,
                        scores=leftCam_scores,
                        classes=leftCam_classes,
                        category_index=leftCam_category_index)

                    rightCam_objects_category, rightCam_object_point, rightCam_x_min, rightCam_y_min, rightCam_x_max, rightCam_y_max, rightCam_count_objects = data_processing_on_detect_objects(
                        cam_index='Side Camera',
                        image_np=rightFrame,
                        width=rightCam_fixedWidth,
                        height=rightCam_fixedHeight,
                        point_buff_size=point_buff_size,
                        point_buff=rightCam_pts_buff,
                        boxes=rightCam_boxes,
                        scores=rightCam_scores,
                        classes=rightCam_classes,
                        category_index=rightCam_category_index)

                    try:
                        # ============================
                        #   Data output per frames
                        # ============================
                        # if countFrame == 0:
                        # check_objects_category = (leftCam_objects_category[0]['name'] == rightCam_objects_category[0]['name'])
                        # check_count_objects = ((leftCam_count_objects == 1) and (rightCam_count_objects == 1))
                        # if check_objects_category and check_count_objects:
                        if input_device == 'camera':
                            # ============================================
                            #   Measuring distance using triangulation
                            # ============================================
                            Pl, Pr = triangulation(leftCam_point=leftCam_object_point,
                                                   rightCam_point=rightCam_object_point,
                                                   rotationMatrix=rotationMatrix,
                                                   translationVector=translationVector,
                                                   leftProjection=leftProjection,
                                                   rightProjection=rightProjection)
                            # print("Distance: {}".format(round(Pr[2] / 300, 2)))

                        elif input_device == 'video':
                            pass

                        # ========================================================================
                        #   Save coordinates(x, y), depth, history(time), frame(width, height)
                        # ========================================================================
                        coordinates_value_list = []
                        column_name = ['coordinates_x', 'coordinates_y', 'depth', 'history', 'timestamp', 'frontCam_w', 'frontCam_h', 'sideCam_w', 'sideCam_h']

                        '''
                        Method 1
                            3D positions (x, y and depth) were assigned combining
                            the horizontal coordinates of the left camera (coordinates_x and coordinates_y) and
                            the vertical coordinate of the right camera (depth using x-axis)
                            leftCam_object_point[0]: coordinates_x
                            leftCam_object_point[1]: coordinates_y
                            rightCam_object_point[0]: depth
                        '''
                        coordinates_value = (leftCam_object_point[0],
                                             leftCam_object_point[1],
                                             rightCam_object_point[0],
                                             time.time(),
                                             datetime.datetime.now(),
                                             leftCam_fixedWidth,
                                             leftCam_fixedHeight,
                                             rightCam_fixedWidth,
                                             rightCam_fixedHeight)

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
                        # coordinates_value = (xs,
                        #                      ys,
                        #                      zs,
                        #                      time.time(),
                        #                      datetime.datetime.now(),
                        #                      leftCam_fixedWidth,
                        #                      leftCam_fixedHeight,
                        #                      rightCam_fixedWidth,
                        #                      rightCam_fixedHeight)

                        coordinates_value_list.append(coordinates_value)

                        # ================================
                        #   Save behavior pattern data
                        # ================================
                        behavior_pattern_save_dir = 'save_behavior_pattern'
                        if not os.path.isdir(behavior_pattern_save_dir):
                            os.mkdir(behavior_pattern_save_dir)

                        save_file = behavior_pattern_save_dir + '/save_{}.csv'.format(str(fileNum).rjust(4, '0'))
                        if os.path.exists(save_file) == True:
                            with open(save_file, 'a') as f:
                                xml_df = pd.DataFrame(coordinates_value_list, columns=column_name)
                                xml_df.to_csv(f, header=False, index=None)
                        else:
                            xml_df = pd.DataFrame(columns=column_name)
                            xml_df.to_csv(save_file, index=None)
                            with open(save_file, 'a') as f:
                                xml_df = pd.DataFrame(coordinates_value_list, columns=column_name)
                                xml_df.to_csv(f, header=False, index=None)

                        # =======================================================
                        #   Save in the last (x) minutes behavior pattern data
                        # =======================================================
                        patternArr.append(coordinates_value_list)
                        patternArr = patternArr[-patternArr_size:]

                        patternArr_file = behavior_pattern_save_dir + '/latest_behavior_pattern.csv'
                        if os.path.exists(patternArr_file) == True:
                            with open(patternArr_file, 'w') as f:
                                xml_df = pd.DataFrame(columns=column_name)
                                xml_df.to_csv(f, header=True, index=None)
                                for i in range(len(patternArr)):
                                    xml_df = pd.DataFrame(patternArr[i], columns=column_name)
                                    xml_df.to_csv(f, header=False, index=None)
                        else:
                            xml_df = pd.DataFrame(columns=column_name)
                            xml_df.to_csv(patternArr_file, index=None)
                            with open(patternArr_file, 'w') as f:
                                for i in range(len(patternArr)):
                                    xml_df = pd.DataFrame(patternArr[i], columns=column_name)
                                    xml_df.to_csv(f, header=False, index=None)

                    # When it can not be detected even from one frame
                    except Exception:
                        pass

                    # ============================================================================
                    #   1st pattern : If object stay on the edge of the screen for a long time
                    #   and abnormal behavior detection area
                    # ============================================================================
                    try:
                        # TODO: Try change data length of abnormal behavior point.
                        abnormal_behavior_size_x = 120
                        abnormal_behavior_size_y = 120
                        error_range = 0

                        cv_red = (0, 0, 255)
                        cv_green = (0, 255, 0)
                        cv_blue = (255, 0, 0)
                        cv_color = (0, 0, 0)
                        rectangle_line_thickness = 1

                        # The front of the Smart-Aquarium
                        check_abnormal_behavior_list = []
                        # Left & Upper
                        check_abnormal_behavior = check_behavior_pattern_1st(x1=0, x2=abnormal_behavior_size_x,
                                                                             y1=0, y2=abnormal_behavior_size_y,
                                                                             z1=0, z2=abnormal_behavior_size_x,
                                                                             patternArr_size=patternArr_size)
                        check_abnormal_behavior_list.append(check_abnormal_behavior)

                        # Right & Upper
                        check_abnormal_behavior = check_behavior_pattern_1st(x1=leftCam_fixedWidth - abnormal_behavior_size_x, x2=leftCam_fixedWidth,
                                                                             y1=0, y2=abnormal_behavior_size_y,
                                                                             z1=0, z2=abnormal_behavior_size_x,
                                                                             patternArr_size=patternArr_size)
                        check_abnormal_behavior_list.append(check_abnormal_behavior)

                        # Left & Bottom
                        check_abnormal_behavior = check_behavior_pattern_1st(x1=0, x2=abnormal_behavior_size_x,
                                                                             y1=leftCam_fixedHeight - abnormal_behavior_size_y, y2=leftCam_fixedHeight,
                                                                             z1=0, z2=abnormal_behavior_size_x,
                                                                             patternArr_size=patternArr_size)
                        check_abnormal_behavior_list.append(check_abnormal_behavior)

                        # Right & Bottom
                        check_abnormal_behavior = check_behavior_pattern_1st(x1=leftCam_fixedWidth - abnormal_behavior_size_x, x2=leftCam_fixedWidth,
                                                                             y1=leftCam_fixedHeight - abnormal_behavior_size_y, y2=leftCam_fixedHeight,
                                                                             z1=0, z2=abnormal_behavior_size_x,
                                                                             patternArr_size=patternArr_size)
                        check_abnormal_behavior_list.append(check_abnormal_behavior)

                        # The back of the Smart-Aquarium
                        # Left & Upper
                        check_abnormal_behavior = check_behavior_pattern_1st(x1=0, x2=abnormal_behavior_size_x,
                                                                             y1=0, y2=abnormal_behavior_size_y,
                                                                             z1=leftCam_fixedWidth - abnormal_behavior_size_x, z2=leftCam_fixedWidth,
                                                                             patternArr_size=patternArr_size)
                        check_abnormal_behavior_list.append(check_abnormal_behavior)

                        # Right & Upper
                        check_abnormal_behavior = check_behavior_pattern_1st(x1=leftCam_fixedWidth - abnormal_behavior_size_x, x2=leftCam_fixedWidth,
                                                                             y1=0, y2=abnormal_behavior_size_y,
                                                                             z1=abnormal_behavior_size_x, z2=leftCam_fixedWidth,
                                                                             patternArr_size=patternArr_size)
                        check_abnormal_behavior_list.append(check_abnormal_behavior)

                        # Left & Bottom
                        check_abnormal_behavior = check_behavior_pattern_1st(x1=0, x2=abnormal_behavior_size_x,
                                                                             y1=leftCam_fixedHeight - abnormal_behavior_size_y, y2=leftCam_fixedHeight,
                                                                             z1=leftCam_fixedWidth - abnormal_behavior_size_x, z2=leftCam_fixedWidth,
                                                                             patternArr_size=patternArr_size)
                        check_abnormal_behavior_list.append(check_abnormal_behavior)

                        # Right & Bottom
                        check_abnormal_behavior = check_behavior_pattern_1st(x1=leftCam_fixedWidth - abnormal_behavior_size_x, x2=leftCam_fixedWidth,
                                                                             y1=leftCam_fixedHeight - abnormal_behavior_size_y, y2=leftCam_fixedHeight,
                                                                             z1=leftCam_fixedWidth - abnormal_behavior_size_x, z2=leftCam_fixedWidth,
                                                                             patternArr_size=patternArr_size)
                        check_abnormal_behavior_list.append(check_abnormal_behavior)

                        # The front of the Smart-Aquarium
                        # Left & Upper
                        if (0 <= leftCam_object_point[0] <= abnormal_behavior_size_x) and (0 <= leftCam_object_point[1] <= abnormal_behavior_size_y):
                            cv2.rectangle(leftFrame,
                                          (0, 0), (abnormal_behavior_size_x, abnormal_behavior_size_y),
                                          cv_red, rectangle_line_thickness)
                        else:
                            cv2.rectangle(leftFrame,
                                          (0, 0), (abnormal_behavior_size_x, abnormal_behavior_size_y),
                                          cv_blue, rectangle_line_thickness)

                        # Right & Upper
                        if (leftCam_fixedWidth - abnormal_behavior_size_x <= leftCam_object_point[0] <= leftCam_fixedWidth) and (0 <= leftCam_object_point[1] <= abnormal_behavior_size_y):
                            cv2.rectangle(leftFrame,
                                          (leftCam_fixedWidth - abnormal_behavior_size_x, 0), (leftCam_fixedWidth, abnormal_behavior_size_y),
                                          cv_red, rectangle_line_thickness)
                        else:
                            cv2.rectangle(leftFrame,
                                          (leftCam_fixedWidth - abnormal_behavior_size_x, 0), (leftCam_fixedWidth, abnormal_behavior_size_y),
                                          cv_blue, rectangle_line_thickness)

                        # Left & Bottom
                        if (0 <= leftCam_object_point[0] <= abnormal_behavior_size_x) and (leftCam_fixedHeight - abnormal_behavior_size_y <= leftCam_object_point[1] <= leftCam_fixedHeight):
                            cv2.rectangle(leftFrame,
                                          (0, leftCam_fixedHeight - abnormal_behavior_size_y), (abnormal_behavior_size_x, leftCam_fixedHeight),
                                          cv_red, rectangle_line_thickness)
                        else:
                            cv2.rectangle(leftFrame,
                                          (0, leftCam_fixedHeight - abnormal_behavior_size_y), (abnormal_behavior_size_x, leftCam_fixedHeight),
                                          cv_blue, rectangle_line_thickness)

                        # Right & Bottom
                        if (leftCam_fixedWidth - abnormal_behavior_size_x <= leftCam_object_point[0] <= leftCam_fixedWidth) and (leftCam_fixedHeight - abnormal_behavior_size_y <= leftCam_object_point[1] <= leftCam_fixedHeight):
                            cv2.rectangle(leftFrame,
                                          (leftCam_fixedWidth - abnormal_behavior_size_x, leftCam_fixedHeight - abnormal_behavior_size_y), (leftCam_fixedWidth, leftCam_fixedHeight),
                                          cv_red, rectangle_line_thickness)
                        else:
                            cv2.rectangle(leftFrame,
                                          (leftCam_fixedWidth - abnormal_behavior_size_x, leftCam_fixedHeight - abnormal_behavior_size_y), (leftCam_fixedWidth, leftCam_fixedHeight),
                                          cv_blue, rectangle_line_thickness)

                        # The back of the Smart-Aquarium
                        # Left & Upper
                        if (0 <= rightCam_object_point[0] <= abnormal_behavior_size_x) and (0 <= rightCam_object_point[1] <= abnormal_behavior_size_y):
                            cv2.rectangle(rightFrame,
                                          (0, 0), (abnormal_behavior_size_x, abnormal_behavior_size_y),
                                          cv_red, rectangle_line_thickness)
                        else:
                            cv2.rectangle(rightFrame,
                                          (0, 0), (abnormal_behavior_size_x, abnormal_behavior_size_y),
                                          cv_blue, rectangle_line_thickness)

                        # Right & Upper
                        if (leftCam_fixedWidth - abnormal_behavior_size_x <= rightCam_object_point[0] <= leftCam_fixedWidth) and (0 <= rightCam_object_point[1] <= abnormal_behavior_size_y):
                            cv2.rectangle(rightFrame,
                                          (rightCam_fixedWidth - abnormal_behavior_size_x, 0), (rightCam_fixedWidth, abnormal_behavior_size_y),
                                          cv_red, rectangle_line_thickness)
                        else:
                            cv2.rectangle(rightFrame,
                                          (rightCam_fixedWidth - abnormal_behavior_size_x, 0), (rightCam_fixedWidth, abnormal_behavior_size_y),
                                          cv_blue, rectangle_line_thickness)

                        # Left & Bottom
                        if (0 <= rightCam_object_point[0] <= abnormal_behavior_size_x) and (leftCam_fixedHeight - abnormal_behavior_size_y <= rightCam_object_point[1] <= leftCam_fixedHeight):
                            cv2.rectangle(rightFrame,
                                          (0, rightCam_fixedHeight - abnormal_behavior_size_y), (abnormal_behavior_size_x, rightCam_fixedHeight),
                                          cv_red, rectangle_line_thickness)
                        else:
                            cv2.rectangle(rightFrame,
                                          (0, rightCam_fixedHeight - abnormal_behavior_size_y), (abnormal_behavior_size_x, rightCam_fixedHeight),
                                          cv_blue, rectangle_line_thickness)

                        # Right & Bottom
                        if (leftCam_fixedWidth - abnormal_behavior_size_x <= rightCam_object_point[0] <= leftCam_fixedWidth) and (leftCam_fixedHeight - abnormal_behavior_size_y <= rightCam_object_point[1] <= leftCam_fixedHeight):
                            cv2.rectangle(rightFrame,
                                          (rightCam_fixedWidth - abnormal_behavior_size_x, rightCam_fixedHeight - abnormal_behavior_size_y), (rightCam_fixedWidth, rightCam_fixedHeight),
                                          cv_red, rectangle_line_thickness)
                        else:
                            cv2.rectangle(rightFrame,
                                          (rightCam_fixedWidth - abnormal_behavior_size_x, rightCam_fixedHeight - abnormal_behavior_size_y), (rightCam_fixedWidth, rightCam_fixedHeight),
                                          cv_blue, rectangle_line_thickness)

                        _check_abnormal_behavior = False
                        for i in check_abnormal_behavior_list:
                            if i == 'Detect abnormal behavior':
                                _check_abnormal_behavior = True
                            if _check_abnormal_behavior == True:
                                display_detect_abnormal_behavior(num_pattern=1,
                                                                 leftFrame=leftFrame,
                                                                 rightFrame=rightFrame,
                                                                 leftCam_w=leftCam_fixedWidth,
                                                                 leftCam_h=leftCam_fixedHeight,
                                                                 rightCam_w=rightCam_fixedWidth,
                                                                 rightCam_h=rightCam_fixedHeight)
                        check_abnormal_behavior_list.clear()

                    except Exception:
                        # The front of the Smart-Aquarium
                        # Left & Upper
                        cv2.rectangle(leftFrame,
                                      (0, 0), (abnormal_behavior_size_x, abnormal_behavior_size_y),
                                      cv_blue, rectangle_line_thickness)
                        # Right & Upper
                        cv2.rectangle(leftFrame,
                                      (leftCam_fixedWidth - abnormal_behavior_size_x, 0), (leftCam_fixedWidth, abnormal_behavior_size_y),
                                      cv_blue, rectangle_line_thickness)
                        # Left & Bottom
                        cv2.rectangle(leftFrame,
                                      (0, leftCam_fixedHeight - abnormal_behavior_size_y), (abnormal_behavior_size_x, leftCam_fixedHeight),
                                      cv_blue, rectangle_line_thickness)
                        # Right & Bottom
                        cv2.rectangle(leftFrame,
                                      (leftCam_fixedWidth - abnormal_behavior_size_x, leftCam_fixedHeight - abnormal_behavior_size_y), (leftCam_fixedWidth, leftCam_fixedHeight),
                                      cv_blue, rectangle_line_thickness)

                        # The back of the Smart-Aquarium
                        # Left & Upper
                        cv2.rectangle(rightFrame,
                                      (0, 0), (abnormal_behavior_size_x, abnormal_behavior_size_y),
                                      cv_blue, rectangle_line_thickness)
                        # Right & Upper
                        cv2.rectangle(rightFrame,
                                      (rightCam_fixedWidth - abnormal_behavior_size_x, 0), (rightCam_fixedWidth, abnormal_behavior_size_y),
                                      cv_blue, rectangle_line_thickness)
                        # Left & Bottom
                        cv2.rectangle(rightFrame,
                                      (0, rightCam_fixedHeight - abnormal_behavior_size_y), (abnormal_behavior_size_x, rightCam_fixedHeight),
                                      cv_blue, rectangle_line_thickness)
                        # Right & Bottom
                        cv2.rectangle(rightFrame,
                                      (rightCam_fixedWidth - abnormal_behavior_size_x, rightCam_fixedHeight - abnormal_behavior_size_y), (rightCam_fixedWidth, rightCam_fixedHeight),
                                      cv_blue, rectangle_line_thickness)
                        pass

                    # ==================================================================
                    #   2st pattern : If the movement is noticeably slower or faster
                    # ==================================================================
                    try:
                        latest_behavior_pattern = pd.read_csv('save_behavior_pattern/latest_behavior_pattern.csv')
                        coordinates_x = latest_behavior_pattern['coordinates_x']
                        coordinates_y = latest_behavior_pattern['coordinates_y']
                        depth = latest_behavior_pattern['depth']
                        history = latest_behavior_pattern['history']
                        num_pattern = len(coordinates_x) - 1  # The last pattern value
                        speed = speed_of_three_dimensional(resolution_x=leftCam_fixedWidth,
                                                           resolution_y=leftCam_fixedHeight,
                                                           resolution_z=rightCam_fixedWidth,
                                                           coordinates_x1=coordinates_x[num_pattern - int(float("{0:.1f}".format(1/sec)))],
                                                           coordinates_x2=coordinates_x[num_pattern],
                                                           coordinates_y1=coordinates_y[num_pattern - int(float("{0:.1f}".format(1/sec)))],
                                                           coordinates_y2=coordinates_y[num_pattern],
                                                           depth1=depth[num_pattern - int(float("{0:.1f}".format(1/sec)))],
                                                           depth2=depth[num_pattern],
                                                           time1=history[num_pattern - int(float("{0:.1f}".format(1/sec)))],
                                                           time2=history[num_pattern])
                        cv2.putText(leftFrame, '{0:.2f}mm/s'.format(speed), (leftCam_x_max, leftCam_y_max), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                        cv2.putText(rightFrame, '{0:.2f}mm/s'.format(speed), (rightCam_x_max, rightCam_y_max), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

                        if countFrame == 0:
                            check_abnormal_behavior = check_behavior_pattern_2st(speed=speed,
                                                                                 queue_size_of_speed=queue_size_of_speed,
                                                                                 queue_of_speed=queue_of_speed)

                            if check_abnormal_behavior == 'Detect abnormal behavior':
                                display_detect_abnormal_behavior(num_pattern=2,
                                                                 leftFrame=leftFrame,
                                                                 rightFrame=rightFrame,
                                                                 leftCam_w=leftCam_fixedWidth,
                                                                 leftCam_h=leftCam_fixedHeight,
                                                                 rightCam_w=rightCam_fixedWidth,
                                                                 rightCam_h=rightCam_fixedHeight)

                    except Exception:
                        pass

                    # ================================================
                    #   3st pattern : If detect white spot disease
                    # ================================================
                    try:
                        # check_objects_category = (leftCam_objects_category[0]['name'] == rightCam_objects_category[0]['name'])
                        # check_count_objects = ((leftCam_count_objects == 1) and (rightCam_count_objects == 1))
                        #
                        # if check_objects_category and check_count_objects:
                        leftCam_check_white_spot_disease = ''
                        rightCam_check_white_spot_disease = ''

                        if (leftCam_x_min != None) and (leftCam_x_max != None) and (leftCam_y_min != None) and (leftCam_y_max != None):
                            cropLeft_object = leftFrame[leftCam_y_min:leftCam_y_max, leftCam_x_min:leftCam_x_max]
                            h = leftCam_y_max-leftCam_y_min
                            w = leftCam_x_max-leftCam_x_min
                            resizeLeft = cv2.resize(cropLeft_object, (w*3, h*3))
                            leftCam_check_white_spot_disease = check_behavior_pattern_3st(frame=resizeLeft,
                                                                                          title='frontCam')

                        if (rightCam_x_min != None) and (rightCam_x_max != None) and (rightCam_y_min != None) and (rightCam_y_max != None):
                            cropRight_object = rightFrame[rightCam_y_min:rightCam_y_max, rightCam_x_min:rightCam_x_max]
                            h = rightCam_y_max-rightCam_y_min
                            w = rightCam_x_max-rightCam_x_min
                            resizeRight = cv2.resize(cropRight_object, (w*3, h*3))
                            rightCam_check_white_spot_disease = check_behavior_pattern_3st(frame=resizeRight,
                                                                                           title='sideCam')

                        if (leftCam_check_white_spot_disease == 'Detect white spot disease') and (rightCam_check_white_spot_disease == 'Detect white spot disease'):
                            display_detect_abnormal_behavior(num_pattern=3,
                                                             leftFrame=leftFrame,
                                                             rightFrame=rightFrame,
                                                             leftCam_w=leftCam_fixedWidth,
                                                             leftCam_h=leftCam_fixedHeight,
                                                             rightCam_w=rightCam_fixedWidth,
                                                             rightCam_h=rightCam_fixedHeight)

                    except Exception:
                        pass

                    countFrame += 1
                    if inference_graph[:16] == 'ssd_inception_v2':
                        checkFrame = int(float("{0:.1f}".format(1/sec)))
                    elif inference_graph[:14] == 'rfcn_resnet101':
                        checkFrame = int(float("{0:.1f}".format(1/sec)))
                    if countFrame > checkFrame:
                        countFrame = 0
                    if datetime.datetime.now() >= endTime:
                        fileNum += 1
                        endTime = datetime.datetime.now() + datetime.timedelta(minutes=60)

                    output_rec_video(frame=leftFrame,
                                     rec_camera=rec_leftCam)
                    output_rec_video(frame=rightFrame,
                                     rec_camera=rec_rightCam)

                    # ================================================
                    #   Adjusting video resolution (width, height)
                    # ================================================
                    # vertical_frame = cv2.vconcat([leftFrame, rightFrame])
                    horizon_frame = cv2.hconcat([leftFrame, rightFrame])
                    cv2.imshow('Detector', cv2.resize(horizon_frame, (720 * 2, 720)))

                    # depth_frame = disparity / DEPTH_VISUALIZATION_SCALE
                    # cv2.imshow('Depth', cv2.resize(depth_frame, (640, 480)))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except Exception:
                    pass

            leftCam.release()
            rightCam.release()
            cv2.destroyAllWindows()
