'''
    Abnormal Behavior Detection

        1st pattern: If object stay on the edge of the screen for a long time and abnormal behavior detection area
        2st pattern: If the movement is noticeably slower or faster
        3st pattern: If detect white spot disease

'''
import numpy as np
import cv2
import math


class AbnormalBehaviorDetection:

    def __init__(self):
        pass

    def display(self, num_pattern, leftFrame, rightFrame, leftCam_w, leftCam_h, rightCam_w, rightCam_h):
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
        # elif num_pattern == 3:
        #     display_color = cv_color_white
        #     cv2.putText(leftFrame, 'Detect white spot disease!!!', (10, 40), cv2.FONT_HERSHEY_PLAIN, text_size, display_color)
        #     cv2.putText(rightFrame, 'Detect white spot disease!!!', (10, 40), cv2.FONT_HERSHEY_PLAIN, text_size, display_color)

        cv2.rectangle(leftFrame, (0, 0), (leftCam_w, leftCam_h), display_color, 20)
        cv2.rectangle(rightFrame, (0, 0), (rightCam_w, rightCam_h), display_color, 20)

    def pattern_1st(self, x1, x2, y1, y2, z1, z2, patternArr_size, patternArr):
        count = 0
        for i in range(len(patternArr)):
            coordinates_x = (patternArr[i][0])[0]
            coordinates_y = (patternArr[i][0])[1]
            depth = (patternArr[i][0])[2]
            if (x1 < coordinates_x < x2) and (y1 < coordinates_y < y2) and (z1 < depth < z2):
                count += 1

        # Detection of abnormal behavior when more than 90% of data are included in the area
        if int(count / patternArr_size * 100) >= 90:
            return 'Detect abnormal behavior'
        else:
            return 'Not detected'

    def speed_of_three_dimensional(self, resolution_x, resolution_y, resolution_z, coordinates_x1, coordinates_x2, coordinates_y1, coordinates_y2, depth1, depth2, time1, time2):
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

        return dist / (time2 - time1)

    def pattern_2st(self, speed, queue_size_of_speed, queue_of_speed):
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

    def erosionMsk(self, frame):
        kernel = np.ones((7, 7), np.uint8)
        frame = cv2.erode(frame, kernel, iterations=1)

        return frame

    def dilationMsk(self, frame):
        kernel = np.ones((7, 7), np.uint8)
        frame = cv2.dilate(frame, kernel, iterations=1)

        return frame

    def pattern_3st(self, frame, title):
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 140

        # Filter by Area
        params.filterByArea = True
        params.minArea = 0.01
        params.maxArea = 50

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.7

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.7

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
        frame = cv2.drawKeypoints(frame, keyPoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show blobs
        # cv2.imshow(title, frame)

        if( len(keyPoints) > 20):

            return 'Detect white spot disease'


'''
    Range of Abnormal Behavior Detection
'''
class RangeOfAbnormalBehaviorDetection:

    def __init__(self, leftFrame, rightFrame, range_x, range_y, leftCam_object_point, rightCam_object_point):

        self.leftFrame = leftFrame
        self.rightFrame = rightFrame

        self.leftHeight, self.leftWidth, _ = self.leftFrame.shape
        self.rightHeight, self.rightWidth, _ = self.rightFrame.shape

        self.range_x = range_x
        self.range_y = range_y

        self.leftCam_object_point = leftCam_object_point
        self.rightCam_object_point = rightCam_object_point

        self.cv_color_blue = (255, 0, 0)
        self.cv_color_green = (0, 255, 0)
        self.cv_color_red = (0, 0, 255)

        self.rectangle_line_thickness = 1

    def line(self):
        # The front of the Smart-Aquarium
        # Left & Upper
        cv2.rectangle(self.leftFrame, (0, 0), (self.range_x, self.range_y), self.cv_color_blue, self.rectangle_line_thickness)
        # Right & Upper
        cv2.rectangle(self.leftFrame, (self.leftWidth - self.range_x, 0), (self.leftWidth, self.range_y), self.cv_color_blue, self.rectangle_line_thickness)
        # Left & Bottom
        cv2.rectangle(self.leftFrame, (0, self.leftHeight - self.range_y), (self.range_x, self.leftHeight), self.cv_color_blue, self.rectangle_line_thickness)
        # Right & Bottom
        cv2.rectangle(self.leftFrame, (self.leftWidth - self.range_x, self.leftHeight - self.range_y), (self.leftWidth, self.leftHeight), self.cv_color_blue, self.rectangle_line_thickness)

        # The back of the Smart-Aquarium
        # Left & Upper
        cv2.rectangle(self.rightFrame, (0, 0), (self.range_x, self.range_y), self.cv_color_blue, self.rectangle_line_thickness)
        # Right & Upper
        cv2.rectangle(self.rightFrame, (self.rightWidth - self.range_x, 0), (self.rightWidth, self.range_y), self.cv_color_blue, self.rectangle_line_thickness)
        # Left & Bottom
        cv2.rectangle(self.rightFrame, (0, self.rightHeight - self.range_y), (self.range_x, self.rightHeight), self.cv_color_blue, self.rectangle_line_thickness)
        # Right & Bottom
        cv2.rectangle(self.rightFrame, (self.rightWidth - self.range_x, self.rightHeight - self.range_y), (self.rightWidth, self.rightHeight), self.cv_color_blue, self.rectangle_line_thickness)

    def line2(self):
        # The front of the Smart-Aquarium
        # Left & Upper
        if (0 <= self.leftCam_object_point[0] <= self.range_x) and (0 <= self.leftCam_object_point[1] <= self.range_y):
            cv2.rectangle(self.leftFrame, (0, 0), (self.range_x, self.range_y), self.cv_color_red, self.rectangle_line_thickness)
        else:
            cv2.rectangle(self.leftFrame, (0, 0), (self.range_x, self.range_y), self.cv_color_blue, self.rectangle_line_thickness)

        # Right & Upper
        if (self.leftWidth - self.range_x <= self.leftCam_object_point[0] <= self.leftWidth) and (0 <= self.leftCam_object_point[1] <= self.range_y):
            cv2.rectangle(self.leftFrame, (self.leftWidth - self.range_x, 0), (self.leftWidth, self.range_y), self.cv_color_red, self.rectangle_line_thickness)
        else:
            cv2.rectangle(self.leftFrame, (self.leftWidth - self.range_x, 0), (self.leftWidth, self.range_y), self.cv_color_blue, self.rectangle_line_thickness)

        # Left & Bottom
        if (0 <= self.leftCam_object_point[0] <= self.range_x) and (self.leftHeight - self.range_y <= self.leftCam_object_point[1] <= self.leftHeight):
            cv2.rectangle(self.leftFrame, (0, self.leftHeight - self.range_y), (self.range_x, self.leftHeight), self.cv_color_red, self.rectangle_line_thickness)
        else:
            cv2.rectangle(self.leftFrame, (0, self.leftHeight - self.range_y), (self.range_x, self.leftHeight), self.cv_color_blue, self.rectangle_line_thickness)

        # Right & Bottom
        if (self.leftWidth - self.range_x <= self.leftCam_object_point[0] <= self.leftWidth) and (self.leftHeight - self.range_y <= self.leftCam_object_point[1] <= self.leftHeight):
            cv2.rectangle(self.leftFrame, (self.leftWidth - self.range_x, self.leftHeight - self.range_y), (self.leftWidth, self.leftHeight), self.cv_color_red, self.rectangle_line_thickness)
        else:
            cv2.rectangle(self.leftFrame, (self.leftWidth - self.range_x, self.leftHeight - self.range_y), (self.leftWidth, self.leftHeight), self.cv_color_blue, self.rectangle_line_thickness)

        # The back of the Smart-Aquarium
        # Left & Upper
        if (0 <= self.rightCam_object_point[0] <= self.range_x) and (0 <= self.rightCam_object_point[1] <= self.range_y):
            cv2.rectangle(self.rightFrame, (0, 0), (self.range_x, self.range_y), self.cv_color_red, self.rectangle_line_thickness)
        else:
            cv2.rectangle(self.rightFrame, (0, 0), (self.range_x, self.range_y), self.cv_color_blue, self.rectangle_line_thickness)

        # Right & Upper
        if (self.leftWidth - self.range_x <= self.rightCam_object_point[0] <= self.leftWidth) and (0 <= self.rightCam_object_point[1] <= self.range_y):
            cv2.rectangle(self.rightFrame, (self.rightWidth - self.range_x, 0), (self.rightWidth, self.range_y), self.cv_color_red, self.rectangle_line_thickness)
        else:
            cv2.rectangle(self.rightFrame, (self.rightWidth - self.range_x, 0), (self.rightWidth, self.range_y), self.cv_color_blue, self.rectangle_line_thickness)

        # Left & Bottom
        if (0 <= self.rightCam_object_point[0] <= self.range_x) and (self.leftHeight - self.range_y <= self.rightCam_object_point[1] <= self.leftHeight):
            cv2.rectangle(self.rightFrame, (0, self.rightHeight - self.range_y), (self.range_x, self.rightHeight), self.cv_color_red, self.rectangle_line_thickness)
        else:
            cv2.rectangle(self.rightFrame, (0, self.rightHeight - self.range_y), (self.range_x, self.rightHeight), self.cv_color_blue, self.rectangle_line_thickness)

        # Right & Bottom
        if (self.leftWidth - self.range_x <= self.rightCam_object_point[0] <= self.leftWidth) and (self.leftHeight - self.range_y <= self.rightCam_object_point[1] <= self.leftHeight):
            cv2.rectangle(self.rightFrame, (self.rightWidth - self.range_x, self.rightHeight - self.range_y), (self.rightWidth, self.rightHeight), self.cv_color_red, self.rectangle_line_thickness)
        else:
            cv2.rectangle(self.rightFrame, (self.rightWidth - self.range_x, self.rightHeight - self.range_y), (self.rightWidth, self.rightHeight), self.cv_color_blue, self.rectangle_line_thickness)
