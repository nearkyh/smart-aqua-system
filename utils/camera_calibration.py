'''
    Stereo Camera Calibration
'''
import numpy as np
import cv2


class CameraCalibration:

    def __init__(self, cap1, cap2, camWidth, camHeight, cropWidth):
        self.REMAP_INTERPOLATION = cv2.INTER_LINEAR
        self.DEPTH_VISUALIZATION_SCALE = 2048
        self.STEREO_CALIBRATION_DATA = 'camera_calibration/calibration.npz'
        print("Syntax: {0} CALIBRATION_FILE".format(self.STEREO_CALIBRATION_DATA))
        self.calibration = np.load(self.STEREO_CALIBRATION_DATA, allow_pickle=False)

        # get imageSize
        self.imageSize = tuple(self.calibration["imageSize"])

        # getMatchingObjectAndImagePoints
        self.leftObjectPoints = self.calibration["leftObjectPoints"]
        self.leftImagePoints = self.calibration["leftImagePoints"]
        self.rightObjectPoints = self.calibration["rightObjectPoints"]
        self.rightImagePoints = self.calibration["rightImagePoints"]

        # calibrateCamera
        self.leftCameraMatrix = self.calibration["leftCameraMatrix"]
        self.leftDistortionCoefficients = self.calibration["leftDistortionCoefficients"]
        self.rightCameraMatrix = self.calibration["rightCameraMatrix"]
        self.rightDistortionCoefficients = self.calibration["rightDistortionCoefficients"]

        # stereoCalibrate
        self.rotationMatrix = self.calibration["rotationMatrix"]
        self.translationVector = self.calibration["translationVector"]

        # stereoRectify
        self.leftRectification = self.calibration["leftRectification"]
        self.rightRectification = self.calibration["rightRectification"]
        self.leftProjection = self.calibration["leftProjection"]
        self.rightProjection = self.calibration["rightProjection"]
        self.dispartityToDepthMap = self.calibration["dispartityToDepthMap"]
        self.leftROI = tuple(self.calibration["leftROI"])
        self.rightROI = tuple(self.calibration["rightROI"])

        # initUndistortRectifyMap
        self.leftMapX = self.calibration["leftMapX"]
        self.leftMapY = self.calibration["leftMapY"]
        self.rightMapX = self.calibration["rightMapX"]
        self.rightMapY = self.calibration["rightMapY"]

        self.stereoMatcher = cv2.StereoBM_create()

        self.cap1 = cap1
        self.cap2 = cap2

        self.camWidth = camWidth
        self.camHeight = camHeight
        self.cropWidth = cropWidth

    def set_stereoMatcher(self):
        # TODO: Try applying brightness/contrast/gamma adjustments to the images.
        self.stereoMatcher.setMinDisparity(4)
        self.stereoMatcher.setNumDisparities(128)
        self.stereoMatcher.setBlockSize(21)
        self.stereoMatcher.setROI1(self.leftROI)
        self.stereoMatcher.setROI2(self.rightROI)
        self.stereoMatcher.setSpeckleRange(16)
        self.stereoMatcher.setSpeckleWindowSize(45)

    def set_resolution(self, cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cropWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camHeight)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


    def crop_horizontal(self, image, camWidth, cropWidth):
        return image[:,
               int((camWidth - cropWidth) / 2):
               int(cropWidth + (camWidth - cropWidth) / 2)]

    def stereo_calibration(self, frame1, frame2):
        leftFrame = self.crop_horizontal(image=frame1,
                                         camWidth=self.camWidth,
                                         cropWidth=self.cropWidth)
        leftHeight, leftWidth = leftFrame.shape[:2]

        rightFrame = self.crop_horizontal(image=frame2,
                                          camWidth=self.camWidth,
                                          cropWidth=self.cropWidth)
        rightHeight, rightWidth = rightFrame.shape[:2]

        if (leftWidth, leftHeight) != self.imageSize:
            print("Left camera has different size than the calibration data")
            exit()
        if (rightWidth, rightHeight) != self.imageSize:
            print("Right camera has different size than the calibration data")
            exit()

        leftFrame = cv2.remap(leftFrame, self.leftMapX, self.leftMapY, self.REMAP_INTERPOLATION)
        rightFrame = cv2.remap(rightFrame, self.rightMapX, self.rightMapY, self.REMAP_INTERPOLATION)

        return leftFrame, rightFrame

    def depth_map_creator(self, frame1, frame2):
        grayLeft = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        disparity = self.stereoMatcher.compute(grayLeft, grayRight)
        coordinates_of_3D = cv2.reprojectImageTo3D(disparity, self.dispartityToDepthMap)

        return disparity, coordinates_of_3D

    def triangulation(self, leftCam_point, rightCam_point):
        R = self.rotationMatrix
        T = self.translationVector
        P1 = self.leftProjection
        P2 = self.rightProjection

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
