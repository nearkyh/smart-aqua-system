import numpy as np
import cv2


def get_resolution(capture):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height


def camera_calibration(image_np, width, height):
    # Calibration dictionary (ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    # calibration_data = np.load('stereoCam_calibration.npz')
    # cameraMatrix = cameraMatrix,
    # cameraMatrix2 = cameraMatrix2,
    # distCoeffs = distCoeffs,
    # distCoeffs2 = distCoeffs2,
    # rvecs = self.rvecs,
    # rvecs2 = self.rvecs2,
    # tvecs = self.tvecs,
    # tvecs2 = self.tvecs2,
    # R = R,
    # T = T,
    # E = E,
    # F = F

    
    # Camera parameters ([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) 
    # mtx = calibration_data['mtx']
    mtx = np.array([[633.50371582, 0., 473.89946909],
                    [0., 634.45000046, 369.61650763],
                    [0., 0., 1.]])
    
    # Distortion coefficients = ([k1, k2, p1, p2, k3)
    # dist = calibration_data['dist']
    dist = np.array([[-0.33912713], [0.17213131], [0], [0], [0]])

    # objpoints = calibration_data['objpoints']
    # imgpoints = calibration_data['imgpoints']

    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 0)

    # 1. Using cv2.undistort()
    # dst = cv2.undistort(image_np, mtx, dist, None, new_camera_mtx)
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]

    # 2. Using remapping
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (width, height), 5)
    dst = cv2.remap(image_np, map_x, map_y, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    # mean_error = 0
    # for i in range(len(objpoints)):
    #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    #     mean_error += error
    # print("total error: {}".format(mean_error / len(objpoints)))

    return dst


cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

width, height = get_resolution(cap)
width2, height2 = get_resolution(cap2)

while True:
    ret, image_np = cap.read()
    ret2, image_np2 = cap2.read()

    image_np = camera_calibration(image_np=image_np,
                                  width=width,
                                  height=height)

    # image_np2 = camera_calibration(image_np=image_np2,
    #                                width=width2,
    #                                height=height2)

    final_image_np = cv2.hconcat([image_np, image_np2])

    cv2.imshow('Calibration', cv2.resize(final_image_np, (1280, 480)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
