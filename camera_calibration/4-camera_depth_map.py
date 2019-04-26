import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_resolution(capture):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height


def camera_calibration(image_np, width, height):
    # Calibration dictionary (ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    calibration_data = np.load('calibration.npz')

    # Camera parameters ([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    mtx = calibration_data['mtx']
    # mtx = np.array([[477.59130484, 0., 332.65514887],
    #               [0., 480.76513178, 239.42255985],
    #               [0., 0., 1.]])

    # Distortion coefficients = ([k1, k2, p1, p2, k3)
    # dist = calibration_data['dist']
    dist = np.array([-0.30237751, 0.06247485, 0, 0, 0])

    objpoints = calibration_data['objpoints']
    imgpoints = calibration_data['imgpoints']
    rvecs = calibration_data['rvecs']
    tvecs = calibration_data['tvecs']

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

    '''
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))
    '''

    return dst


def draw_lines(image_np, image_np2, lines, pts1, pts2):
    ''' image_np - image on which we draw the epilines for the points in image_np2
        lines - corresponding epilines '''
    r, c = image_np.shape[:2]
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    # image_np2 = cv2.cvtColor(image_np2, cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        image_np = cv2.line(image_np, (x0,y0), (x1,y1), color, 1)
        image_np = cv2.circle(image_np, tuple(pt1), 5, color, -1)
        image_np2 = cv2.circle(image_np2, tuple(pt2), 5, color, -1)

    return image_np, image_np2


def epipolar_geometry(image_np, image_np2):
    # sift = cv2.SIFT()
    # If Error=="AttributeError: module 'cv2.cv2' has no attribute 'SIFT')"
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_np, None)
    kp2, des2 = sift.detectAndCompute(image_np2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = draw_lines(image_np, image_np2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = draw_lines(image_np2, image_np, lines2, pts2, pts1)

    return img5, img3


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

    image_np2 = camera_calibration(image_np=image_np2,
                                   width=width2,
                                   height=height2)

    # image_np, image_np2 = epipolar_geometry(image_np=image_np,
    #                                         image_np2=image_np2)

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    image_np2 = cv2.cvtColor(image_np2, cv2.COLOR_BGR2GRAY)

    stereoMatcher = cv2.StereoSGBM_create()
    stereoMatcher.setMinDisparity(4)
    stereoMatcher.setNumDisparities(128)
    stereoMatcher.setBlockSize(21)
    stereoMatcher.setSpeckleRange(16)
    stereoMatcher.setSpeckleWindowSize(45)
    disparity = stereoMatcher.compute(image_np, image_np2)

    # compute disparity
    disparity = stereoMatcher.compute(image_np, image_np2).astype(np.float32) / 16.0
    disparity = (disparity - 4) / 128
    # disparity = cv2.threshold(disparity, 0.6, 1.0, cv2.THRESH_BINARY)[1]

    cv2.imshow('Calibration', disparity)

    # final_image_np = cv2.hconcat([image_np2, image_np])
    # cv2.imshow('Epipolar Geometry', cv2.resize(final_image_np, (1280, 480)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
