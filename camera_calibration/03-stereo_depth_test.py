from __future__ import print_function
import sys
import numpy as np
import cv2

REMAP_INTERPOLATION = cv2.INTER_LINEAR

if len(sys.argv) != 2:
    print("Syntax: {0} CALIBRATION_FILE".format(sys.argv[0]))
    sys.exit(1)

calibration = np.load(sys.argv[1], allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])


CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# TODO: Use more stable identifiers
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# Increase the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# The distortion in the cap and cap2 edges prevents a good calibration, so
# discard the edges
CROP_WIDTH = 960
def cropHorizontal(image):
    return image[:,
            int((CAMERA_WIDTH-CROP_WIDTH)/2):
            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]

# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images

def nothing(x):
    pass
#Trackbar callback function
def onTrackbarChange(trackbarValue):
    num_disp = 240 - trackbarValue * 16
# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('depth map')
num_disp = 16
# create trackbars for color change
cv2.createTrackbar('min_disp','depth map',1,7,onTrackbarChange)
#cv2.createTrackbar('max_disp','depth map',5,15,nothing)
cv2.createTrackbar('window_size','depth map',5,150,nothing)
cv2.createTrackbar('Disp12MaxDiff_','depth map',1,10,nothing)
cv2.createTrackbar('uniquenessRatio','depth map',1,50,nothing)
cv2.createTrackbar('speckleRange','depth map',1,100,nothing)
cv2.createTrackbar('speckleWindow','depth map',1,255,nothing)
cv2.createTrackbar('DEPTH_VISUALIZATION_SCALE','depth map',1,2014,nothing)
# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    if not cap.grab() or not cap2.grab():
        print("No more frames")
        break

    _, leftFrame = cap.retrieve()
    leftFrame = cropHorizontal(leftFrame)
    leftHeight, leftWidth = leftFrame.shape[:2]
    _, rightFrame = cap2.retrieve()
    rightFrame = cropHorizontal(rightFrame)
    rightHeight, rightWidth = rightFrame.shape[:2]
    if (leftWidth, leftHeight) != imageSize:
        print("Left camera has different size than the calibration data")
        break

    if (rightWidth, rightHeight) != imageSize:
        print("Right camera has different size than the calibration data")
        break

    fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)


    stereo = cv2.StereoSGBM_create(0,122,16)
    # get current positions of four trackbars
    #min_disp = cv2.getTrackbarPos('min_disp', 'depth map')
    #max_disp = cv2.getTrackbarPos('max_disp', 'depth map')
    #max_dusp = max_disp * (16 + min_disp % 16)
    #num_disp = max_disp - min_disp
    #if(max_disp <= min_disp) :
    #    max_disp = 240
    #window_size = cv2.getTrackbarPos('window_size', 'depth map')
    #if window_size % 2 == 0:
    #     window_size = window_size + 1
    # Disp12MaxDiff_ = cv2.getTrackbarPos('Disp12MaxDiff_', 'depth map')
    # uniquenessRatio_ = cv2.getTrackbarPos('uniquenessRatio', 'depth map')
    # speckleRange_ = cv2.getTrackbarPos('speckleRange', 'depth map')
    # speckleWindow_ = cv2.getTrackbarPos('speckleWindow', 'depth map')
    # DEPTH_VISUALIZATION_SCALE = cv2.getTrackbarPos('DEPTH_VISUALIZATION_SCALE', 'depth map')
    #
    # stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=window_size)
    # stereo.setMinDisparity(min_disp)
    # stereo.setNumDisparities(num_disp)
    # stereo.setBlockSize(window_size)
    # stereo.setDisp12MaxDiff(Disp12MaxDiff_)
    # stereo.setUniquenessRatio(uniquenessRatio_)
    # stereo.setSpeckleRange(speckleRange_)
    # stereo.setSpeckleWindowSize(speckleWindow_)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereo.compute(grayLeft, grayRight)
    disp_map = (depth - min_disp) / DEPTH_VISUALIZATION_SCALE
    cv2.imshow('left', fixedLeft)
    cv2.imshow('right', fixedRight)
    cv2.imshow('depth', disp_map)
    horizon_frame = cv2.hconcat([fixedLeft, fixedRight])
    cv2.imshow('Test', cv2.resize(horizon_frame, (640 * 2, 480)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cap2.release()
cv2.destroyAllWindows()
