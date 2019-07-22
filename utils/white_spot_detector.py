#!/usr/bin/python

# Standard imports
import cv2
import numpy as np

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 140


# Filter by Area.
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


cap = cv2.VideoCapture("video/white_spot_test03.mp4")
while(cap.isOpened()):
        ret, frame = cap.read()
        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
                detector = cv2.SimpleBlobDetector(params)
        else : 
                detector = cv2.SimpleBlobDetector_create(params)
        frame = ~frame       
        keypoints = detector.detect(frame)
        frame = ~frame       
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print("white spot number: ", len(keypoints))
        if( len(keypoints) >20):
                cv2.putText(im_with_keypoints, "white spot warning", (350,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        
        cv2.imshow("Keypoints", im_with_keypoints)
       
cap.release()
                                        
'''       
# Read image
im = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("1.jpg")
cv2.imshow("org", im2)
im= ~im

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
'''
