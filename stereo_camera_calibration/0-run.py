'''
    USB CAMERA INFO

        Model Name : USB8MP02G

        Maximum Image Transfer Rate :
        3264X2448  MJPG  15fps  YUY2  2fps
        2592X1944  MJPG  15fps  YUY2  3fps
        2048X1536  MJPG  20fps  YUY2  3fps
        1600X1200  MJPG  20fps  YUY2  10fps
        1280X960   MJPG  20fps  YUY2  10fps
        1024X768   MJPG  30fps  YUY2  10fps
        800X600    MJPG  30fps  YUY2  30fps
        640X480    MJPG  30fps  YUY2  30fps

'''
import cv2


leftCam = cv2.VideoCapture(0)
rightCam = cv2.VideoCapture(1)

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960

# Increase the resolution
leftCam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
leftCam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
rightCam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
rightCam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

CROP_WIDTH = 960
def cropHorizontal(image):
    return image[:,
           int((CAMERA_WIDTH-CROP_WIDTH)/2):
           int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]

while(True):
    if not (leftCam.grab() and rightCam.grab()):
        print("No more frames")
        break

    _, leftFrame = leftCam.retrieve()
    _, rightFrame = rightCam.retrieve()

    # leftFrame = cropHorizontal(leftFrame)
    # rightFrame = cropHorizontal(rightFrame)

    cv2.imshow('left', leftFrame)
    cv2.imshow('right', rightFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

leftCam.release()
rightCam.release()
cv2.destroyAllWindows()
