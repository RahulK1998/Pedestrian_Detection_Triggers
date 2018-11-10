# import the necessary packages
# source - https://github.com/geekysethi/pedestrian-detection ;
# source 2 - https://github.com/AdityaPai2398/Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades/tree/master/Main%20Project/Main%20Project/Pedestrian%20Detection


from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import time
import datetime


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=300, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    cap = VideoStream(src=1).start()
    time.sleep(2.0)

# otherwise, we are reading from a video file
else:
    cap = cv2.VideoCapture(args["video"])


fgbg = cv2.createBackgroundSubtractorMOG2()
bike_cascade = cv2.CascadeClassifier('haarcascade_pedestrian.xml')

while True:
    start_time = time.time()
    # grab the current frame and initialize the occupied/unoccupied
    # text
    frame = cap.read()
    frame = frame if args.get("video", None) is None else frame[1]


    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    img = imutils.resize(frame, width=500)

    fgbg.apply(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bike = bike_cascade.detectMultiScale(gray, 1.3, 2)
    image = img.copy()
    for (a, b, c, d) in bike:
        cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 210), 4)


    bike = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bike])
    pick = non_max_suppression(bike, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # cv2.imshow('video', img)
    cv2.imshow('video2 after NMS',image)

    print("FPS: ", 1.0 / (time.time() - start_time))
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
cap.stop() if args.get("video", None) is None else cap.release()
cv2.destroyAllWindows()

#21 fps