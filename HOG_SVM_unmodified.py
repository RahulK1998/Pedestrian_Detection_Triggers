# import the necessary packages
# source - Pyimage search
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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", help="path to images directory")
ap.add_argument("-v", "--video")
args = vars(ap.parse_args())

if args.get("video", None) is None:
    vs = VideoStream(src=1).start()
    time.sleep(2.0)
# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])


# initialize the first frame in the video stream
firstFrame = None

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# loop over the frames of the video


while True:
    start_time = time.time()
# grab the current frame and initialize the occupied/unoccupied
    # text
    frame = vs.read()

    frame = frame if args.get("video", None) is None else frame[1]
    text = "Unoccupied"

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    image = imutils.resize(frame, width=min(500, frame.shape[1]))

    orig = image.copy()
    cv2.imshow("real time",orig)
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show the output images
    #cv2.imshow("Before NMS", orig)
    cv2.imshow("Final footage", image)

    print("FPS: ", 1.0 / (time.time() - start_time))
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()

# initialize the HOG descriptor/person detector
## 6 fps max speed