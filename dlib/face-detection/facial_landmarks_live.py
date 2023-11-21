# displays facial landmarks (eyes, eyebrows, nose, jawline, mouth/lips)
from imutils import face_utils
import dlib
import cv2 as cv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
	help="path to dlib's face landmark recognition model")
args = vars(ap.parse_args())

model = args["model"]
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)

video_capture = cv.VideoCapture(0)
while video_capture.isOpened():
   # read video frame by frame and apply face detector
    _, frame = video_capture.read()

    # convert to gray scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    rects = face_detector(gray, 0)

    for (i, rect) in enumerate(rects):
      shape = predictor(gray, rect)
      shape = face_utils.shape_to_np(shape)

      # draw 68 points as circles
      for (x, y) in shape:
        cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # display manipulated image frame on the window
    cv.imshow('Facial Landmarks', frame)

    # quit window when 'q' is pressed
    if cv.waitKey(1) & 0xff == ord('q'):
      break

video_capture.release()
cv.destroyAllWindows()
