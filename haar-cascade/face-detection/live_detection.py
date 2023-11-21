# does not run inside a container
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-fc", "--face-classifier", type=str, required=True,
	help="path to HaarCascade face classifier")
ap.add_argument("-ec", "--eye-classifier", type=str, required=True,
	help="path to HaarCascade eye classifier")
ap.add_argument("-sc", "--smile-classifier", type=str, required=True,
	help="path to HaarCascade smile classifier")
ap.add_argument("-gc", "--glasses-classifier", type=str, required=True,
	help="path to HaarCascade glasses classifier")
args = vars(ap.parse_args())

face_cascade = cv2.CascadeClassifier(args["face_classifier"])
eye_cascade = cv2.CascadeClassifier(args["eye_classifier"])
smile_cascade = cv2.CascadeClassifier(args["smile_classifier"])
glasses_cascade = cv2.CascadeClassifier(args["glasses_classifier"])

def detect(gray, frame):
  faces = face_cascade.detectMultiScale(gray, 1.3, 6)
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]
    smiles = smile_cascade.detectMultiScale(roi_gray, 1.5, 25)

    for (sx, sy, sw, sh) in smiles:
      cv2.putText(roi_color, 'smiling', (sx,sy-5), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=1)
      cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 255, 0), 2)

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.8, 20)
    for (sx, sy, sw, sh) in eyes:
      cv2.putText(roi_color, 'eyes', (sx,sy-5), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), thickness=1)
      cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (255, 255, 255), 2)

    # glasses = glasses_cascade.detectMultiScale(roi_gray, 1.8, 20)
    # for (sx, sy, sw, sh) in glasses:
    #   cv2.putText(roi_color, 'glasses', (sx,sy-5), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), thickness=1)
    #   cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 255, 255), 2)

  return frame

video_capture = cv2.VideoCapture(0)
while video_capture.isOpened():
   # Captures video_capture frame by frame
    _, frame = video_capture.read()

    # To capture image in monochrome
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # calls the detect() function
    canvas = detect(gray, frame)

    # Displays the result on camera feed
    cv2.imshow('Video', canvas)

    # The control breaks once q key is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the capture once all the processing is done.
video_capture.release()
cv2.destroyAllWindows()
