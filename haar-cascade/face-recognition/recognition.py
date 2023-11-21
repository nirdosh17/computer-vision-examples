import os
import argparse
import cv2
import numpy as np

MODEL_FILE_PATH = 'haar-cascade/face-recognition/model/face_trained_model.yml'
RESULT_FILE = 'result_haar_face_recognition.jpg'

# reading all folder name so that we can use them as labels later on
def get_folder_names(dir):
  labels = []
  for i in os.listdir(dir):
    path = os.path.join(dir, i)
    if not i.startswith('.') and os.path.isdir(path):
      labels.append(i)
  return labels

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-c", "--face-classifier", type=str, required=True,
	help="path to HaarCascade face classifier")
ap.add_argument("-td", "--training-dataset", type=str, required=True,
	help="training folder is only used to extract sub folder names(person) and use them as labels")
args = vars(ap.parse_args())

people = get_folder_names(args["training_dataset"])

haar_cascade = cv2.CascadeClassifier(args["face_classifier"])

# features = np.load('features.npy')
# labels = np.load('labels.npy')

# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100.0)
face_recognizer.read(MODEL_FILE_PATH)

img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30),
#                                         flags=cv2.CASCADE_SCALE_IMAGE)



# loop through all faces as the image could contain multiple persons
for (x,y,w,h) in faces_rect:
  faces_roi = gray[y:y+h, x:x+w]
  label, confidence = face_recognizer.predict(faces_roi)

  # if confidence > 60:
  print(f'Label = {people[label]} | Confidence = {confidence}')
  cv2.putText(img, str(people[label]), (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=1)
  cv2.putText(img, str(round(confidence))+'%', (x,y+50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=1)
  cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv2.imwrite(RESULT_FILE, img)
cv2.imshow("HaarCascade Recognition (Press 'q' to quit)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
