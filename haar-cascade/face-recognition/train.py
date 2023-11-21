import os
import argparse
import cv2
import numpy as np

MODEL_PATH = 'haar-cascade/face-recognition/model/face_trained_model.yml'

def get_folder_names(dir):
  labels = []
  for i in os.listdir(dir):
    path = os.path.join(dir, i)
    if not i.startswith('.') and os.path.isdir(path):
      labels.append(i)
  return labels

ap = argparse.ArgumentParser()
ap.add_argument("-td", "--training-dataset", type=str, required=True,
	help="training dataset container that contains person's name as folder and images of their faces")
ap.add_argument("-c", "--face-classifier", type=str, required=True,
	help="path to HaarCascade face classifier")
args = vars(ap.parse_args())

training_dataset = args["training_dataset"]

# reading all folder name so that we can use them as labels later on
people = get_folder_names(training_dataset)

# Training sets contains:
# - features: image array of the face
# - label: who's face is it
features = []
labels = []
haar_cascade = cv2.CascadeClassifier(args["face_classifier"])

def create_train():
  for person in people:
    path = os.path.join(training_dataset, person)

    # numeric label help to reduce computation so using index here
    # can be replaced by some kind of id
    label = people.index(person)

    for img in os.listdir(path):
      if not img.endswith('.jpg') and not img.endswith('.png'):
        print(f'skipped = {img}', end='\r')
        continue

      # single image path
      img_path = os.path.join(path, img)
      img_array = cv2.imread(img_path)

      gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
      gray = cv2.equalizeHist(gray)

      faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
      # faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(30, 30),
      #                                   flags=cv2.CASCADE_SCALE_IMAGE)


      # cropping out faces only while training
      for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        features.append(faces_roi)
        labels.append(label)

    print(f'processed folder = {person}')

create_train()
print('Training complete!')

print(f'Features count = {len(features)}')
print(f'Labels count = {len(labels)}')

# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100.0)


features = np.array(features, dtype='object')
labels = np.array(labels)

# train recognizer on feature and label list
face_recognizer.train(features, labels)

face_recognizer.save(MODEL_PATH)
print(f'Saved trained model in {MODEL_PATH}')

# np.save('features.npy', features)
# np.save('labels.npy', labels)
