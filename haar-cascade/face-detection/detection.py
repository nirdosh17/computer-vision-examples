import cv2
import argparse

SMILE_CLASSIFIER_PATH = 'haar-cascade/classifiers/haarcascade_smile.xml'
RESULT_FILE = 'result_haar_face_detection.jpg'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-c", "--face-classifier", type=str, required=True,
	help="path to HaarCascade face classifier")
args = vars(ap.parse_args())

# img = cv2.imread('training/face-detection/single.jpg')
img = cv2.imread(args["image"])
classifier_path = args["face_classifier"]

# convert to gray scale to improve computational efficiency.
# we only have to deal with intensity of a single color instead of multiple colors, hence less computation
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# histogram equalization helps areas of lower local contrast to gain a higher contrast which makes image sharper and edge detection is easier
# https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
gray = cv2.equalizeHist(gray)

# cv2.imwrite('detection_gray.jpg', gray)

# sample haarcascade classifiers(face, cat, eyes e.t.c) can be found in this Github repo
# https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
haar_cascade = cv2.CascadeClassifier(classifier_path)
smile_cascade = cv2.CascadeClassifier(SMILE_CLASSIFIER_PATH)

# list of coordinates of bounding box for each detected faces
# noise or false positives can be minimized by changing sensitivity i.e. minNeighbors
# e.g more faces are found when minNeighbors is 1 and less are found if changed to 6
# haarcascade is popular and easily to use with minimal setup but not so advanced, dlibs face recognizer is better

# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# trying other configs
faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)

print(f'Number of faces found = {len(faces_rect)}')

# looping through coordinates and drawing rectangle in all faces
for (x, y, w, h) in faces_rect:
  cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=3)
  roi_gray = gray[y:y + h, x:x + w]
  roi_color = img[y:y + h, x:x + w]
  smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 13)

  # draw rectangles for smiles
  # for (sx, sy, sw, sh) in smiles:
  #   cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)


cv2.imwrite(RESULT_FILE, img)
cv2.imshow("HaarCascade Detection (Press 'q' to quit)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
