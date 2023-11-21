.DEFAULT_GOAL=help

TEST_IMAGE:="dataset/test-samples/friends.png"
TRAINING_DATASET:="dataset/faces"

HAAR_FACE_CLASSIFIER:="haar-cascade/classifiers/haarcascade_frontalface_alt2.xml"
HAAR_SMILE_CLASSIFIER:="haar-cascade/classifiers/haarcascade_smile.xml"
HAAR_EYE_CLASSIFIER:="haar-cascade/classifiers/haarcascade_eye.xml"
HAAR_GLASSES_CLASSIFIER:="haar-cascade/classifiers/haarcascade_eye_tree_eyeglasses.xml"

DLIB_FACE_LANDMARK_MODEL:="dlib/models/shape_predictor_68_face_landmarks.dat"

install: ## install all python dependencies
	pip install -r requirements.txt

build: ## build docker image with all OpenCV packages installed
	docker build -t opencv-playground .

console: ## SSH to docker container to try out OpenCV and other libraries
	docker run -it -v ./:/opencv opencv-playground bash

live-detect: ## opens web cam and starts face detection
	@python3 haar-cascade/face-detection/live_detection.py \
		--face-classifier $(HAAR_FACE_CLASSIFIER) \
		--smile-classifier $(HAAR_SMILE_CLASSIFIER) \
		--eye-classifier $(HAAR_EYE_CLASSIFIER) \
		--glasses-classifier $(HAAR_GLASSES_CLASSIFIER)

haar-detect: ## detect face in given image using HaarCascade
	@python3 haar-cascade/face-detection/detection.py \
		--image $(TEST_IMAGE) \
		--face-classifier $(HAAR_FACE_CLASSIFIER)

haar-train: ## train face recognition model for HaarCascade
	@python3 haar-cascade/face-recognition/train.py \
		--training-dataset $(TRAINING_DATASET) \
		--face-classifier $(HAAR_FACE_CLASSIFIER)

haar-recognize: ## run face recognition for given image using last generated HaarCascade model
	@python3 haar-cascade/face-recognition/recognition.py \
		--image $(TEST_IMAGE) \
		--face-classifier $(HAAR_FACE_CLASSIFIER) \
		--training-dataset $(TRAINING_DATASET)

hog-detect: ## face detection using Dlibs(HOG + Linear SVM)
	python3 dlib/face-detection/hog_face_detection.py --upsample 2 --image $(TEST_IMAGE)

cnn-detect: ## face detection using Dlibs(MMOD CNN)
	python3 dlib/face-detection/cnn_face_detection.py --upsample 2 --image $(TEST_IMAGE)

live-facial-landmarks: ## shows facial landmarks (head, mouth, eyes e.t.c) in live webcam feed
	python3 dlib/face-detection/facial_landmarks_live.py --model $(DLIB_FACE_LANDMARK_MODEL)

run-all: ## try both face detection and recognition in OpenCV and Dlib
	make haar-detect haar-recognize hog-detect cnn-detect


help:
	@grep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
