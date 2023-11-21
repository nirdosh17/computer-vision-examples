# Computer Vision Examples
Computer vision examples in python using OpenCV and other libraries.

At the moment, there are working examples only for face detection and face recognition algorithms like OpenCV and Dlib in Python. Object detection will be added later on.

## Installation
Make sure you have Python3 installed.
1. Clone the repo and cd to project root directory. We will be running all commands from the root dir.

2. To install dependencies run: `make install`

3. If you run `make help`, it shows list of available commands to run face detection and recognition algorithms on your images.

## Face Detection

In face detection, we are only interested to identify which parts of the image are human faces.

Libraries like OpenCV and Dlib provide open source classifiers which can be used with minimal configuration. Having said that, we need to tweak the configs to increase or decrease the sensitivity to filter out false positives.
We need not to train model for this.

Existing classifiers (HaarCascade, Dlib) are used to get coordinates of bounding box for face and we draw them in the image.

**[HaarCascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)**
- Simple, fast, good for smaller devices.
- Less accurate than other models like Dlib so needs a bit of fine tuning for correct results.
- Most used classifiers(face, eyes, cats, license plate) are offered by OpenCV.
- Good candidate for RaspberryPI as well.

Commands:
- Run detection in default test image:

  ```bash
  make haar-detect
  ``````

  ![Screenshot 2023-11-21 at 7 00 17 PM](https://github.com/nirdosh17/computer-vision-examples/assets/5920689/b60cba38-ce6b-4f8f-8d6b-098d031566db)

- Test on your image:
  ```bash
  make haar-detect TEST_IMAGE="dataset/test-samples/friends.jpg"
  ```

- Run live detection from webcam: `make live-detect`

  ![output](https://github.com/nirdosh17/computer-vision-examples/assets/5920689/e99bd7cb-91f9-4148-b0db-f8f8c275c748)


**[Dlib](http://dlib.net)**:


- **Face detection using Dlib(HOG + Linear SVM)**

  _Higher accuracy than HaarCascade, still faster than MMOD CNN._
  ```bash
  make hog-detect
  ```


- **Face detection using Dlib(MMOD CNN)**

  _Higher accuracy than Dlib HOG, but needs high compute, takes longer time, slower in smaller machines._
  ```bash
  make cnn-detect
  ```

- **Facial Landmark Detection**

  _Detect facial features like eyes, eyebrows, nose, mouth, lips._
  ```bash
  make live-facial-landmarks
  ```

  ![Screenshot 2023-11-21 at 7 43 05 PM](https://github.com/nirdosh17/computer-vision-examples/assets/5920689/d5c3b285-8d7a-41ef-92e8-c4275516a58e)


## Face Recognition

Face Recognition has few steps than detection because it is necessary to first detect our faces from the images and label them so that we can identify them later from our test image which will be not included in the training dataset.

In high level, these are the steps in face recognition using OpenCV:
- We extract faces using face detection classifiers.
- Use the facial data and label(e.g. person's name) and train it. What we get is a model which can be saved as a yml file.
- We then use OpenCV or other face recognizers to predict the label from given input(facial image data).
- We get label and a confidence score as the result.

### HaarCascade

- Using given classifier, we will be creating a model training it on our images(faces).
- One image should have one face.
- Training data set should contain variation of lighting, angles, background for better results.

#### 1. Dataset Preparation

We need sample images to train for positive and negative values which are inside `dataset/faces` folder.

  For each person, there is a unique folder where the images are kept. Folder name is important here as we use it as **Label** for image when detected.
  ```
    .datasets
    ├── faces
        ├── Chandler
        │   ├── 1.png
        │   ├── 2.png
        │   ├── ...
        │   └── 50.png
        ├── Joey
        │   ├── 1.png
        │   ├── 2.png
        │   ├── ...
        │   └── 50.png
        └── Unknown
            ├── 1.png
            ├── 2.png
            ├── ...
            └── 50.png

  ```
  Single training image should contain one face only.

#### 2. Train
Run this command which will apply HaarCascade classifier on our image folders, create a model and save it in yml file.

```bash
make haar-train
```

#### 3. Apply image recognition using the model on our target image

```bash
# default image
make haar-recognize
# run recognition on your own image
make haar-recognize TEST_IMAGE="dataset/test-samples/friends.png"
```

![Screenshot 2023-11-21 at 7 29 21 PM](https://github.com/nirdosh17/computer-vision-examples/assets/5920689/92e3b17c-a417-42cc-8482-35894693ecf9)
_It is not 100% accurate. Need to tweak the configs and use with good training dataset._

---
### References:
- [OpenCV Python FreeCodeCamp](https://www.youtube.com/watch?v=oXlwWbU8l2o)
- [Face detection with Dlib](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/)
- [Face Recognition with OpenCV](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
- [Face Recognition with Dlib](https://dontrepeatyourself.org/post/face-recognition-with-python-dlib-and-deep-learning/)
- [Face Recognition with OpenCV](https://pyimagesearch.com/2018/09/24/opencv-face-recognition/)
- [Face Detection with OpenCV](https://www.datacamp.com/tutorial/face-detection-python-opencv)
- [Histogram equalization](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
