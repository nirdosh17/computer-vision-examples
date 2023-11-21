# Playground for opencv operations with all installed libraries
FROM python:3.9-bullseye

WORKDIR  /opencv

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
# RUN pip install opencv-contrib-python dlib imutils

COPY ./ /opencv
