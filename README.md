# Raspberry Pi Computer Vision Projects

Computer vision projects running on a Raspberry Pi 4 with a Raspberry Pi Camera Module.
All projects use OpenCV and pre-trained models. No GPU and no pip packages are required.

## Projects

1. Face detection (01_face_detection/face.py)
   Detects faces in the live camera feed with a Haar cascade and draws a box around each one.

2. Image classification (02_classification/classify.py)
   Uses MobileNetV2 to name the object in front of the camera. Shows the top 3 guesses with percentages.

3. Face recognition (03_face_recognition/enroll.py and recognize.py)
   Enroll a person from a photo, then the live feed labels that person by name. Unknown faces are marked in red.

4. Object detection (04_object_detection/detect.py)
   Uses NanoDet to find multiple objects at once (80 COCO classes) and draws a labelled box around each.

## Hardware and software

- Raspberry Pi 4 (tested with 8 GB RAM)
- Raspberry Pi Camera Module (tested with v1, also works with v2, v3 and HQ)
- Raspberry Pi OS Trixie, 64-bit, desktop edition
- OpenCV 4.10, NumPy 2, Picamera2 (installed by setup.sh)

## Setup on a fresh Pi

    git clone <repository url>
    cd cv-course
    bash setup.sh

## How to run

Run every command from inside the cv-course folder, because the scripts load the models with relative paths.
If you are connected over SSH, put DISPLAY=:0 in front of the command so the window opens on the Pi desktop.
Press q in the window to quit.

    python3 01_face_detection/face.py
    python3 02_classification/classify.py
    python3 03_face_recognition/enroll.py Name photo.jpg
    python3 03_face_recognition/recognize.py
    python3 04_object_detection/detect.py

## Folder layout

- 01_face_detection to 04_object_detection: one folder per project
- models: the pre-trained networks
- faces: face data created by enroll.py (ignored by git, personal data)
- setup.sh: installs the required packages

## Models

All models come from the OpenCV Model Zoo (https://github.com/opencv/opencv_zoo), Apache 2.0 license.

- mobilenetv2.onnx: MobileNetV2 trained on ImageNet (1000 classes), used for classification
- yunet.onnx: YuNet face detector
- sface.onnx: SFace face recognition model
- nanodet.onnx: NanoDet-Plus object detector trained on COCO (80 classes)

mobilenet.py and nanodet.py are helper files from the same model zoo.
The Haar cascade used in project 1 comes from the opencv-data package.
