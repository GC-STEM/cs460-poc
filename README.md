# Raspberry Pi Computer Vision Projects — AI Camera (IMX500)

Computer vision projects for a Raspberry Pi with the Raspberry Pi AI Camera (Sony IMX500).
The classification and object-detection projects run their neural-network inference on the AI Camera's IMX500 accelerator. Face detection and face recognition still use OpenCV on the Raspberry Pi CPU, with the AI Camera providing the live image stream. No GPU or pip packages are required.

## Projects

1. Face detection (`01_face_detection/face.py`)
   Detects faces in the live AI Camera feed with an OpenCV Haar cascade and draws a box around each one. Detection runs on the Raspberry Pi CPU.

2. Image classification (`02_classification/classify.py`)
   Uses the IMX500 MobileNetV2 model to classify the live camera image. Shows the top 3 guesses with percentages. Inference runs on the AI Camera.

3. Face recognition (`03_face_recognition/enroll.py` and `recognize.py`)
   Enroll a person from a photo, then the live AI Camera feed labels that person by name. Unknown faces are marked in red. YuNet/SFace inference runs on the Raspberry Pi CPU.

4. Object detection (`04_object_detection/detect.py`)
   Uses the IMX500 NanoDet-Plus model to find multiple objects at once (80 COCO classes) and draws a labelled box around each. Inference runs on the AI Camera.

## Hardware and software

- Raspberry Pi AI Camera (Sony IMX500)
- Raspberry Pi OS Trixie, 64-bit, desktop edition
- Picamera2, OpenCV, NumPy, IMX500 firmware, and IMX500 model files installed by `setup.sh`

Raspberry Pi's AI Camera documentation uses Raspberry Pi 4 Model B and Raspberry Pi 5 as its primary reference systems and notes that other Raspberry Pi models with a camera connector can also be used with minor changes. Performance will vary by Pi model, especially for the CPU-based face projects.

## Setup on a fresh Pi

Clone the `imx500` branch, run the setup script, and reboot:

    git clone --branch imx500 --single-branch https://github.com/GC-STEM/cs460-poc.git
    cd cs460-poc
    bash setup.sh
    sudo reboot

After reboot, you can confirm that the AI Camera is detected with:

    rpicam-hello --list-cameras

The first time an IMX500 neural-network model is loaded, firmware loading may take noticeably longer than later runs.

## How to run

Run every command from inside the `cs460-poc` folder because the CPU-based face scripts load models with relative paths.
If you are connected over SSH, put `DISPLAY=:0` in front of the command so the window opens on the Pi desktop.
Press `q` in the window to quit.

    python3 01_face_detection/face.py
    python3 02_classification/classify.py
    python3 03_face_recognition/enroll.py Name photo.jpg
    python3 03_face_recognition/recognize.py
    python3 04_object_detection/detect.py

## Folder layout

- `01_face_detection` to `04_object_detection`: one folder per project
- `models`: OpenCV model files used by the CPU-based face-recognition project, plus the original CPU model files retained from the main branch
- `faces`: face data created by `enroll.py` (ignored by git, personal data)
- `setup.sh`: installs the Raspberry Pi AI Camera/IMX500 and OpenCV dependencies

## Models

The IMX500 branch uses Raspberry Pi's packaged IMX500 models from `/usr/share/imx500-models/`:

- `imx500_network_mobilenet_v2.rpk`: MobileNetV2 trained on ImageNet (1000 classes), used for classification
- `imx500_network_nanodet_plus_416x416_pp.rpk`: NanoDet-Plus trained on COCO (80 classes), used for object detection; post-processing runs on the IMX500

The face-recognition project continues to use the OpenCV Model Zoo files in this repository:

- `models/yunet.onnx`: YuNet face detector
- `models/sface.onnx`: SFace face recognition model

The Haar cascade used in project 1 comes from the `opencv-data` package.
The original `models/mobilenetv2.onnx`, `models/nanodet.onnx`, and their helper code remain in this branch for now, but the IMX500 classification and object-detection scripts no longer use those CPU models.
