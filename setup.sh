#!/bin/bash
# One-time setup on a fresh Raspberry Pi OS (Trixie, 64-bit, desktop).
# Run from inside the cv-course folder:  bash setup.sh
set -e
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 python3-numpy opencv-data
python3 -c "import cv2, numpy; from picamera2 import Picamera2; print('OpenCV', cv2.__version__, '- setup complete')"
