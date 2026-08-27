#!/bin/bash
# One-time setup for the imx500 branch on Raspberry Pi OS Trixie (64-bit, desktop).
# Run from inside the cs460-poc folder:  bash setup.sh
set -euo pipefail

sudo apt update
sudo apt full-upgrade -y
sudo apt install -y \
    imx500-all \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-munkres \
    python3-imath \
    opencv-data

python3 - <<'PY'
from pathlib import Path

import cv2
import numpy
from picamera2 import Picamera2
from picamera2.devices import IMX500

required_models = [
    Path("/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk"),
    Path("/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk"),
]
missing = [str(path) for path in required_models if not path.is_file()]
if missing:
    raise SystemExit("Missing IMX500 model file(s): " + ", ".join(missing))

print(f"OpenCV {cv2.__version__}; NumPy {numpy.__version__}; Picamera2/IMX500 imports OK")
print("IMX500 model files found - setup complete")
PY

echo
echo "Reboot before first use of the AI Camera: sudo reboot"
