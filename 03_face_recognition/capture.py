import os
import sys
import time

import cv2
from picamera2 import Picamera2

if len(sys.argv) != 2:
    sys.exit("usage: python3 03_face_recognition/capture.py Name")

name = sys.argv[1]
os.makedirs("faces", exist_ok=True)
photo_path = f"faces/{name}.jpg"

cam = Picamera2()
cam.configure(
    cam.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)
cam.start()

# Allow exposure and white balance to settle.
time.sleep(2)

try:
    while True:
        frame = cam.capture_array()
        preview = frame.copy()

        cv2.putText(
            preview,
            "SPACE = capture   q = cancel",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Capture face", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not cv2.imwrite(photo_path, frame):
                sys.exit(f"could not save {photo_path}")
            print(f"saved {photo_path}")
            break

        if key == ord("q"):
            print("capture cancelled")
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()