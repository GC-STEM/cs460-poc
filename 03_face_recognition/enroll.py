import sys, os
import cv2
import numpy as np

if len(sys.argv) != 3:
    sys.exit("usage: python3 enroll.py Name photo.jpg")
name, photo = sys.argv[1], sys.argv[2]

img = cv2.imread(photo)
if img is None:
    sys.exit(f"could not read {photo}")

# shrink very large photos; the detector likes images around 1000 px or less
h, w = img.shape[:2]
scale = 1000 / max(h, w)
if scale < 1:
    img = cv2.resize(img, None, fx=scale, fy=scale)
    h, w = img.shape[:2]

detector = cv2.FaceDetectorYN.create("models/yunet.onnx", "", (w, h), 0.8, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create("models/sface.onnx", "")

_, faces = detector.detect(img)
if faces is None:
    sys.exit("no face found in the photo; try a clearer, front-facing one")

face = max(faces, key=lambda f: f[2] * f[3])          # the largest face in the photo
feature = recognizer.feature(recognizer.alignCrop(img, face))

os.makedirs("faces", exist_ok=True)
np.save(f"faces/{name}.npy", feature)
print(f"enrolled '{name}': {len(faces)} face(s) found, used the {int(face[2])}x{int(face[3])} px one")
