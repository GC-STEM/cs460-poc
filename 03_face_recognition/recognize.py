import glob, os
import cv2
import numpy as np
from picamera2 import Picamera2

W, H = 640, 480
THRESHOLD = 0.363     # cosine similarity; OpenCV's recommended cut-off for SFace

detector = cv2.FaceDetectorYN.create("models/yunet.onnx", "", (W, H), 0.8, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create("models/sface.onnx", "")

known = {os.path.basename(p)[:-4]: np.load(p) for p in glob.glob("faces/*.npy")}
print("enrolled people:", list(known))

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (W, H)}))
cam.start()

while True:
    frame = cam.capture_array()
    _, faces = detector.detect(frame)
    for f in (faces if faces is not None else []):
        x, y, w, h = f[:4].astype(int)
        feature = recognizer.feature(recognizer.alignCrop(frame, f))
        best, score = "Unknown", 0.0
        for name, known_feature in known.items():
            s = recognizer.match(feature, known_feature, cv2.FaceRecognizerSF_FR_COSINE)
            if s > score:
                best, score = name, s
        match = score >= THRESHOLD
        color = (0, 255, 0) if match else (0, 0, 255)
        label = f"{best if match else 'Unknown'} {score:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, "q = quit", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.imshow("Face recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()
