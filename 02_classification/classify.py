import cv2
import numpy as np
from picamera2 import Picamera2
from mobilenet import MobileNet   # used only for its list of the 1000 ImageNet names

labels = MobileNet.LABELS_IMAGENET_1K.splitlines()
net = cv2.dnn.readNet("models/mobilenetv2.onnx")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def classify(square_bgr):
    rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[np.newaxis]   # 1 x 3 x 224 x 224
    net.setInput(blob)
    scores = net.forward()[0]
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()                                          # softmax -> percentages
    top = probs.argsort()[::-1][:3]
    return [(labels[i], float(probs[i])) for i in top]

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
cam.start()

while True:
    frame = cam.capture_array()
    h, w = frame.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    square = frame[y0:y0 + s, x0:x0 + s]      # the network only sees this centre square
    results = classify(square)

    cv2.rectangle(frame, (x0, y0), (x0 + s - 1, y0 + s - 1), (255, 200, 0), 1)
    for n, (name, p) in enumerate(results):
        cv2.putText(frame, f"{name}: {p * 100:.0f}%", (10, 30 + n * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "q = quit", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.imshow("Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()
