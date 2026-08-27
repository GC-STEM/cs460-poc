import cv2
import numpy as np
from picamera2 import Picamera2
from nanodet import NanoDet      # helper from OpenCV's model zoo: runs the net and decodes the boxes

W, H = 640, 480
SIZE = 416                        # the network's input size (square)

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
           'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
           'toothbrush')

model = NanoDet(modelPath="models/nanodet.onnx", prob_threshold=0.35, iou_threshold=0.6,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV, target_id=cv2.dnn.DNN_TARGET_CPU)

# Our frame is 640x480 but the network wants 416x416, so we shrink it to 416 wide
# and pad the top and bottom with black bars (a "letterbox"). These numbers undo that later.
scale = SIZE / W                  # 0.65
new_h = int(H * scale)            # 312
top = (SIZE - new_h) // 2         # 52 px of black above and below

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (W, H)}))
cam.start()
timer = cv2.TickMeter()

while True:
    frame = cam.capture_array()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (SIZE, new_h), interpolation=cv2.INTER_AREA)
    boxed = cv2.copyMakeBorder(small, top, SIZE - new_h - top, 0, 0, cv2.BORDER_CONSTANT, value=0)

    timer.start()
    preds = model.infer(boxed)    # each row: x1, y1, x2, y2, confidence, class id
    timer.stop()

    for p in preds:
        x1 = int(p[0] / scale)
        y1 = int((p[1] - top) / scale)
        x2 = int(p[2] / scale)
        y2 = int((p[3] - top) / scale)
        conf, cls = p[4], int(p[5])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{CLASSES[cls]} {conf:.2f}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"{timer.getFPS():.1f} fps   q = quit", (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.imshow("Object detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()
