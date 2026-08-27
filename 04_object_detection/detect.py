import cv2
from picamera2 import Picamera2
from picamera2.devices import IMX500

W, H = 640, 480
MODEL = "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk"
THRESHOLD = 0.35

CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

# The post-processed NanoDet network runs on the IMX500 accelerator in the AI Camera.
imx500 = IMX500(MODEL)
intrinsics = imx500.network_intrinsics
if intrinsics is None:
    raise RuntimeError("IMX500 model intrinsics are missing; reinstall the imx500-all package")
intrinsics.update_with_defaults()
labels = intrinsics.labels or list(CLASSES)

cam = Picamera2(imx500.camera_num)
config = cam.create_preview_configuration(
    main={"format": "RGB888", "size": (W, H)},
    controls={"FrameRate": intrinsics.inference_rate},
    buffer_count=12,
)
cam.configure(config)
imx500.show_network_fw_progress_bar()
cam.start()
if intrinsics.preserve_aspect_ratio:
    imx500.set_auto_aspect_ratio()

last_detections = []
while True:
    with cam.captured_request() as request:
        frame = request.make_array("main")
        metadata = request.get_metadata()
        outputs = imx500.get_outputs(metadata, add_batch=True)

        if outputs is not None:
            boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
            input_w, input_h = imx500.get_input_size()

            if intrinsics.bbox_normalization:
                boxes = boxes / input_h
            if intrinsics.bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]

            last_detections = []
            for box, score, category in zip(boxes, scores, classes):
                if score <= THRESHOLD:
                    continue
                x, y, w, h = imx500.convert_inference_coords(box, metadata, cam)
                last_detections.append((x, y, w, h, int(category), float(score)))

    for x, y, w, h, category, score in last_detections:
        name = labels[category] if category < len(labels) else str(category)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name} {score:.2f}",
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(frame, "q = quit", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.imshow("Object detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()
