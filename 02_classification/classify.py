import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500.postprocess import softmax
from mobilenet import MobileNet

W, H = 640, 480
MODEL = "/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk"
FALLBACK_LABELS = MobileNet.LABELS_IMAGENET_1K.splitlines()

# IMX500 must be created before Picamera2 so the network can be loaded on the AI Camera.
imx500 = IMX500(MODEL)
intrinsics = imx500.network_intrinsics
if intrinsics is None:
    raise RuntimeError("IMX500 model intrinsics are missing; reinstall the imx500-all package")
intrinsics.update_with_defaults()
labels = intrinsics.labels or FALLBACK_LABELS

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

results = []
while True:
    with cam.captured_request() as request:
        frame = request.make_array("main")
        outputs = imx500.get_outputs(request.get_metadata())

        if outputs is not None:
            scores = np.squeeze(outputs[0])
            if intrinsics.softmax:
                scores = softmax(scores)

            active_labels = labels[1:] if len(labels) == len(scores) + 1 else labels
            top = np.argpartition(-scores, 3)[:3]
            top = top[np.argsort(-scores[top])]
            results = [
                (active_labels[i] if i < len(active_labels) else str(i), float(scores[i]))
                for i in top
            ]

    for n, (name, score) in enumerate(results):
        cv2.putText(
            frame,
            f"{name}: {score * 100:.0f}%",
            (10, 30 + n * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    cv2.putText(frame, "q = quit", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.imshow("Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()
