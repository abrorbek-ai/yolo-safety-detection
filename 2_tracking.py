import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "Safety-Helmet-Detection-main/output/best.pt"
CONF = 0.25

HELMET_CLASS = 0
HEAD_CLASS = 1
PERSON_CLASS = 2

CLASS_NAMES = {
    HELMET_CLASS: "helmet",
    HEAD_CLASS: "head",
    PERSON_CLASS: "person",
}

CLASS_COLORS = {
    HELMET_CLASS: (0, 255, 0),
    HEAD_CLASS: (255, 0, 0),
    PERSON_CLASS: (0, 165, 255),
}

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        conf=CONF,
        persist=True,
        tracker="/Users/abran/miniconda3/envs/yolo1/lib/python3.10/site-packages/ultralytics/cfg/trackers/bytetrack.yaml",
        verbose=False,
    )

    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            xyxy = tuple(map(int, box.xyxy[0]))
            track_id = int(box.id[0]) if box.id is not None else None
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            cls_id = int(box.cls[0]) if box.cls is not None else -1

            # Accept helmet, head, or person but display them all as Worker
            if cls_id not in [HELMET_CLASS, HEAD_CLASS, PERSON_CLASS]:
                continue

            # Show the same worker label for person, head, and helmet
            if track_id is not None:
                label = f"Worker: ({track_id})"
            else:
                label = "Worker"

            color = (0, 255, 255)  # same color for all workers

            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(
                frame,
                label,
                (xyxy[0], max(25, xyxy[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    cv2.imshow("Camera Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
