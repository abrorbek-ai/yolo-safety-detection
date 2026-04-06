import cv2
import time
from ultralytics import YOLO


MODEL_PATH = "Safety-Helmet-Detection-main/output/best.pt"
CONF = 0.25


model = YOLO(MODEL_PATH)

# track how long a person has no helmet
no_helmet_start = {}
ALERT_DELAY = 3  # seconds

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF)

    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        cls_id = int(box.cls[0]) if box.cls is not None else -1

        label = model.names.get(cls_id, "object").lower()

        # create a stable ID based on approximate position (prevents reset when moving slightly)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        center_id = (cx // 100, cy // 100)

        # show helmet or no-helmet status
        if "helmet" in label:
            if center_id in no_helmet_start:
                del no_helmet_start[center_id]

            color = (0, 255, 0)
            text = f"HELMET {conf:.2f}"

        else:
            if center_id not in no_helmet_start:
                no_helmet_start[center_id] = time.time()

            elapsed = time.time() - no_helmet_start[center_id]

            color = (0, 0, 255)
            text = f"NO HELMET {conf:.2f}"

            # after 3 seconds show blinking SOS alarm with red flashing area
            if elapsed > ALERT_DELAY:
                text = "SOS - NO HELMET"

                if int(time.time() * 2) % 2 == 0:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                    # stronger red alarm light (background ~35% visible)
                    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    cv2.imshow("Safety Pipeline: Detect + Track + Alert", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
