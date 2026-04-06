import cv2
import math
import time
import os
from ultralytics import YOLO

# models
POSE_MODEL = "yolov8n-pose.pt"

# thresholds
FALL_ANGLE_THRESHOLD = 45
FALL_CONFIRM_SECONDS = 3

pose_model = YOLO(POSE_MODEL)

cap = cv2.VideoCapture(0)

fall_start_time = None
alarm_played = False


def body_angle(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    return angle


while True:

    ret, frame = cap.read()
    if not ret:
        break

    pose_results = pose_model(frame, conf=0.3)
    frame = pose_results[0].plot()

    fall_detected = False

    if pose_results[0].keypoints is not None:

        for person in pose_results[0].keypoints.xy:

            px = person[:,0]
            py = person[:,1]
            x1, x2 = int(px.min()), int(px.max())
            y1, y2 = int(py.min()), int(py.max())

            left_shoulder = person[5]
            right_shoulder = person[6]
            left_hip = person[11]
            right_hip = person[12]

            shoulder = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2,
            )

            hip = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2,
            )

            angle = body_angle(shoulder, hip)

            if angle < FALL_ANGLE_THRESHOLD:
                fall_detected = True

    # fall timer
    if fall_detected:

        if fall_start_time is None:
            fall_start_time = time.time()

        elapsed = time.time() - fall_start_time

        if elapsed > FALL_CONFIRM_SECONDS:

            # blinking SOS around fallen person
            if int(time.time() * 2) % 2 == 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,255), -1)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)

            cv2.putText(
                frame,
                "SOS - FALL DETECTED",
                (x1, max(40, y1 - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 255),
                4,
            )

            # play alarm once
            if not alarm_played:
                os.system("afplay /System/Library/Sounds/Sosumi.aiff &")
                alarm_played = True

    else:
        fall_start_time = None
        alarm_played = False

    cv2.imshow("Construction Safety AI (Fall + Helmet)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
