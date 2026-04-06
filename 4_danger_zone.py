import cv2
import numpy as np

# simple motion detector
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# store polygon points
zone_points = []
drawing_done = False

def mouse_callback(event, x, y, flags, param):
    global zone_points, drawing_done

    if drawing_done:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append((x, y))


cap = cv2.VideoCapture(1)

cv2.namedWindow("Danger Zone Setup")
cv2.setMouseCallback("Danger Zone Setup", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    display = frame.copy()

    motion_mask = fgbg.apply(frame)
    _, motion_mask = cv2.threshold(motion_mask, 200, 255, cv2.THRESH_BINARY)

    # strengthen small motion regions
    kernel = np.ones((3,3), np.uint8)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

    # draw current polygon
    if len(zone_points) > 0:
        for p in zone_points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)

    if len(zone_points) > 1:
        pts = np.array(zone_points, dtype=np.int32)
        cv2.polylines(display, [pts], False, (0, 0, 255), 2)

    # close and fill polygon if finished
    if drawing_done and len(zone_points) >= 3:
        pts = np.array(zone_points, dtype=np.int32)
        overlay = display.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
        cv2.polylines(display, [pts], True, (0, 0, 255), 2)

        # detect motion inside the zone
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # allow detection of smaller moving objects (small people far from camera)
            if cv2.contourArea(cnt) < 300:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                # SOS alert
                if int(cv2.getTickCount() / cv2.getTickFrequency() * 2) % 2 == 0:
                    alert_overlay = display.copy()
                    cv2.fillPoly(alert_overlay, [pts], (0, 0, 255))
                    cv2.addWeighted(alert_overlay, 0.65, display, 0.35, 0, display)

                cv2.putText(display, "SOS - DANGER ZONE!", (40, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

                cv2.rectangle(display, (x, y), (x+w, y+h), (0,0,255), 3)

    cv2.putText(display, "Left click: add point", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(display, "Press C: close zone | R: reset | Q: quit", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Danger Zone Setup", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        drawing_done = True

    elif key == ord("r"):
        zone_points = []
        drawing_done = False

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
