from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# YOLO model
# Eslatma: `yolov8n.pt` (COCO) ichida "helmet" klassi yo'q, shuning uchun alohida helmet weights kerak.
# Quyidagi default helmet weights HuggingFace'dan olinadi (internet bo'lishi shart).
DEFAULT_WEIGHTS = "yolov8n.pt"
HELMET_WEIGHTS_URL = "https://huggingface.co/sharathhhhh/safetyHelmet-detection-yolov8/resolve/main/best.pt"

USE_HELMET = True

try:
    model = YOLO(HELMET_WEIGHTS_URL if USE_HELMET else DEFAULT_WEIGHTS)
    helmet_class_ids = [
        i for i, name in getattr(model, "names", {}).items()
        if ("helmet" in str(name).lower()) or ("helm" in str(name).lower())
    ]
    with_helmet_id = None
    without_helmet_id = None
    for i, name in getattr(model, "names", {}).items():
        # Belgilardagi ortiqcha bo'shliqlarni olib tashlaymiz, masalan:
        # " With helmet" yoki "  Without helmet" -> "with helmet"
        n = " ".join(str(name).lower().split())
        if n == "with helmet":
            with_helmet_id = i
        elif n == "without helmet":
            without_helmet_id = i
    if USE_HELMET and not helmet_class_ids:
        print("Helmet weights yuklandi, lekin helmet class nomi topilmadi. Hammasini chizaman.")
    if USE_HELMET:
        print("Helmet classes:", getattr(model, "names", {}))
        print("with_helmet_id:", with_helmet_id, "without_helmet_id:", without_helmet_id)
except Exception as e:
    print("Helmet weights yuklanmadi, default COCO model ishlaydi:", e)
    model = YOLO(DEFAULT_WEIGHTS)
    helmet_class_ids = []
    with_helmet_id = None
    without_helmet_id = None

CONF_WITH_THR = 0.95  # "bor" deb aytish uchun with confidence juda yuqori bo'lishi kerak.
CONF_NO_HELMET_THR = 0.10
STATUS_IF_WITH_HELMET = "bor"  # Siz "vor" deb yozgansiz, "bor" ma'nosida oldim.
STATUS_IF_NO_HELMET = "yoq"

# Kamera
cap = cv2.VideoCapture(0)

def generate():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Faqat helmetga o'xshash klasslarni ajratib predict qilish (bo'lmasa, hammasi).
        results = model(frame, classes=helmet_class_ids) if helmet_class_ids else model(frame)

        annotated = results[0].plot()
        status_text = STATUS_IF_NO_HELMET
        max_with = None
        max_without = None

        # YOLO natijasidan "With helmet / Without helmet" ni aniqlaymiz.
        # Ultralytics: results[0].boxes.cls va results[0].boxes.conf ko'rinishida keladi.
        boxes = results[0].boxes
        if boxes is not None and boxes.cls is not None and with_helmet_id is not None:
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None

            if conf is None:
                # conf bo'lmasa, faqat class bor/yo'qligiga tayanamiz.
                if (cls == with_helmet_id).any():
                    status_text = STATUS_IF_WITH_HELMET
            else:
                # Eng katta confidence bilan qaror qilamiz.
                max_with = conf[cls == with_helmet_id].max() if (cls == with_helmet_id).any() else None
                if without_helmet_id is not None and (cls == without_helmet_id).any():
                    max_without = conf[cls == without_helmet_id].max()

                # Asosiy qaror: "bor" deyish faqat with confidence thresholddan oshsa.
                if max_with is not None and max_with >= CONF_WITH_THR:
                    # Agar both detect bo'lsa ham, baribir eng ishonchli "with" ni tanlaymiz.
                    if max_without is None or max_with >= max_without:
                        status_text = STATUS_IF_WITH_HELMET

        # Debug uchun confidence ko'rsatamiz.
        with_s = f"{max_with:.2f}" if max_with is not None else "-"
        without_s = f"{max_without:.2f}" if max_without is not None else "-"
        cv2.putText(
            annotated,
            f"with:{with_s} without:{without_s} with_thr:{CONF_WITH_THR} no_thr:{CONF_NO_HELMET_THR}",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Yuqorida katta matn chiqaramiz.
        cv2.putText(
            annotated,
            f"DUBULGA: {status_text}",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if status_text == STATUS_IF_WITH_HELMET else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        ret, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)