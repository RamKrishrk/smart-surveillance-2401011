import cv2
import numpy as np
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# ================= CONFIG =================
MODEL_PATH = "yolov5s.onnx"
CLASS_NAMES = "coco.names"
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45
ALERT_CLASS = "person"
ALERT_MIN_CONF = 0.6
ALERT_COOLDOWN = 20

# ======= SMTP Email Settings =======
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "ramakrishnan.learnings14@gmail.com"
EMAIL_PASSWORD = "esdx xeyg rppz ooak"  # app password
EMAIL_RECEIVER = "im.ramakrishnan.l@gmail.com"

# Directory for saving detection images
SAVE_DIR = os.path.join("static", "detections")
os.makedirs(SAVE_DIR, exist_ok=True)

with open(CLASS_NAMES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

print(f"✅ Loaded {len(classes)} classes")

# Load model
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("🚀 Using CUDA acceleration")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
else:
    print("⚙️ Using CPU inference")

last_alert_time = 0


def send_email_alert(image, class_name, confidence):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"🚨 ALERT: {class_name.upper()} Detected ({confidence*100:.1f}%)"

        body = f"A {class_name} was detected with confidence {confidence*100:.1f}% at {time.ctime()}."
        msg.attach(MIMEText(body, 'plain'))

        _, img_encoded = cv2.imencode('.jpg', image)
        msg.attach(MIMEImage(img_encoded.tobytes(), name="alert.jpg"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print("✅ Email alert sent")
    except Exception as e:
        print("❌ Email failed:", e)


def detect_objects(frame):
    """Run YOLO detection and return annotated frame."""
    global last_alert_time
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()[0]

    boxes, confidences, class_ids = [], [], []
    x_factor, y_factor = W / 640, H / 640

    for det in preds:
        conf = det[4]
        if conf < CONF_THRESHOLD:
            continue

        scores = det[5:]
        class_id = np.argmax(scores)
        total_conf = conf * scores[class_id]
        if total_conf < CONF_THRESHOLD:
            continue

        cx, cy, w, h = det[:4]
        x = int((cx - w / 2) * x_factor)
        y = int((cy - h / 2) * y_factor)
        w = int(w * x_factor)
        h = int(h * y_factor)

        boxes.append([x, y, w, h])
        confidences.append(float(total_conf))
        class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    alert_triggered = False

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cls = classes[class_ids[i]]
            conf = confidences[i]
            color = (0, 255, 0) if cls != ALERT_CLASS else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{cls}: {conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # alert email logic
            if cls == ALERT_CLASS and conf >= ALERT_MIN_CONF:
                current_time = time.time()
                if current_time - last_alert_time >= ALERT_COOLDOWN:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{cls}_{timestamp}.jpg"
                    save_path = os.path.join(SAVE_DIR, filename)
                    cv2.imwrite(save_path, frame)
                    print(f"💾 Saved detection image to {save_path}")
                    
                    send_email_alert(frame, cls, conf)
                    last_alert_time = current_time
                    alert_triggered = True

    return frame, alert_triggered
