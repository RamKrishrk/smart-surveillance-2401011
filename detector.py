"""
detector.py
------------
Handles object detection using a pre-trained YOLOv5 ONNX model,
draws bounding boxes, and sends alert emails when a person is detected.
"""

import cv2
import numpy as np
import os
import time
import smtplib
# Email utilities
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# ================= CONFIG =================
MODEL_PATH = "yolov5s.onnx" # Pre-trained YOLO model (ONNX format)
CLASS_NAMES = "coco.names"  # COCO dataset class labels
CONF_THRESHOLD = 0.4        # Minimum confidence for detection
NMS_THRESHOLD = 0.45        # Threshold for Non-Max Suppression
ALERT_CLASS = "person"      # Class to trigger alert
ALERT_MIN_CONF = 0.6        # Minimum confidence for alert
ALERT_COOLDOWN = 20         # Cooldown time between alerts (seconds)

# ======= SMTP Email Settings =======
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "a@gmail.com"
EMAIL_PASSWORD = "asc"  # app password
EMAIL_RECEIVER = "b@gmail.com"

# Directory for saving detection images
SAVE_DIR = os.path.join("static", "detections")
os.makedirs(SAVE_DIR, exist_ok=True)

# Load COCO class names into list
with open(CLASS_NAMES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

print(f"✅ Loaded {len(classes)} classes")

# Load YOLO ONNX model
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# Enable GPU acceleration if CUDA is available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("🚀 Using CUDA acceleration")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
else:
    print("⚙️ Using CPU inference")

# Track last alert time to prevent alert spamming
last_alert_time = 0


def send_email_alert(image, class_name, confidence):
    """
    Sends an email alert with the detected image attached.
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"🚨 ALERT: {class_name.upper()} Detected ({confidence*100:.1f}%)"

        body = f"A {class_name} was detected with confidence {confidence*100:.1f}% at {time.ctime()}."
        msg.attach(MIMEText(body, 'plain'))

        # Attach detected image
        _, img_encoded = cv2.imencode('.jpg', image)
        msg.attach(MIMEImage(img_encoded.tobytes(), name="alert.jpg"))

        # Send email using SMTP
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print("✅ Email alert sent")
    except Exception as e:
        print("❌ Email failed:", e)


def detect_objects(frame):
    """ Runs object detection on a single video frame.

    Returns:
        frame: Annotated frame with bounding boxes
        alert_triggered: Boolean flag indicating alert status"""
    global last_alert_time
    H, W = frame.shape[:2]
        
     # Convert frame into blob (model input format)
     # Normalize pixel values
     # Model input size
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    # Forward pass through YOLO model
    preds = net.forward()[0]

    boxes, confidences, class_ids = [], [], []
    # Scaling factors to map detections back to original image size
    x_factor, y_factor = W / 640, H / 640

    for det in preds:
        conf = det[4]  # Objectness confidence
        # Ignore weak detections
        if conf < CONF_THRESHOLD:
            continue

        scores = det[5:]
        class_id = np.argmax(scores)
        # Final confidence = objectness * class probability
        total_conf = conf * scores[class_id]
        if total_conf < CONF_THRESHOLD:
            continue

        # Convert bounding box from normalized to pixel coordinates
        # Extract bounding box
        cx, cy, w, h = det[:4]
        x = int((cx - w / 2) * x_factor)
        y = int((cy - h / 2) * y_factor)
        w = int(w * x_factor)
        h = int(h * y_factor)

        boxes.append([x, y, w, h])
        confidences.append(float(total_conf))
        class_ids.append(class_id)
        
        # Apply Non-Max Suppression to remove overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    alert_triggered = False

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cls = classes[class_ids[i]]
            conf = confidences[i]

                        # Red box for alert class, green for others
            color = (0, 255, 0) if cls != ALERT_CLASS else (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Put class label and confidence
            cv2.putText(frame, f"{cls}: {conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Trigger alert if person detected
            if cls == ALERT_CLASS and conf >= ALERT_MIN_CONF:
                current_time = time.time()
                # Cooldown check to avoid spamming emails
                if current_time - last_alert_time >= ALERT_COOLDOWN:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{cls}_{timestamp}.jpg"
                    save_path = os.path.join(SAVE_DIR, filename)
                    
                     # Save detection image
                    cv2.imwrite(save_path, frame)
                    print(f"💾 Saved detection image to {save_path}")
                    
                    # Send email alert
                    send_email_alert(frame, cls, conf)
                    last_alert_time = current_time
                    alert_triggered = True

    return frame, alert_triggered
