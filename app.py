from flask import Flask, render_template, Response
import cv2
from detector import detect_objects

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # or RTSP URL

def generate_frames():
    if not camera.isOpened():
        print("❌ Camera not opened")
        return
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        # Run detection here
        frame, _ = detect_objects(frame)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 Flask server running at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
