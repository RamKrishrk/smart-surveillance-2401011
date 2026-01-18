"""
app.py
-------
Flask application to capture live video, run object detection,
and stream the processed video to a web browser in real time.
"""

# Import Flask for web server, Response for video streaming
from flask import Flask, render_template, Response
# OpenCV for camera access and image processing
import cv2
# Custom object detection function
from detector import detect_objects  # Import detection logic

# Initialize Flask application
app = Flask(__name__)

# Initialize camera (0 = default webcam, can be replaced with RTSP URL)
camera = cv2.VideoCapture(0)  # or RTSP URL

def generate_frames():
    """
    Generator function that:
    - Continuously reads frames from camera
    - Runs object detection
    - Encodes frames as JPEG
    - Streams frames to browser
    """

    # Check if camera opened successfully
    if not camera.isOpened():
        print("Camera not opened")
        return

    while True:
        # Read frame from camera
        success, frame = camera.read()

        # If frame capture fails, stop streaming
        if not success:
            break

        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)

        # Run YOLO object detection
        frame, _ = detect_objects(frame)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame for HTTP streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# Root endpoint → loads HTML page
@app.route('/')
def index():
    return render_template('index.html')
# Video feed endpoint → streams live video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# Application entry point
if __name__ == '__main__':
    print("Flask server running at http://127.0.0.1:5000")
        # Run Flask server on all interfaces for Docker compatibility
    app.run(host='0.0.0.0', port=5000, debug=False)
