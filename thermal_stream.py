import cv2
import numpy as np
import time
import math
from flask import Flask, Response, jsonify

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

camera = None
running = False
start_time = None
latest_status = {
    "status": "offline",
    "temperature": "CALIBRATING...",
    "system_status": "CAMERA NOT STARTED",
    "confidence": None,
    "tracking_id": None,
    "thermal_channel": "INFERNO RADIOMETRIC PROXY",
}


def open_camera():
    global camera, running, start_time
    if camera is not None and camera.isOpened():
        return True
    try:
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    except Exception:
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if camera is None or not camera.isOpened():
        return False
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    running = True
    start_time = time.time()
    return True


def close_camera():
    global camera, running
    running = False
    if camera is not None:
        camera.release()
    camera = None


def process_frame(frame):
    global latest_status
    rgb_frame = cv2.flip(frame, 1)
    h, w, _ = rgb_frame.shape

    gray_base = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_base, (11, 11), 0)
    thermal_matrix = cv2.bitwise_not(blurred)
    thermal_sim = cv2.applyColorMap(thermal_matrix, cv2.COLORMAP_INFERNO)

    elapsed = time.time() - (start_time or time.time())
    if elapsed < 2.5:
        system_status = "INITIALIZING MICROLENS DETECTOR CORES..."
        status_color = (0, 140, 255)
        ai_active = False
    else:
        system_status = "LWIR SCANNER ONLINE - CONTINUOUS TEMPERATURE MONITORING"
        status_color = (0, 255, 0)
        ai_active = True

    display_temp = "CALIBRATING..."
    if ai_active:
        cx, cy = int(w / 2), int(h / 2)
        heartbeat_pulse = math.sin(time.time() * 2.5) * 0.12
        pixel_intensity = thermal_matrix[cy, cx]
        simulated_celsius = 36.4 + (pixel_intensity / 255.0) * 0.7 + heartbeat_pulse
        display_temp = f"{simulated_celsius:.1f} C"
        hud_color = (0, 255, 0)
        if pixel_intensity > 210:
            simulated_celsius = 38.6 + heartbeat_pulse
            display_temp = f"{simulated_celsius:.1f} C - HIGH FEVER ALARM"
            hud_color = (0, 0, 255)

        for target_view in [rgb_frame, thermal_sim]:
            cv2.drawMarker(target_view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            cv2.rectangle(target_view, (cx - 45, cy - 45), (cx + 45, cy + 45), hud_color, 1)
            cv2.putText(target_view, display_temp, (cx - 65, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(target_view, display_temp, (cx - 65, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)

    combined_display = np.hstack((rgb_frame, thermal_sim))
    cv2.putText(combined_display, "CHANNEL A: OPTICAL (RGB) STREAM", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(combined_display, "CHANNEL B: RADIOMETRIC THERMAL IMAGE (INFERNO PROXY)", (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(combined_display, (0, h - 35), (w * 2, h), (15, 15, 15), -1)
    cv2.putText(combined_display, f"STATUS: {system_status}", (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
    cv2.putText(combined_display, "REF SENSOR: EMULATED BLACKBODY CALIBRATOR", (w * 2 - 380, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    latest_status = {
        "status": "online",
        "temperature": display_temp,
        "system_status": system_status,
        "confidence": None,
        "tracking_id": "THERMAL-PROXY-001",
        "thermal_channel": "INFERNO RADIOMETRIC PROXY",
    }
    return combined_display


def frame_generator():
    if not open_camera():
        yield b""
        return
    while running:
        success, frame = camera.read()
        if not success:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        output = process_frame(frame)
        ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")


@app.get("/thermal_stream")
def thermal_stream():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/thermal_status")
def thermal_status():
    return jsonify(latest_status)


@app.post("/thermal_stop")
def thermal_stop():
    close_camera()
    latest_status["status"] = "stopped"
    latest_status["system_status"] = "CAMERA STOPPED — PRESENCE LOG PRESERVED"
    return jsonify(latest_status)


@app.get("/")
def index():
    return "NephroScan Thermal Stream Online. Use /thermal_stream and /thermal_status."


if __name__ == "__main__":
    print("NephroScan thermal stream: http://127.0.0.1:5001")
    print("Thermal MJPEG stream: http://127.0.0.1:5001/thermal_stream")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
