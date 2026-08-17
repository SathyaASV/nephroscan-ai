import cv2
import numpy as np
import time
import math
from flask import Flask, Response, jsonify

app = Flask(__name__)

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

camera = None
running = False
start_time = None
latest_status = {
    "status": "offline",
    "emulated_index": "CALIBRATING...",
    "system_status": "CAMERA NOT STARTED",
    "tracking_id": None,
    "proxy_mode": "RGB SOFTWARE PROXY",
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
    inverted_gray = cv2.bitwise_not(blurred)
    thermal_sim = cv2.applyColorMap(inverted_gray, cv2.COLORMAP_INFERNO)

    elapsed = time.time() - (start_time or time.time())
    if elapsed < 2.5:
        system_status = "INITIALIZING SOFTWARE PROXY..."
        status_color = (0, 140, 255)
        ai_active = False
    else:
        system_status = "RGB SOFTWARE PROXY: ACTIVE"
        status_color = (0, 255, 0)
        ai_active = True

    display_index = "CALIBRATING..."
    if ai_active:
        cx, cy = int(w / 2), int(h / 2)
        heartbeat_pulse = math.sin(time.time() * 2.5) * 0.12

        roi_half = 30
        y1 = max(0, cy - roi_half)
        y2 = min(h, cy + roi_half)
        x1 = max(0, cx - roi_half)
        x2 = min(w, cx + roi_half)
        roi_region = gray_base[y1:y2, x1:x2]
        roi_brightness = float(np.mean(roi_region)) if roi_region.size > 0 else 128.0

        simulated_celsius = 36.4 + (roi_brightness / 255.0) * 0.7 + heartbeat_pulse
        display_index = f"{simulated_celsius:.1f} C"
        hud_color = (0, 255, 0)
        if roi_brightness > 210:
            simulated_celsius = 38.6 + heartbeat_pulse
            display_index = f"{simulated_celsius:.1f} C - HIGH FEVER ALARM"
            hud_color = (0, 0, 255)

        for target_view in [rgb_frame, thermal_sim]:
            cv2.drawMarker(target_view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            cv2.rectangle(target_view, (cx - 45, cy - 45), (cx + 45, cy + 45), hud_color, 1)
            cv2.putText(target_view, display_index, (cx - 65, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(target_view, display_index, (cx - 65, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)

    combined_display = np.hstack((rgb_frame, thermal_sim))

    cv2.putText(combined_display, "RGB SOFTWARE PROXY: ACTIVE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_display, "THERMAL PROXY: SOFTWARE EMULATION", (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.putText(combined_display, "NOT AN INFRARED MEASUREMENT", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(combined_display, "NOT AN INFRARED MEASUREMENT", (w + 15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)

    cv2.rectangle(combined_display, (0, h - 35), (w * 2, h), (15, 15, 15), -1)
    cv2.putText(combined_display, f"STATUS: {system_status}", (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
    cv2.putText(combined_display, "EMULATED INDEX — SOFTWARE PROXY", (w * 2 - 380, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    latest_status = {
        "status": "online",
        "emulated_index": display_index,
        "system_status": system_status,
        "tracking_id": "THERMAL-PROXY-HOST-001",
        "proxy_mode": "RGB SOFTWARE PROXY",
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
    return """<!DOCTYPE html>
<html><head><title>NephroScan Thermal Host</title>
<style>body{margin:0;background:#0b1222;color:#eef6ff;font-family:system-ui;display:flex;flex-direction:column;align-items:center;padding:20px}
h2{color:#00c8ff;margin-bottom:8px}.sub{color:#7794ae;margin-bottom:16px;font-size:13px}
img{max-width:90%;border:1px solid #1a3050;border-radius:8px}
.status{margin-top:12px;padding:8px 16px;background:#0d1b2d;border:1px solid #1a3050;border-radius:6px;font-size:13px;color:#9fb4c8}
.disclaimer{margin-top:12px;color:#5b6d7d;font-size:12px;text-align:center;max-width:700px}
</style></head><body>
<h2>NephroScan Thermal Host Proxy</h2>
<div class="sub">RGB SOFTWARE PROXY — NOT AN INFRARED MEASUREMENT</div>
<img src="/thermal_stream" />
<div class="status" id="status">Loading...</div>
<div class="disclaimer">This is a SOFTWARE THERMAL PROXY using a standard RGB webcam. It does not measure real temperature. The emulated index is derived from grayscale brightness values mapped through an Inferno colormap. For research and demonstration purposes only.</div>
<script>setInterval(async()=>{try{const r=await fetch('/thermal_status');const d=await r.json();
document.getElementById('status').innerHTML='Status: '+d.status+' | Index: '+d.emulated_index+' | Mode: '+d.proxy_mode+' | ID: '+d.tracking_id;}catch(e){}},1000);
</script></body></html>"""


if __name__ == "__main__":
    print("NephroScan Thermal Host Proxy")
    print("Stream: http://127.0.0.1:5001/thermal_stream")
    print("Status: http://127.0.0.1:5001/thermal_status")
    print("Press 'q' in the OpenCV window to stop.")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
