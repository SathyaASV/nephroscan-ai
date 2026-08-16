from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import requests
from flask import Flask, Response, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
AI_SCRIPT = ROOT / "ai" / "ai_server.py"
THERMAL_SCRIPT = ROOT / "thermal_stream.py"
AI_ORIGIN = "http://127.0.0.1:5000"
THERMAL_ORIGIN = "http://127.0.0.1:5001"
GATEWAY_PORT = 8000
CLOUDFLARED_CANDIDATES = [
    ROOT / "cloudflared" / "cloudflared-windows-amd64.exe",
    ROOT / "cloudflared.exe",
]

app = Flask(__name__, static_folder=None)
children: list[subprocess.Popen] = []


def launch_service(script: Path, label: str) -> None:
    if not script.exists():
        raise FileNotFoundError(f"{label} file not found: {script}")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        creationflags=creationflags,
    )
    children.append(process)
    print(f"Started {label}: {script}")


def start_cloudflare() -> None:
    executable = next((path for path in CLOUDFLARED_CANDIDATES if path.exists()), None)
    if executable is None:
        print("Cloudflare executable not found; run the gateway locally at port 8000.")
        return
    process = subprocess.Popen(
        [str(executable), "tunnel", "--url", f"http://127.0.0.1:{GATEWAY_PORT}"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    children.append(process)

    def forward_output() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            print("[cloudflared]", line.rstrip())

    threading.Thread(target=forward_output, daemon=True).start()
    print("Started Cloudflare Quick Tunnel automatically.")


def cleanup() -> None:
    for process in children:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=4)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


atexit.register(cleanup)


def rewrite_frontend(html: str) -> str:
    # Keep the existing frontend logic intact while making all service calls
    # same-origin through this gateway. This avoids localhost URLs on phones
    # and remote reviewer devices.
    replacements = {
        "http://127.0.0.1:5000": "/ai",
        "http://localhost:5000": "/ai",
        "http://127.0.0.1:5001": "/thermal",
        "http://localhost:5001": "/thermal",
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    return html


@app.get("/")
def frontend_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return jsonify({"error": f"Frontend not found: {index_path}"}), 500
    html = index_path.read_text(encoding="utf-8")
    return Response(rewrite_frontend(html), mimetype="text/html")


@app.get("/<path:filename>")
def frontend_asset(filename: str):
    if filename.startswith(("ai/", "thermal/")):
        return jsonify({"error": "Reserved gateway path"}), 404
    return send_from_directory(FRONTEND_DIR, filename)


def _forward_request(origin: str, path: str):
    target = f"{origin}/{path.lstrip('/')}"
    try:
        if request.method in {"POST", "PUT", "PATCH"}:
            files = {}
            for name, storage in request.files.items():
                files[name] = (
                    storage.filename or "upload.bin",
                    storage.stream,
                    storage.mimetype or "application/octet-stream",
                )
            upstream = requests.request(
                request.method,
                target,
                params=request.args,
                data=request.form,
                files=files or None,
                json=request.get_json(silent=True) if not files else None,
                timeout=None,
                stream=True,
            )
        else:
            upstream = requests.request(
                request.method,
                target,
                params=request.args,
                timeout=None,
                stream=True,
            )
    except requests.RequestException as exc:
        return jsonify({"error": f"Upstream service unavailable: {exc}"}), 502

    excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    headers = [
        (key, value)
        for key, value in upstream.headers.items()
        if key.lower() not in excluded
    ]
    return Response(
        upstream.iter_content(chunk_size=64 * 1024),
        status=upstream.status_code,
        headers=headers,
    )


@app.route("/ai/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def ai_proxy(path: str):
    return _forward_request(AI_ORIGIN, path)


@app.route("/thermal/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def thermal_proxy(path: str):
    return _forward_request(THERMAL_ORIGIN, path)


@app.get("/gateway_health")
def gateway_health():
    return jsonify({
        "status": "online",
        "service": "NephroScan unified gateway",
        "frontend": "/",
        "ai": "/ai/health",
        "thermal": "/thermal/thermal_status",
        "single_origin": True,
    })


def main() -> None:
    print("Starting NephroScan unified gateway...")
    launch_service(AI_SCRIPT, "AI backend")
    launch_service(THERMAL_SCRIPT, "thermal stream")
    time.sleep(2)
    print(f"NephroScan website: http://127.0.0.1:{GATEWAY_PORT}")
    start_cloudflare()
    print(f"Gateway health: http://127.0.0.1:{GATEWAY_PORT}/gateway_health")
    print("One public tunnel command:")
    print(f"cloudflared tunnel --url http://127.0.0.1:{GATEWAY_PORT}")
    app.run(host="0.0.0.0", port=GATEWAY_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
