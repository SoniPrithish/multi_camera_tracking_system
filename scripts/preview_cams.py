#!/usr/bin/env python3
"""Quick preview server: shows sample frames from all 6 cameras in a browser."""
import cv2, base64, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    html = "<html><head><title>Camera Preview</title>"
    html += "<style>body{background:#111;color:#fff;font-family:sans-serif;text-align:center;}"
    html += "img{margin:8px;border:2px solid #444;border-radius:6px;}"
    html += "h1{color:#0f0;} .grid{display:flex;flex-wrap:wrap;justify-content:center;}</style></head>"
    html += "<body><h1>6-Camera Preview (frame 300)</h1><div class='grid'>"

    for i in range(1, 7):
        path = f"data/samples/cam{i}.mp4"
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ok, frame = cap.read()
        cap.release()
        if ok:
            small = cv2.resize(frame, (480, 270))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf).decode()
            html += f"<div><h3>CAM {i}</h3><img src='data:image/jpeg;base64,{b64}' width='480'/></div>"
        else:
            html += f"<div><h3>CAM {i} - FAILED</h3></div>"

    html += "</div></body></html>"
    return html

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    uvicorn.run(app, host="0.0.0.0", port=8888)
