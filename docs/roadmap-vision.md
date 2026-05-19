# Roadmap: Hailo-8 Vision Pipeline

> **Status: Not yet implemented.** The Hailo-8 driver is active (`/dev/hailo0` confirmed at `0001:01:00.0`) and TAPPAS is installed. This document describes the planned implementation.

## Architecture Goal

```
Camera (Blink / RTSP / USB webcam)
    │
    ▼
GStreamer + Hailo-8 TAPPAS pipeline
    │  26 TOPS, YOLOv8s, ~30fps
    ▼
Detection events (JSON: class, confidence, bounding box)
    │
    ▼
Vision bridge service (Python daemon)
    │  formats: "Camera sees: person (87%), car (72%)"
    ▼
Agent Hub — POST /vision/update
    │
    ├── Injected into LLM context for chat queries
    └── llava-phi3 for visual question answering on captured frames
```

## Camera Integration Options

### Option A: Blink Camera (Snapshot Polling)

Blink cameras on local network can be polled for snapshots. For local-only operation without Blink's cloud:

```python
import requests, time, io
from PIL import Image

BLINK_API = "http://<blink-sync-module-ip>/api/v1"

def get_snapshot(network_id: int, camera_id: int, token: str) -> bytes:
    url = f"{BLINK_API}/accounts/<acct>/networks/{network_id}/cameras/{camera_id}/thumbnail"
    r = requests.get(url, headers={"token-auth": token}, timeout=10)
    r.raise_for_status()
    return r.content

def poll_loop(interval=5):
    while True:
        frame = get_snapshot(...)
        yield frame
        time.sleep(interval)
```

> **Note:** Blink camera RTSP availability depends on the sync module generation. Check if your setup exposes an RTSP stream before defaulting to snapshot polling.

### Option B: RTSP Stream

If your camera or Blink sync module exposes RTSP:

```bash
# Test RTSP stream availability
gst-launch-1.0 rtspsrc location=rtsp://<ip>:<port>/stream ! decodebin ! autovideosink
```

### Option C: USB Webcam (Lowest Latency)

Direct V4L2 capture for USB cameras — no network overhead:

```bash
ls /dev/video*  # list available cameras
v4l2-ctl --device /dev/video0 --list-formats-ext
```

## GStreamer TAPPAS Pipeline

Full detection pipeline outputting JSON detections:

```bash
gst-launch-1.0 \
  v4l2src device=/dev/video0 ! \
  videoconvert ! video/x-raw,width=640,height=480,framerate=15/1,format=RGB ! \
  hailonet hef-path=/home/merry/models/yolov8s_h8.hef batch-size=1 ! \
  hailofilter \
    so-path=/usr/lib/hailo-post-processes/libyolo_hailortpp.so \
    function-name=yolov8 \
    config-path=/usr/share/hailo-models/yolov8/yolov8s.json ! \
  hailostreamrouter name=router \
  router.src_0 ! hailoexportfile location=/tmp/hailo_detections.jsonl ! fakesink \
  router.src_1 ! hailooverlay ! videoconvert ! autovideosink
```

## Vision Bridge Service

Python daemon that reads detection events and formats them for the LLM:

```python
# /home/merry/vision-bridge/bridge.py
import json, asyncio
from pathlib import Path
import httpx

DETECTION_FILE = Path("/tmp/hailo_detections.jsonl")
AGENT_HUB_URL = "http://localhost:8000"

COCO_LABELS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog",
    # ... full COCO 80 classes
}

def format_detections(detections: list[dict], threshold=0.45) -> str:
    items = []
    for d in detections:
        label = COCO_LABELS.get(d.get("class_id", -1), "unknown")
        conf = d.get("confidence", 0)
        if conf >= threshold:
            items.append(f"{label} ({int(conf*100)}%)")
    if not items:
        return "Camera: no objects detected above threshold."
    return f"Camera detects: {', '.join(items)}"

async def update_context(context: str):
    async with httpx.AsyncClient(timeout=5) as client:
        await client.post(f"{AGENT_HUB_URL}/vision/update",
                         json={"context": context})

async def main():
    last_pos = 0
    while True:
        if DETECTION_FILE.exists():
            with open(DETECTION_FILE) as f:
                f.seek(last_pos)
                new_lines = f.readlines()
                last_pos = f.tell()
            for line in new_lines:
                try:
                    data = json.loads(line.strip())
                    ctx = format_detections(data.get("detections", []))
                    await update_context(ctx)
                except (json.JSONDecodeError, KeyError):
                    pass
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## llava-phi3 Visual Question Answering

For "describe what the camera sees" or "is anyone at the front door?":

```python
import base64, httpx

async def ask_about_frame(image_path: str, question: str) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": "llava-phi3",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }],
                "max_tokens": 200
            }
        )
    return resp.json()["choices"][0]["message"]["content"]
```

## Implementation Order

1. Download `yolov8s_h8.hef` from Hailo Model Zoo
2. Determine camera source — test RTSP, then fallback to USB or snapshot polling
3. Test GStreamer pipeline on a static JPEG first, then live feed
4. Write and test vision bridge service (`/home/merry/vision-bridge/bridge.py`)
5. Add `POST /vision/update` endpoint to Agent Hub (`main.py`)
6. Inject latest vision context into LLM system prompt when query is vision-related
7. Create `hailo-vision.service` systemd unit
8. End-to-end test: ask "what do you see?" → verify camera context in response

## Planned Systemd Service

```ini
[Unit]
Description=Hailo-8 Vision Bridge
After=network.target

[Service]
Type=simple
User=merry
WorkingDirectory=/home/merry/vision-bridge
Environment="VIRTUAL_ENV=/home/merry/venv"
Environment="PATH=/home/merry/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/merry/venv/bin/python3 bridge.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
