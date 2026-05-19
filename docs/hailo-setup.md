# Hailo-8 AI Accelerator Setup

The Hailo-8 is a **vision and CNN inference accelerator** — it is NOT a general-purpose AI chip and cannot run LLMs. It excels at running pre-compiled neural network models (YOLO, ResNet, EfficientDet) at 26 TOPS with ~2–4W power draw. In OffGridAI, it handles real-time object detection from camera feeds while the CPU handles language reasoning.

## What the Hailo-8 Can and Cannot Do

| Capability | Hailo-8 | Notes |
|---|---|---|
| YOLOv8 object detection | ✅ 26 TOPS | Sub-millisecond per frame |
| ResNet / EfficientDet classification | ✅ | Wide model zoo support |
| Face detection | ✅ | Lightweight .hef models available |
| Pose estimation | ✅ | Full body keypoints |
| Transformer (LLM) inference | ❌ | Fixed-function dataflow, no support |
| Arbitrary matrix operations | ❌ | Must compile to .hef format first |
| CUDA / ROCm compatibility | ❌ | Proprietary Hailo SDK only |

The Hailo-8 uses a **Dataflow Compiler (DFC)** architecture — models must be compiled to `.hef` (Hailo Executable Format) before deployment. This is why llama.cpp runs on CPU: transformer attention is not a supported dataflow pattern.

## Physical Installation

1. Power off the Raspberry Pi 5 completely
2. Connect the FPC ribbon cable: one end to Hailo-8 HAT PCIe connector, other end to RPi 5 HAT+ connector
3. Orient ribbon cable with blue tab facing up on both ends
4. Seat the Hailo-8 HAT on the 40-pin GPIO header
5. Secure with M2.5 standoffs (4 corners)
6. Power on

> **Note:** The RPi 5 has ONE PCIe 2.0 x1 interface shared between the HAT+ connector and the M.2 HAT (NVMe). Cannot use both simultaneously without a PCIe switch.

## Software Installation

```bash
sudo apt update
sudo apt install -y hailo-all
sudo reboot
```

`hailo-all` installs:
- `hailort-dkms` — PCIe kernel module (reboot required to load)
- `hailort` — firmware and runtime library (`/usr/lib/libhailort.so`)
- `hailo-tappas-core` — GStreamer pipeline framework
- `hailortcli` — command-line management tool

## Verification

```bash
# Should show: Hailo Devices: [-] Device: 0001:01:00.0
hailortcli scan

# Confirmed device node
ls -la /dev/hailo*

# Read firmware config (USER_CONFIG_NOT_LOADED warning is normal on fresh device)
hailortcli fw-config read
```

> **The `USER_CONFIG_NOT_LOADED` warning is normal.** It means no custom user firmware configuration has been written. The device is fully functional. This is expected on a fresh Hailo-8.

## Setting Device Permissions

```bash
sudo usermod -aG hailo $USER
# Log out and back in, then verify:
groups  # 'hailo' should appear
```

## Downloading Model Files (.hef)

```bash
mkdir -p ~/models

# YOLOv8s object detection (~12 MB)
wget "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/yolov8s.hef" \
  -O ~/models/yolov8s_h8.hef

# Face detection (~4 MB)
wget "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/scrfd_10g.hef" \
  -O ~/models/face_detection_h8.hef
```

Browse the full model zoo: https://github.com/hailo-ai/hailo_model_zoo

## Running Inference with TAPPAS

### Static image test:
```bash
gst-launch-1.0 \
  filesrc location=test.jpg ! jpegdec ! videoconvert ! \
  video/x-raw,format=RGB ! \
  hailonet hef-path=~/models/yolov8s_h8.hef ! \
  hailofilter so-path=/usr/lib/hailo-post-processes/libyolo_hailortpp.so \
    function-name=yolov8 ! \
  hailooverlay ! videoconvert ! jpegenc ! filesink location=output.jpg
```

### USB webcam:
```bash
gst-launch-1.0 \
  v4l2src device=/dev/video0 ! \
  videoconvert ! video/x-raw,width=640,height=480,format=RGB ! \
  hailonet hef-path=~/models/yolov8s_h8.hef ! \
  hailofilter so-path=/usr/lib/hailo-post-processes/libyolo_hailortpp.so \
    function-name=yolov8 ! \
  hailooverlay ! videoconvert ! autovideosink
```

### RTSP stream:
```bash
gst-launch-1.0 \
  rtspsrc location=rtsp://<camera-ip>/stream ! rtph264depay ! avdec_h264 ! \
  videoconvert ! video/x-raw,format=RGB ! \
  hailonet hef-path=~/models/yolov8s_h8.hef ! \
  hailofilter so-path=/usr/lib/hailo-post-processes/libyolo_hailortpp.so \
    function-name=yolov8 ! \
  hailooverlay ! autovideosink
```

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `/dev/hailo0` not found | Driver not loaded | Reboot after `apt install hailo-all` |
| `hailortcli scan` shows no devices | PCIe link not established | Reseat HAT and ribbon cable |
| Permission denied on `/dev/hailo0` | User not in hailo group | `sudo usermod -aG hailo $USER` then re-login |
| GStreamer element `hailonet` not found | TAPPAS not installed | `sudo apt install hailo-tappas-core` |
| `USER_CONFIG_NOT_LOADED` warning | Normal on fresh device | Not an error — ignore |
| Slow inference / thermal throttle | Overheating | Add active cooling, run `vcgencmd measure_temp` |
