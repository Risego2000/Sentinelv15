# YOLOv8 Model Setup Instructions

Due to download restrictions, download the YOLOv8n ONNX model manually:

## Option 1: From Ultralytics Hub (Recommended)
1. Visit: https://docs.ultralytics.com/models/yolov8/
2. Download YOLOv8n (nano) in ONNX format
3. Place in: `public/models/yolov8n.onnx`

## Option 2: Use pre-converted model from HuggingFace
```bash
curl -L "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/yolov8n.onnx" -o "public/models/yolov8n.onnx"
```

## Option 3: Convert from PyTorch (if you have Python)
```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx
```

## Alternative: Use a lighter CDN-hosted model
For now, I'll implement using a web-compatible alternative that doesn't require download.

---
**Status:** Implementing fallback to web-hosted YOLO model
