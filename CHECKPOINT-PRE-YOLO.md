# CHECKPOINT: MIGRATION TO YOLOv11 + BYTETRACK

**Date:** 2026-01-10
**Description:** This checkpoint captures the state of the application immediately before removing the TensorFlow.js/COCO-SSD detection system to replace it with YOLOv11 (ONNX) and a ByteTrack algorithm.

## State Preserved:
- **Detection Model:** TensorFlow.js COCO-SSD (via unpkg CDN).
- **Tracking Logic:** Custom Euclidian/IoU based tracking with Kalman-like smoothing.
- **Performance:** Optimized for browser (~30-60 FPS depending on device).
- **Files:** `index.tsx` contains the full monolithic logic.

## Reason for Migration:
- User requested YOLOv11n (Nano) for better accuracy/performance.
- Integration of "ByteTrack" or "BoT-SORT" logic for professional tracking.
- Removing dependency on legacy TFJS models.

## Backup Files:
- Ensure `index.tsx.backup-coco-ssd-*` exists before proceeding.
