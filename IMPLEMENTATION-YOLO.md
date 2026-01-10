# IMPLEMENTATION PLAN: YOLOv11 & ByteTrack Migration

## 1. Objective
Replace the existing TensorFlow.js (COCO-SSD) detection system with **YOLOv11 Nano (ONNX)** and implement **ByteTrack** for professional-grade vehicle tracking.

## 2. Dependencies
- **Runtime:** `onnxruntime-web` (Already installed).
- **Models:**
  - `upload/yolo11n_640.onnx` (Detection)
  - `upload/yolo11n_pose.onnx` (Pose - Future/Secondary) for now we focus on main detection.
- **Tracker:** Custom TypeScript implementation of ByteTrack (Kalman Filter + Dual-Stage matching).

## 3. Architecture Changes

### A. Detection Engine (`src/engine/YoloDetector.ts`)
- **Loader:** Load ONNX models using `ort.InferenceSession`.
- **Pre-processing:**
  - Resize video frame to 640x640.
  - Normalize pixel data (0-1).
  - Transpose to CHW tensor layout.
- **Inference:** Run model.
- **Post-processing:**
  - Decode output tensor `[1, 84, 8400]` (cx, cy, w, h, ...classes).
  - Apply Efficient NMS (Non-Maximum Suppression).
  - Scale coordinates back to video dimensions.

### B. Tracking Engine (`src/engine/ByteTracker.ts`)
- **Kalman Filter:** State vector `[x, y, aspect, height, vx, vy, va, vh]`.
- **Matching Cascade (ByteTrack Logic):**
  1. **High-Confidence Match:** Associate high-conf detections with tracks (IoU).
  2. **Low-Confidence Match:** Associate remaining low-conf detections with unmatched tracks (IoU).
  3. **New Track Creation:** Create tracks from unmatched high-conf detections.
  4. **Lost Track Management:** Mark tracks as lost, delete after persistence threshold.

### C. Application Logic (`index.tsx`)
- Remove `cocoSsd.load()` and `model.detect()`.
- Initialize `YoloDetector` and `ByteTracker`.
- Update `processFrame` loop to use new pipeline.
- Update UI to reflect "YOLOv11n (ONNX)" status.

## 4. Execution Steps
1. Create `src/engine` directory.
2. Implement `KalmanFilter.ts`.
3. Implement `ByteTracker.ts`.
4. Implement `YoloDetector.ts`.
5. Refactor `index.tsx` to switch engines.
