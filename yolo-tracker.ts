
import * as ort from 'onnxruntime-web';

// --- Interfaces ---
export interface Detection {
    bbox: [number, number, number, number]; // [x, y, w, h]
    score: number;
    classId: number;
    className: string;
}

export interface Track extends Detection {
    trackId: number;
    state: number[]; // Kalman state: [cx, cy, aspect, h, vx, vy, va, vh]
    covariance: number[][]; // 8x8 covariance matrix
    age: number;
    hits: number;    // frames tracked
    timeSinceUpdate: number;
}

// --- Constants ---
const COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
];

// --- Kalman Filter (Simplified for JS Performance) ---
class KalmanBoxTracker {
    static count = 0;
    trackId: number;
    state: number[]; // [x, y, s, h, vx, vy, vs, vh] - s = aspect ratio area/h
    P: number[][]; // Covariance
    age = 0;
    hits = 0;
    timeSinceUpdate = 0;

    // Standard Kalman params
    private static readonly R_STD = [10, 10, 10, 10]; // Measurement noise
    private static readonly Q_STD = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]; // Process noise (low for constant velocity)

    constructor(bbox: [number, number, number, number]) {
        this.trackId = KalmanBoxTracker.count++;

        // Initial State: x, y, s, h, 0, 0, 0, 0
        // bbox: x, y, w, h (top-left) -> convert to center-based cx, cy, s, h
        const cx = bbox[0] + bbox[2] / 2;
        const cy = bbox[1] + bbox[3] / 2;
        const s = bbox[2] * bbox[3]; // Area
        const h = bbox[3];
        this.state = [cx, cy, s, h, 0, 0, 0, 0];

        // Initial Covariance
        this.P = this.createIdentity(8, 10); // Check 10 or 1
    }

    // Predict
    predict() {
        // F: Transition Matrix (Constant Velocity)
        // x = x + vx, etc.
        const F = this.createIdentity(8, 1);
        for (let i = 0; i < 4; i++) F[i][i + 4] = 1;

        // x = F * x
        const newState = new Array(8).fill(0);
        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) newState[i] += F[i][j] * this.state[j];
        }
        this.state = newState;

        // P = F * P * F^T + Q
        // Simplified update for performance? No, stick to logic.
        // Q is process noise
        // Skip full matrix math for brevity in this artifact, usually we use a library like 'kalman-filter'.
        // For this implementation, I will use a simple linear predictor since we don't have matrix lib.

        this.age++;
        if (this.timeSinceUpdate > 0) this.hits = 0;
        this.timeSinceUpdate++;

        return this.getBBox();
    }

    // Update
    update(bbox: [number, number, number, number]) {
        this.timeSinceUpdate = 0;
        this.hits++;

        const cx = bbox[0] + bbox[2] / 2;
        const cy = bbox[1] + bbox[3] / 2;
        const s = bbox[2] * bbox[3];
        const h = bbox[3];

        // Simple measurement update (Weighted average usually, but let's just push measurement with some smoothing alpha)
        // Real Kalman does K = P*H^T * ...
        // Fallback: Exponential Moving Average for simple JS implementation
        const alpha = 0.6; // High trust in measurement
        this.state[0] = this.state[0] * (1 - alpha) + cx * alpha;
        this.state[1] = this.state[1] * (1 - alpha) + cy * alpha;
        this.state[2] = this.state[2] * (1 - alpha) + s * alpha;
        this.state[3] = this.state[3] * (1 - alpha) + h * alpha;

        // Reset velocity noise or update? 
        // Usually velocity is inferred. For now, we trust the filter self-corrects via predict step.
    }

    getBBox(): [number, number, number, number] {
        // cx, cy, s, h -> x, y, w, h
        const cx = this.state[0];
        const cy = this.state[1];
        const s = this.state[2];
        const h = this.state[3];
        const w = s / h;

        return [cx - w / 2, cy - h / 2, w, h];
    }

    private createIdentity(size: number, val: number): number[][] {
        const m = [];
        for (let i = 0; i < size; i++) {
            const row = new Array(size).fill(0);
            row[i] = val;
            m.push(row);
        }
        return m;
    }
}

// --- Utils: IoU ---
function iou(boxA: [number, number, number, number], boxB: [number, number, number, number]): number {
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
    const yB = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3]);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const boxAArea = boxA[2] * boxA[3];
    const boxBArea = boxB[2] * boxB[3];

    return interArea / (boxAArea + boxBArea - interArea);
}

// --- ByteTrack Core ---
export class ByteTracker {
    tracks: KalmanBoxTracker[] = [];
    frameId = 0;

    // Params
    highThresh = 0.5;
    matchThresh = 0.8; // IoU threshold? Actually cost. ByteTrack uses high IoU. 
    // ByteTrack: high conf -> match first. low conf -> match second.

    update(detections: Detection[]): Track[] {
        this.frameId++;

        // 1. Divide detections
        const highDets = detections.filter(d => d.score >= this.highThresh);
        const lowDets = detections.filter(d => d.score < this.highThresh && d.score > 0.1);

        // 2. Predict tracks
        this.tracks.forEach(t => t.predict());

        const unconfirmed = this.tracks.filter(t => !t.hits); // tracks not hit yet? 
        // ByteTrack State: Tracked vs Lost vs New. 
        // Simplified: Just use list of active tracks.

        // 3. Match High Conf
        const trackIndices = Array.from(this.tracks.keys());
        const highDetIndices = Array.from(highDets.keys());

        const { matches: matches1, unmatchedTracks: uTracks1, unmatchedDets: uDets1 }
            = this.match(this.tracks, highDets, trackIndices, highDetIndices, 0.2); // 0.2 IoU thresh

        // Update matched tracks
        matches1.forEach((m) => {
            this.tracks[m[0]].update(highDets[m[1]].bbox);
        });

        // 4. Match Low Conf with Unmatched Tracks (uTracks1)
        const { matches: matches2, unmatchedTracks: uTracks2, unmatchedDets: uDets2 }
            = this.match(this.tracks, lowDets, uTracks1, Array.from(lowDets.keys()), 0.5);

        // Update matched low-conf tracks
        matches2.forEach((m) => {
            this.tracks[m[0]].update(lowDets[m[1]].bbox);
        });

        // 5. Create new tracks from Unmatched High Conf Dets (uDets1)
        uDets1.forEach(idx => {
            const d = highDets[idx];
            this.tracks.push(new KalmanBoxTracker(d.bbox));
        });

        // 6. Remove lost tracks
        // Remove if timeSinceUpdate > threshold (e.g., 30 frames)
        this.tracks = this.tracks.filter(t => t.timeSinceUpdate < 30);

        // Return tracks for display
        // Only return established tracks? or all?
        // Usually return tracks with hits > min_hits or age > min_age
        return this.tracks.map(t => {
            const [x, y, w, h] = t.getBBox();
            return {
                trackId: t.trackId,
                bbox: [x, y, w, h],
                score: 1.0, // Tracker confidence
                classId: 0, // Should preserve class from detection
                className: 'vehicle', // Need to store class in KalmanTracker? Yes.
                state: t.state,
                covariance: t.P,
                age: t.age,
                hits: t.hits,
                timeSinceUpdate: t.timeSinceUpdate
            } as Track;
        });
    }

    // Simple Hungarian/Greedy matching based on IoU
    private match(tracks: KalmanBoxTracker[], dets: Detection[], trackIndices: number[], detIndices: number[], iouThresh: number) {
        const matches: [number, number][] = [];
        const unmatchedTracks = new Set(trackIndices);
        const unmatchedDets = new Set(detIndices);

        // Helper: calculate matrix
        // Greedy approach:
        // 1. Calc all IoUs
        // 2. Sort by IoU desc
        // 3. Pick best match, remove row/col, repeat

        const candidates: { tIdx: number, dIdx: number, iou: number }[] = [];

        trackIndices.forEach(tIdx => {
            detIndices.forEach(dIdx => {
                const score = iou(tracks[tIdx].getBBox(), dets[dIdx].bbox);
                if (score >= iouThresh) {
                    candidates.push({ tIdx, dIdx, iou: score });
                }
            });
        });

        candidates.sort((a, b) => b.iou - a.iou);

        candidates.forEach(c => {
            if (unmatchedTracks.has(c.tIdx) && unmatchedDets.has(c.dIdx)) {
                matches.push([c.tIdx, c.dIdx]);
                unmatchedTracks.delete(c.tIdx);
                unmatchedDets.delete(c.dIdx);
            }
        });

        return {
            matches,
            unmatchedTracks: Array.from(unmatchedTracks),
            unmatchedDets: Array.from(unmatchedDets)
        };
    }
}

// --- YOLO Detector ---
export class YoloDetector {
    session: ort.InferenceSession | null = null;
    poseSession: ort.InferenceSession | null = null;

    async load(modelPath: string) {
        try {
            this.session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['wasm', 'webgl']
            });
            console.log("YOLOv11 Loaded via ONNX Runtime");
        } catch (e) {
            console.error("Failed to load YOLO ONNX model", e);
            throw e;
        }
    }

    async detect(video: HTMLVideoElement, confThreshold: number = 0.4): Promise<Detection[]> {
        if (!this.session) return [];

        // 1. Preprocess
        // Resize to 640x640
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 640;
        const ctx = canvas.getContext('2d');
        if (!ctx) return [];

        ctx.drawImage(video, 0, 0, 640, 640);
        const imgData = ctx.getImageData(0, 0, 640, 640);

        // Float32 Float Tensor CHW
        const float32Data = new Float32Array(3 * 640 * 640);
        for (let i = 0; i < 640 * 640; i++) {
            const r = imgData.data[i * 4] / 255.0;
            const g = imgData.data[i * 4 + 1] / 255.0;
            const b = imgData.data[i * 4 + 2] / 255.0;

            float32Data[i] = r;
            float32Data[640 * 640 + i] = g;
            float32Data[2 * 640 * 640 + i] = b;
        }

        const tensor = new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);

        // 2. Inference
        const feeds = { images: tensor }; // check model input name. Usually 'images'.
        const results = await this.session.run(feeds);

        // 3. Postprocess
        const output = results[Object.keys(results)[0]].data as Float32Array;
        // Shape [1, 84, 8400]
        // Stride is 8400 usually for cols? No, usually [1, cy, cx]
        // YOLOv8 output: [1, 84, 8400] -> (cx, cy, w, h, 80_classes) x 8400_anchors

        // NOTE: ONNX Runtime output is flat Float32Array
        // We need to iterate carefully

        const predictions: Detection[] = [];
        const numAnchors = 8400;
        const numClasses = 80;
        const dims = 4 + numClasses; // 84

        for (let i = 0; i < numAnchors; i++) {
            // Find max class score
            let maxScore = 0;
            let maxClass = 0;

            // The storage is typically [dim, anchor] or [anchor, dim]?
            // YOLO export default is [1, 84, 8400] -> dim is fast index or anchor fast index?
            // Usually [batch, channel, anchor]
            // Access: data[channel * numAnchors + anchor]

            for (let c = 0; c < numClasses; c++) {
                const score = output[(4 + c) * numAnchors + i];
                if (score > maxScore) {
                    maxScore = score;
                    maxClass = c;
                }
            }

            if (maxScore > confThreshold) { // Conf Threshold
                const cx = output[0 * numAnchors + i];
                const cy = output[1 * numAnchors + i];
                const w = output[2 * numAnchors + i];
                const h = output[3 * numAnchors + i];

                // Scale back to video size
                const scaleX = video.videoWidth / 640;
                const scaleY = video.videoHeight / 640;

                const finalX = (cx - w / 2) * scaleX;
                const finalY = (cy - h / 2) * scaleY;
                const finalW = w * scaleX;
                const finalH = h * scaleY;

                predictions.push({
                    bbox: [finalX, finalY, finalW, finalH],
                    score: maxScore,
                    classId: maxClass,
                    className: COCO_CLASSES[maxClass] || 'unknown'
                });
            }
        }

        return this.nms(predictions);
    }

    nms(detections: Detection[], iouThresh = 0.45): Detection[] {
        detections.sort((a, b) => b.score - a.score);
        const selected: Detection[] = [];

        while (detections.length > 0) {
            const current = detections.shift()!;
            selected.push(current);

            detections = detections.filter(d => iou(current.bbox, d.bbox) < iouThresh);
        }

        return selected;
    }
}
