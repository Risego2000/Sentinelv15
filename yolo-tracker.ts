
import * as ort from 'onnxruntime-web';

// --- Interfaces ---
export interface Detection {
    bbox: [number, number, number, number]; // [x, y, w, h]
    score: number;
    classId: number;
    className: string;
}

export interface Keypoint {
    x: number;
    y: number;
    confidence: number;
}

export interface PoseDetection extends Detection {
    keypoints: Keypoint[]; // 17 keypoints (COCO format)
}

export interface Track extends Detection {
    trackId: number;
    state: number[]; // Kalman state: [cx, cy, aspect, h, vx, vy, va, vh]
    covariance: number[][]; // 8x8 covariance matrix
    age: number;
    hits: number;    // frames tracked
    timeSinceUpdate: number;
    appearance?: number[]; // BoT-SORT: appearance feature vector
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
    appearance?: number[]; // BoT-SORT: Color histogram as appearance feature

    // Standard Kalman params
    private static readonly R_STD = [10, 10, 10, 10]; // Measurement noise
    private static readonly Q_STD = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]; // Process noise (low for constant velocity)

    constructor(bbox: [number, number, number, number], appearance?: number[]) {
        this.trackId = KalmanBoxTracker.count++;

        // Initial State: x, y, s, h, 0, 0, 0, 0
        // bbox: x, y, w, h (top-left) -> convert to center-based cx, cy, s, h
        const cx = bbox[0] + bbox[2] / 2;
        const cy = bbox[1] + bbox[3] / 2;
        const s = bbox[2] * bbox[3]; // Area
        const h = bbox[3];
        this.state = [cx, cy, s, h, 0, 0, 0, 0];
        this.appearance = appearance;

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
    update(bbox: [number, number, number, number], appearance?: number[]) {
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

        // Update appearance with smooth blending (BoT-SORT)
        if (appearance && this.appearance) {
            const beta = 0.9; // High momentum for appearance
            this.appearance = this.appearance.map((v, i) => v * beta + appearance[i] * (1 - beta));
        } else if (appearance) {
            this.appearance = appearance;
        }
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

// --- Cosine Similarity for Appearance Matching (BoT-SORT) ---
function cosineSimilarity(a: number[], b: number[]): number {
    if (!a || !b || a.length !== b.length) return 0;
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

// --- ByteTrack Core ---
export class ByteTracker {
    tracks: KalmanBoxTracker[] = [];
    frameId = 0;

    // Configurable parameters
    highThresh = 0.5;
    matchThresh = 0.25;
    trackBufferFrames = 30; // Frames to keep lost tracks

    update(detections: Detection[]): Track[] {
        this.frameId++;

        // 1. Divide detections
        const highDets = detections.filter(d => d.score >= this.highThresh);
        const lowDets = detections.filter(d => d.score < this.highThresh && d.score > 0.1);

        // 2. Predict tracks
        this.tracks.forEach(t => t.predict());

        // 3. Match High Conf
        const trackIndices = Array.from(this.tracks.keys());
        const highDetIndices = Array.from(highDets.keys());

        const { matches: matches1, unmatchedTracks: uTracks1, unmatchedDets: uDets1 }
            = this.match(this.tracks, highDets, trackIndices, highDetIndices, this.matchThresh);

        // Update matched tracks
        matches1.forEach((m) => {
            this.tracks[m[0]].update(highDets[m[1]].bbox);
        });

        // 4. Match Low Conf with Unmatched Tracks (uTracks1)
        const { matches: matches2, unmatchedTracks: uTracks2, unmatchedDets: uDets2 }
            = this.match(this.tracks, lowDets, uTracks1, Array.from(lowDets.keys()), this.matchThresh);

        // Update matched low-conf tracks
        matches2.forEach((m) => {
            this.tracks[m[0]].update(lowDets[m[1]].bbox);
        });

        // 5. Create new tracks from Unmatched High Conf Dets (uDets1)
        uDets1.forEach(idx => {
            const d = highDets[idx];
            this.tracks.push(new KalmanBoxTracker(d.bbox));
        });

        // 6. Remove lost tracks using configurable buffer
        this.tracks = this.tracks.filter(t => t.timeSinceUpdate < this.trackBufferFrames);

        // Return tracks for display
        return this.tracks.map(t => {
            const [x, y, w, h] = t.getBBox();
            return {
                trackId: t.trackId,
                bbox: [x, y, w, h],
                score: 1.0,
                classId: 0,
                className: 'vehicle',
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

// === BoT-SORT Implementation ===
export class BoTSORT {
    tracks: KalmanBoxTracker[] = [];
    frameId = 0;

    // Configurable parameters
    highThresh = 0.6;
    matchThresh = 0.25;
    trackBufferFrames = 40; // Longer buffer for re-identification
    appearanceWeight = 0.5; // Balance between IoU and appearance

    // Extract simple color histogram as appearance feature
    extractAppearance(imageData: ImageData, bbox: [number, number, number, number]): number[] {
        const [x, y, w, h] = bbox.map(v => Math.floor(v));
        const histogram = new Array(16).fill(0); // 16-bin RGB histogram (simplified)

        let count = 0;
        for (let py = Math.max(0, y); py < Math.min(imageData.height, y + h); py++) {
            for (let px = Math.max(0, x); px < Math.min(imageData.width, x + w); px++) {
                const idx = (py * imageData.width + px) * 4;
                const r = Math.floor(imageData.data[idx] / 64); // 0-3
                const g = Math.floor(imageData.data[idx + 1] / 64);
                const b = Math.floor(imageData.data[idx + 2] / 64);
                const binIdx = r * 4 + g; // Simple 16-bin histogram
                histogram[binIdx]++;
                count++;
            }
        }

        // Normalize
        return histogram.map(v => v / (count + 1e-8));
    }

    update(detections: Detection[], videoFrame?: HTMLVideoElement): Track[] {
        this.frameId++;

        // Extract appearance features if video frame provided
        let appearances: (number[] | undefined)[] = [];
        if (videoFrame) {
            const canvas = document.createElement('canvas');
            canvas.width = videoFrame.videoWidth;
            canvas.height = videoFrame.videoHeight;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.drawImage(videoFrame, 0, 0);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                appearances = detections.map(d => this.extractAppearance(imageData, d.bbox));
            }
        }

        // 1. Divide detections
        const highDets = detections.filter(d => d.score >= this.highThresh);
        const lowDets = detections.filter(d => d.score < this.highThresh && d.score > 0.1);
        const highApps = appearances.slice(0, highDets.length);
        const lowApps = appearances.slice(highDets.length);

        // 2. Predict tracks
        this.tracks.forEach(t => t.predict());

        // 3. Match High Conf with Appearance
        const trackIndices = Array.from(this.tracks.keys());
        const highDetIndices = Array.from(highDets.keys());

        const { matches: matches1, unmatchedTracks: uTracks1, unmatchedDets: uDets1 }
            = this.matchWithAppearance(this.tracks, highDets, trackIndices, highDetIndices, this.matchThresh, highApps);

        // Update matched tracks
        matches1.forEach((m) => {
            this.tracks[m[0]].update(highDets[m[1]].bbox, highApps[m[1]]);
        });

        // 4. Re-ID: Try to match unmatched tracks with low detections (lost objects recovery)
        const { matches: matches2, unmatchedTracks: uTracks2, unmatchedDets: uDets2 }
            = this.matchWithAppearance(this.tracks, lowDets, uTracks1, Array.from(lowDets.keys()), this.matchThresh * 0.7, lowApps);

        // Update re-identified tracks
        matches2.forEach((m) => {
            this.tracks[m[0]].update(lowDets[m[1]].bbox, lowApps[m[1]]);
        });

        // 5. Create new tracks from Unmatched High Conf Dets
        uDets1.forEach(idx => {
            const d = highDets[idx];
            this.tracks.push(new KalmanBoxTracker(d.bbox, highApps[idx]));
        });

        // 6. Remove lost tracks
        this.tracks = this.tracks.filter(t => t.timeSinceUpdate < this.trackBufferFrames);

        // Return tracks
        return this.tracks.map(t => {
            const [x, y, w, h] = t.getBBox();
            return {
                trackId: t.trackId,
                bbox: [x, y, w, h],
                score: 1.0,
                classId: 0,
                className: 'vehicle',
                state: t.state,
                covariance: t.P,
                age: t.age,
                hits: t.hits,
                timeSinceUpdate: t.timeSinceUpdate,
                appearance: t.appearance
            } as Track;
        });
    }

    // Match with combined IoU + Appearance similarity
    private matchWithAppearance(
        tracks: KalmanBoxTracker[],
        dets: Detection[],
        trackIndices: number[],
        detIndices: number[],
        iouThresh: number,
        appearances: (number[] | undefined)[]
    ) {
        const matches: [number, number][] = [];
        const unmatchedTracks = new Set(trackIndices);
        const unmatchedDets = new Set(detIndices);

        const candidates: { tIdx: number, dIdx: number, score: number }[] = [];

        trackIndices.forEach(tIdx => {
            detIndices.forEach(dIdx => {
                const iouScore = iou(tracks[tIdx].getBBox(), dets[dIdx].bbox);

                // Combine IoU with appearance if available
                let finalScore = iouScore;
                if (tracks[tIdx].appearance && appearances[dIdx]) {
                    const appSim = cosineSimilarity(tracks[tIdx].appearance!, appearances[dIdx]!);
                    finalScore = (1 - this.appearanceWeight) * iouScore + this.appearanceWeight * appSim;
                }

                if (finalScore >= iouThresh) {
                    candidates.push({ tIdx, dIdx, score: finalScore });
                }
            });
        });

        candidates.sort((a, b) => b.score - a.score);

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

    async load(modelPath: string, poseModelPath?: string) {
        try {
            // Configure WASM paths to ensure they are found in public/ or root
            // Use BASE_URL to support GitHub Pages subdirectory deployment
            ort.env.wasm.wasmPaths = import.meta.env.BASE_URL;

            this.session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['wasm'] // Start with WASM CPU for stability
            });
            console.log("YOLOv11 Loaded", this.session.outputNames, this.session.inputNames);

            // Load pose model if provided
            if (poseModelPath) {
                this.poseSession = await ort.InferenceSession.create(poseModelPath, {
                    executionProviders: ['wasm']
                });
                console.log("YOLOv11-Pose Loaded", this.poseSession.outputNames, this.poseSession.inputNames);
            }
        } catch (e) {
            console.error("Failed to load YOLO ONNX model", e);
            throw e;
        }
    }

    async detectPose(video: HTMLVideoElement, confThreshold: number = 0.5): Promise<PoseDetection[]> {
        if (!this.poseSession) return [];

        // 1. Preprocess (same as detect)
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 640;
        const ctx = canvas.getContext('2d');
        if (!ctx) return [];

        ctx.drawImage(video, 0, 0, 640, 640);
        const imgData = ctx.getImageData(0, 0, 640, 640);

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
        const feeds = { images: tensor };
        const results = await this.poseSession.run(feeds);

        // 3. Postprocess Pose Output
        // YOLO Pose output: [1, 56, 8400] or [1, 8400, 56]
        // 56 = 4 (bbox) + 1 (conf) + 51 (17 keypoints * 3 [x,y,conf])
        const output = results[Object.keys(results)[0]].data as Float32Array;
        const dims = results[Object.keys(results)[0]].dims;

        let numAnchors = 8400;
        let isTransposed = false;

        if (dims && dims.length === 3) {
            if (dims[1] > dims[2]) {
                isTransposed = true;
                numAnchors = dims[1];
            } else {
                numAnchors = dims[2];
            }
        }

        const predictions: PoseDetection[] = [];
        const scaleX = video.videoWidth / 640;
        const scaleY = video.videoHeight / 640;

        for (let i = 0; i < numAnchors; i++) {
            let cx, cy, w, h, score;

            if (isTransposed) {
                // [1, 8400, 56]
                const row = i * 56;
                cx = output[row + 0];
                cy = output[row + 1];
                w = output[row + 2];
                h = output[row + 3];
                score = output[row + 4];
            } else {
                // [1, 56, 8400]
                cx = output[0 * numAnchors + i];
                cy = output[1 * numAnchors + i];
                w = output[2 * numAnchors + i];
                h = output[3 * numAnchors + i];
                score = output[4 * numAnchors + i];
            }

            if (score > confThreshold) {
                // Extract 17 keypoints
                const keypoints: Keypoint[] = [];
                for (let k = 0; k < 17; k++) {
                    let kx, ky, kconf;
                    if (isTransposed) {
                        const row = i * 56;
                        kx = output[row + 5 + k * 3];
                        ky = output[row + 5 + k * 3 + 1];
                        kconf = output[row + 5 + k * 3 + 2];
                    } else {
                        kx = output[(5 + k * 3) * numAnchors + i];
                        ky = output[(5 + k * 3 + 1) * numAnchors + i];
                        kconf = output[(5 + k * 3 + 2) * numAnchors + i];
                    }

                    keypoints.push({
                        x: kx * scaleX,
                        y: ky * scaleY,
                        confidence: kconf
                    });
                }

                predictions.push({
                    bbox: [(cx - w / 2) * scaleX, (cy - h / 2) * scaleY, w * scaleX, h * scaleY],
                    score,
                    classId: 0, // Person
                    className: 'person',
                    keypoints
                });
            }
        }

        return this.nmsPose(predictions);
    }

    nmsPose(detections: PoseDetection[], iouThresh = 0.45): PoseDetection[] {
        detections.sort((a, b) => b.score - a.score);
        const selected: PoseDetection[] = [];

        while (detections.length > 0) {
            const current = detections.shift()!;
            selected.push(current);

            detections = detections.filter(d => iou(current.bbox, d.bbox) < iouThresh);
        }

        return selected;
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
        const dims = results[Object.keys(results)[0]].dims;

        let numAnchors = 8400;
        let numClasses = 80;
        let isTransposed = false;

        // Auto-detect shape
        // [1, 84, 8400] -> Default
        // [1, 8400, 84] -> Transposed
        // We assume dim with >1000 is anchors
        if (dims && dims.length === 3) {
            if (dims[1] > dims[2]) {
                isTransposed = true;
                numAnchors = dims[1];
                numClasses = dims[2] - 4;
            } else {
                isTransposed = false;
                numAnchors = dims[2];
                numClasses = dims[1] - 4;
            }
        }

        // Debug
        // console.log("YOLO Shape:", dims, "Transposed:", isTransposed);

        const predictions: Detection[] = [];

        for (let i = 0; i < numAnchors; i++) {
            let maxScore = 0;
            let maxClass = 0;

            // Find class with max score
            for (let c = 0; c < numClasses; c++) {
                let score = 0;
                if (isTransposed) {
                    // [1, 8400, 84] -> data[i * 84 + (4 + c)]
                    score = output[i * (numClasses + 4) + (4 + c)];
                } else {
                    // [1, 84, 8400] -> data[(4 + c) * 8400 + i]
                    score = output[(4 + c) * numAnchors + i];
                }

                if (score > maxScore) {
                    maxScore = score;
                    maxClass = c;
                }
            }

            if (maxScore > confThreshold) {
                let cx, cy, w, h;
                if (isTransposed) {
                    const row = i * (numClasses + 4);
                    cx = output[row + 0];
                    cy = output[row + 1];
                    w = output[row + 2];
                    h = output[row + 3];
                } else {
                    cx = output[0 * numAnchors + i];
                    cy = output[1 * numAnchors + i];
                    w = output[2 * numAnchors + i];
                    h = output[3 * numAnchors + i];
                }

                // Scale to video size
                const scaleX = video.videoWidth / 640;
                const scaleY = video.videoHeight / 640;

                predictions.push({
                    bbox: [(cx - w / 2) * scaleX, (cy - h / 2) * scaleY, w * scaleX, h * scaleY],
                    score: maxScore,
                    classId: maxClass,
                    className: COCO_CLASSES[maxClass] || 'unknown'
                });
            }
        }

        return this.nms(predictions);
    }

    // Helper NMS
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
