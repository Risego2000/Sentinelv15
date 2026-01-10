/**
 * YOLO11 + ByteTrack Implementation
 * Optimized for browser performance with ONNX Runtime
 */

import * as ort from 'onnxruntime-web';

export interface Detection {
    bbox: [number, number, number, number]; // [x, y, width, height]
    score: number;
    class: string;
    classId: number;
}

export interface Track {
    id: number;
    bbox: [number, number, number, number];
    score: number;
    class: string;
    age: number;
    totalFrames: number;
    consecutiveFrames: number;
    state: 'tracked' | 'lost' | 'removed';
    velocity: [number, number]; // [vx, vy]
    kalmanState: {
        x: number;
        y: number;
        vx: number;
        vy: number;
    };
}

export class YOLO11Detector {
    private session: ort.InferenceSession | null = null;
    private poseSession: ort.InferenceSession | null = null;
    private inputSize = 640;

    async loadModels(modelPath: string, posePath: string) {
        try {
            this.session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            this.poseSession = await ort.InferenceSession.create(posePath, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            console.log('✅ YOLO11 models loaded successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to load YOLO11 models:', error);
            return false;
        }
    }

    async detect(videoElement: HTMLVideoElement, confThreshold: number = 0.3): Promise<Detection[]> {
        if (!this.session) return [];

        const { width: videoWidth, height: videoHeight } = videoElement;

        // Prepare input tensor
        const input = await this.preprocessImage(videoElement);
        const tensor = new ort.Tensor('float32', input, [1, 3, this.inputSize, this.inputSize]);

        // Run inference (non-blocking)
        const results = await this.session.run({ [this.session.inputNames[0]]: tensor });
        const output = results[this.session.outputNames[0]].data as Float32Array;

        // Parse detections
        const detections = this.parseOutput(output, confThreshold, videoWidth, videoHeight);

        // Apply NMS
        return this.applyNMS(detections, 0.45);
    }

    private async preprocessImage(videoElement: HTMLVideoElement): Promise<Float32Array> {
        const canvas = document.createElement('canvas');
        canvas.width = this.inputSize;
        canvas.height = this.inputSize;
        const ctx = canvas.getContext('2d')!;

        ctx.drawImage(videoElement, 0, 0, this.inputSize, this.inputSize);
        const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);

        const input = new Float32Array(3 * this.inputSize * this.inputSize);
        const { data } = imageData;

        for (let i = 0; i < this.inputSize * this.inputSize; i++) {
            input[i] = data[i * 4] / 255.0; // R
            input[i + this.inputSize * this.inputSize] = data[i * 4 + 1] / 255.0; // G
            input[i + 2 * this.inputSize * this.inputSize] = data[i * 4 + 2] / 255.0; // B
        }

        return input;
    }

    private parseOutput(
        output: Float32Array,
        confThreshold: number,
        videoWidth: number,
        videoHeight: number
    ): Detection[] {
        const detections: Detection[] = [];
        const numProposals = 8400;
        const numClasses = 80;

        const scaleX = videoWidth / this.inputSize;
        const scaleY = videoHeight / this.inputSize;

        for (let i = 0; i < numProposals; i++) {
            let maxScore = 0;
            let maxClassId = -1;

            // Find class with highest score
            for (let j = 0; j < numClasses; j++) {
                const score = output[(j + 4) * numProposals + i];
                if (score > maxScore) {
                    maxScore = score;
                    maxClassId = j;
                }
            }

            if (maxScore > confThreshold) {
                const cx = output[0 * numProposals + i] * scaleX;
                const cy = output[1 * numProposals + i] * scaleY;
                const w = output[2 * numProposals + i] * scaleX;
                const h = output[3 * numProposals + i] * scaleY;

                detections.push({
                    bbox: [cx - w / 2, cy - h / 2, w, h],
                    score: maxScore,
                    class: this.getClassName(maxClassId),
                    classId: maxClassId
                });
            }
        }

        return detections;
    }

    private applyNMS(detections: Detection[], iouThreshold: number): Detection[] {
        const sorted = [...detections].sort((a, b) => b.score - a.score);
        const keep: Detection[] = [];
        const suppressed = new Set<number>();

        for (let i = 0; i < sorted.length; i++) {
            if (suppressed.has(i)) continue;

            keep.push(sorted[i]);

            for (let j = i + 1; j < sorted.length; j++) {
                if (suppressed.has(j)) continue;

                const iou = this.calculateIoU(sorted[i].bbox, sorted[j].bbox);
                if (iou > iouThreshold) {
                    suppressed.add(j);
                }
            }
        }

        return keep;
    }

    private calculateIoU(boxA: number[], boxB: number[]): number {
        const xA = Math.max(boxA[0], boxB[0]);
        const yA = Math.max(boxA[1], boxB[1]);
        const xB = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
        const yB = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3]);

        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        const boxAArea = boxA[2] * boxA[3];
        const boxBArea = boxB[2] * boxB[3];

        return interArea / (boxAArea + boxBArea - interArea);
    }

    private getClassName(classId: number): string {
        const classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ];
        return classes[classId] || 'unknown';
    }
}

export class ByteTracker {
    private tracks: Track[] = [];
    private nextId = 1;
    private maxAge = 30;
    private minHits = 3;
    private iouThreshold = 0.3;

    update(detections: Detection[], deltaTime: number = 0.033): Track[] {
        // Predict new locations
        this.predict(deltaTime);

        // Match detections to tracks
        const { matched, unmatchedDetections, unmatchedTracks } = this.associate(detections);

        // Update matched tracks
        matched.forEach(([trackIdx, detIdx]) => {
            this.updateTrack(this.tracks[trackIdx], detections[detIdx]);
        });

        // Mark unmatched tracks as lost
        unmatchedTracks.forEach(idx => {
            this.tracks[idx].consecutiveFrames = 0;
            this.tracks[idx].state = 'lost';
        });

        // Create new tracks for unmatched detections
        unmatchedDetections.forEach(idx => {
            this.initTrack(detections[idx]);
        });

        // Remove old tracks
        this.tracks = this.tracks.filter(t =>
            t.age < this.maxAge && (t.state === 'tracked' || t.consecutiveFrames >= this.minHits)
        );

        return this.tracks.filter(t => t.state === 'tracked' && t.consecutiveFrames >= this.minHits);
    }

    private predict(deltaTime: number) {
        this.tracks.forEach(track => {
            track.age++;
            track.kalmanState.x += track.kalmanState.vx * deltaTime;
            track.kalmanState.y += track.kalmanState.vy * deltaTime;
            track.bbox[0] = track.kalmanState.x - track.bbox[2] / 2;
            track.bbox[1] = track.kalmanState.y - track.bbox[3] / 2;
        });
    }

    private associate(detections: Detection[]): {
        matched: [number, number][];
        unmatchedDetections: number[];
        unmatchedTracks: number[];
    } {
        if (this.tracks.length === 0) {
            return {
                matched: [],
                unmatchedDetections: detections.map((_, i) => i),
                unmatchedTracks: []
            };
        }

        // Calculate IoU matrix
        const iouMatrix: number[][] = [];
        for (let i = 0; i < this.tracks.length; i++) {
            iouMatrix[i] = [];
            for (let j = 0; j < detections.length; j++) {
                iouMatrix[i][j] = this.calculateIoU(this.tracks[i].bbox, detections[j].bbox);
            }
        }

        // Hungarian algorithm (simplified greedy matching)
        const matched: [number, number][] = [];
        const matchedTracks = new Set<number>();
        const matchedDetections = new Set<number>();

        // Sort by IoU descending
        const pairs: [number, number, number][] = [];
        for (let i = 0; i < this.tracks.length; i++) {
            for (let j = 0; j < detections.length; j++) {
                if (iouMatrix[i][j] > this.iouThreshold) {
                    pairs.push([i, j, iouMatrix[i][j]]);
                }
            }
        }
        pairs.sort((a, b) => b[2] - a[2]);

        pairs.forEach(([trackIdx, detIdx]) => {
            if (!matchedTracks.has(trackIdx) && !matchedDetections.has(detIdx)) {
                matched.push([trackIdx, detIdx]);
                matchedTracks.add(trackIdx);
                matchedDetections.add(detIdx);
            }
        });

        const unmatchedDetections = detections
            .map((_, i) => i)
            .filter(i => !matchedDetections.has(i));

        const unmatchedTracks = this.tracks
            .map((_, i) => i)
            .filter(i => !matchedTracks.has(i));

        return { matched, unmatchedDetections, unmatchedTracks };
    }

    private updateTrack(track: Track, detection: Detection) {
        const cx = detection.bbox[0] + detection.bbox[2] / 2;
        const cy = detection.bbox[1] + detection.bbox[3] / 2;

        // Kalman update
        const alpha = 0.7; // Smoothing factor
        track.kalmanState.vx = alpha * track.kalmanState.vx + (1 - alpha) * (cx - track.kalmanState.x);
        track.kalmanState.vy = alpha * track.kalmanState.vy + (1 - alpha) * (cy - track.kalmanState.y);
        track.kalmanState.x = cx;
        track.kalmanState.y = cy;

        track.bbox = detection.bbox;
        track.score = detection.score;
        track.consecutiveFrames++;
        track.totalFrames++;
        track.state = 'tracked';
    }

    private initTrack(detection: Detection) {
        const cx = detection.bbox[0] + detection.bbox[2] / 2;
        const cy = detection.bbox[1] + detection.bbox[3] / 2;

        this.tracks.push({
            id: this.nextId++,
            bbox: detection.bbox,
            score: detection.score,
            class: detection.class,
            age: 0,
            totalFrames: 1,
            consecutiveFrames: 1,
            state: 'tracked',
            velocity: [0, 0],
            kalmanState: {
                x: cx,
                y: cy,
                vx: 0,
                vy: 0
            }
        });
    }

    private calculateIoU(boxA: number[], boxB: number[]): number {
        const xA = Math.max(boxA[0], boxB[0]);
        const yA = Math.max(boxA[1], boxB[1]);
        const xB = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
        const yB = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3]);

        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        const boxAArea = boxA[2] * boxA[3];
        const boxBArea = boxB[2] * boxB[3];

        return interArea / (boxAArea + boxBArea - interArea);
    }
}
