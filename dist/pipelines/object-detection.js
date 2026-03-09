/**
 * edgeFlow.js - Object Detection Pipeline
 *
 * Detect objects in images with bounding boxes and class labels.
 */
import { BasePipeline, registerPipeline } from './base.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { ImagePreprocessor } from '../utils/preprocessor.js';
import { loadModelData } from '../utils/model-loader.js';
import { loadModelFromBuffer, runInference } from '../core/runtime.js';
// ============================================================================
// Default Model (YOLOS-tiny, quantized)
// ============================================================================
const DEFAULT_MODELS = {
    model: 'https://huggingface.co/Xenova/yolos-tiny/resolve/main/onnx/model_quantized.onnx',
};
// ============================================================================
// COCO Labels
// ============================================================================
export const COCO_LABELS = [
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
// ============================================================================
// Object Detection Pipeline
// ============================================================================
export class ObjectDetectionPipeline extends BasePipeline {
    preprocessor;
    onnxModel = null;
    labels;
    modelUrl;
    constructor(config, labels) {
        super(config ?? {
            task: 'object-detection',
            model: 'default',
        });
        this.labels = labels ?? COCO_LABELS;
        this.modelUrl = (config?.model && config.model !== 'default') ? config.model : DEFAULT_MODELS.model;
        this.preprocessor = new ImagePreprocessor({
            width: 640,
            height: 640,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            channelFormat: 'CHW',
        });
    }
    async initialize() {
        await super.initialize();
        if (!this.onnxModel) {
            const modelData = await loadModelData(this.modelUrl, { cache: this.config.cache ?? true });
            this.onnxModel = await loadModelFromBuffer(modelData);
        }
    }
    setLabels(labels) {
        this.labels = labels;
    }
    async run(input, options) {
        await this.initialize();
        const tensorInputs = await this.preprocess(input);
        const outputs = await this.runModelInference(tensorInputs);
        return this.postprocess(outputs, options);
    }
    async preprocess(input) {
        const inputs = Array.isArray(input) ? input : [input];
        if (inputs.length === 1) {
            const tensor = await this.preprocessor.process(inputs[0]);
            return [new EdgeFlowTensor(tensor.toFloat32Array(), [1, ...tensor.shape], 'float32')];
        }
        return [await this.preprocessor.processBatch(inputs)];
    }
    async runModelInference(inputs) {
        const outputs = await runInference(this.onnxModel, inputs);
        return outputs;
    }
    async postprocess(outputs, options) {
        const opts = options ?? {};
        const threshold = opts.threshold ?? 0.5;
        const topK = opts.topK ?? 100;
        const nms = opts.nms ?? true;
        const iouThreshold = opts.iouThreshold ?? 0.5;
        if (!outputs[0]) {
            return [];
        }
        const outputData = outputs[0].toFloat32Array();
        const shape = [...outputs[0].shape];
        const detections = this.parseDetections(outputData, shape, threshold);
        let filtered = nms ? this.nonMaxSuppression(detections, iouThreshold) : detections;
        filtered.sort((a, b) => b.score - a.score);
        filtered = filtered.slice(0, topK);
        return filtered;
    }
    parseDetections(data, shape, threshold) {
        const detections = [];
        const numBoxes = shape[1] ?? 0;
        const boxSize = shape[2] ?? 0;
        if (boxSize >= 5) {
            const numClasses = boxSize - 5;
            for (let i = 0; i < numBoxes; i++) {
                const offset = i * boxSize;
                const objectness = data[offset + 4] ?? 0;
                if (objectness < threshold)
                    continue;
                let maxClassScore = 0;
                let maxClassIdx = 0;
                for (let c = 0; c < numClasses; c++) {
                    const score = data[offset + 5 + c] ?? 0;
                    if (score > maxClassScore) {
                        maxClassScore = score;
                        maxClassIdx = c;
                    }
                }
                const confidence = objectness * maxClassScore;
                if (confidence < threshold)
                    continue;
                const x = data[offset] ?? 0;
                const y = data[offset + 1] ?? 0;
                const w = data[offset + 2] ?? 0;
                const h = data[offset + 3] ?? 0;
                detections.push({
                    label: this.labels[maxClassIdx] ?? `class_${maxClassIdx}`,
                    score: confidence,
                    classId: maxClassIdx,
                    box: {
                        x: Math.max(0, x - w / 2),
                        y: Math.max(0, y - h / 2),
                        width: w,
                        height: h,
                    },
                    boxNormalized: {
                        x: Math.max(0, x - w / 2),
                        y: Math.max(0, y - h / 2),
                        width: w,
                        height: h,
                    },
                });
            }
        }
        else if (boxSize === 4) {
            for (let i = 0; i < numBoxes; i++) {
                const offset = i * boxSize;
                const x1 = data[offset] ?? 0;
                const y1 = data[offset + 1] ?? 0;
                const x2 = data[offset + 2] ?? 0;
                const y2 = data[offset + 3] ?? 0;
                detections.push({
                    label: this.labels[0] ?? 'object',
                    score: 1.0,
                    classId: 0,
                    box: {
                        x: x1,
                        y: y1,
                        width: x2 - x1,
                        height: y2 - y1,
                    },
                    boxNormalized: {
                        x: x1,
                        y: y1,
                        width: x2 - x1,
                        height: y2 - y1,
                    },
                });
            }
        }
        return detections;
    }
    nonMaxSuppression(detections, iouThreshold) {
        if (detections.length === 0)
            return [];
        const sorted = [...detections].sort((a, b) => b.score - a.score);
        const selected = [];
        const active = new Array(sorted.length).fill(true);
        for (let i = 0; i < sorted.length; i++) {
            if (!active[i])
                continue;
            const current = sorted[i];
            selected.push(current);
            for (let j = i + 1; j < sorted.length; j++) {
                if (!active[j])
                    continue;
                const other = sorted[j];
                if (current.classId !== other.classId)
                    continue;
                const iou = this.computeIoU(current.box, other.box);
                if (iou > iouThreshold) {
                    active[j] = false;
                }
            }
        }
        return selected;
    }
    computeIoU(a, b) {
        const xOverlap = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
        const yOverlap = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
        const intersection = xOverlap * yOverlap;
        const aArea = a.width * a.height;
        const bArea = b.width * b.height;
        const union = aArea + bArea - intersection;
        return union > 0 ? intersection / union : 0;
    }
}
// ============================================================================
// Factory
// ============================================================================
export function createObjectDetectionPipeline(config, labels) {
    return new ObjectDetectionPipeline(config, labels);
}
registerPipeline('object-detection', (config) => new ObjectDetectionPipeline(config));
//# sourceMappingURL=object-detection.js.map