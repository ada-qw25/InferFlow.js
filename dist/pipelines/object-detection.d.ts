/**
 * edgeFlow.js - Object Detection Pipeline
 *
 * Detect objects in images with bounding boxes and class labels.
 */
import { BasePipeline, ObjectDetectionResult } from './base.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { type ImageInput } from '../utils/preprocessor.js';
export interface ObjectDetectionOptions extends PipelineOptions {
    threshold?: number;
    topK?: number;
    nms?: boolean;
    iouThreshold?: number;
}
export interface BoundingBox {
    x: number;
    y: number;
    width: number;
    height: number;
}
export interface Detection extends ObjectDetectionResult {
    classId: number;
    boxNormalized: BoundingBox;
}
export declare const COCO_LABELS: string[];
export declare class ObjectDetectionPipeline extends BasePipeline<ImageInput | ImageInput[], Detection[]> {
    private preprocessor;
    private onnxModel;
    private labels;
    private modelUrl;
    constructor(config?: PipelineConfig, labels?: string[]);
    initialize(): Promise<void>;
    setLabels(labels: string[]): void;
    run(input: ImageInput | ImageInput[], options?: ObjectDetectionOptions): Promise<Detection[]>;
    protected preprocess(input: ImageInput | ImageInput[]): Promise<EdgeFlowTensor[]>;
    private runModelInference;
    protected postprocess(outputs: EdgeFlowTensor[], options?: PipelineOptions): Promise<Detection[]>;
    private parseDetections;
    private nonMaxSuppression;
    private computeIoU;
}
export declare function createObjectDetectionPipeline(config?: PipelineConfig, labels?: string[]): ObjectDetectionPipeline;
//# sourceMappingURL=object-detection.d.ts.map