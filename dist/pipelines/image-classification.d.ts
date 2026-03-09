/**
 * edgeFlow.js - Image Classification Pipeline
 *
 * Classify images into categories using vision models.
 */
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { BasePipeline, ImageClassificationResult } from './base.js';
export interface ImageClassificationOptions extends PipelineOptions {
    returnAllScores?: boolean;
    labels?: string[];
    topK?: number;
}
export type ImageInput = HTMLImageElement | HTMLCanvasElement | ImageBitmap | ImageData | string;
export declare class ImageClassificationPipeline extends BasePipeline<ImageInput | ImageInput[], ImageClassificationResult | ImageClassificationResult[]> {
    private preprocessor;
    private onnxModel;
    private labels;
    private modelUrl;
    constructor(config: PipelineConfig, labels?: string[], _numClasses?: number);
    initialize(): Promise<void>;
    setLabels(labels: string[]): void;
    run(input: ImageInput | ImageInput[], options?: ImageClassificationOptions): Promise<ImageClassificationResult | ImageClassificationResult[]>;
    protected preprocess(input: ImageInput | ImageInput[]): Promise<EdgeFlowTensor[]>;
    private runModelInference;
    protected postprocess(outputs: EdgeFlowTensor[], options?: ImageClassificationOptions): Promise<ImageClassificationResult>;
}
export declare function createImageClassificationPipeline(config?: Partial<PipelineConfig>, labels?: string[]): ImageClassificationPipeline;
//# sourceMappingURL=image-classification.d.ts.map