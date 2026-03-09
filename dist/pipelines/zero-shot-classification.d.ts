/**
 * edgeFlow.js - Zero-shot Classification Pipeline
 *
 * Classify text into any set of labels without fine-tuning,
 * using a real NLI (Natural Language Inference) model.
 */
import { BasePipeline, PipelineResult } from './base.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { Tokenizer } from '../utils/tokenizer.js';
export interface ZeroShotClassificationOptions extends PipelineOptions {
    multiLabel?: boolean;
    hypothesisTemplate?: string;
}
export interface ZeroShotClassificationResult extends PipelineResult {
    sequence: string;
    labels: string[];
    scores: number[];
}
export interface ZeroShotInput {
    text: string | string[];
    candidateLabels: string[];
}
export declare class ZeroShotClassificationPipeline extends BasePipeline<ZeroShotInput, ZeroShotClassificationResult | ZeroShotClassificationResult[]> {
    private tokenizer;
    private onnxModel;
    private hypothesisTemplate;
    private modelUrl;
    private tokenizerUrl;
    constructor(config?: PipelineConfig);
    initialize(): Promise<void>;
    setTokenizer(tokenizer: Tokenizer): void;
    classify(text: string | string[], candidateLabels: string[], options?: ZeroShotClassificationOptions): Promise<ZeroShotClassificationResult | ZeroShotClassificationResult[]>;
    run(input: ZeroShotInput, options?: PipelineOptions): Promise<ZeroShotClassificationResult | ZeroShotClassificationResult[]>;
    private classifySingle;
    /**
     * Score a single hypothesis using the real NLI ONNX model.
     * Returns the entailment logit.
     */
    private scoreHypothesis;
    protected preprocess(input: ZeroShotInput): Promise<EdgeFlowTensor[]>;
    protected postprocess(_outputs: EdgeFlowTensor[], _options?: PipelineOptions): Promise<ZeroShotClassificationResult | ZeroShotClassificationResult[]>;
}
export declare function createZeroShotClassificationPipeline(config?: PipelineConfig): ZeroShotClassificationPipeline;
//# sourceMappingURL=zero-shot-classification.d.ts.map