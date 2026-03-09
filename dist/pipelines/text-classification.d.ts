/**
 * edgeFlow.js - Text Classification Pipeline
 *
 * High-level API for text classification tasks including
 * sentiment analysis, topic classification, etc.
 */
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { BasePipeline, TextClassificationResult } from './base.js';
export interface TextClassificationOptions extends PipelineOptions {
    returnAllScores?: boolean;
    labels?: string[];
    topK?: number;
}
export declare class TextClassificationPipeline extends BasePipeline<string | string[], TextClassificationResult | TextClassificationResult[]> {
    private tokenizer;
    private onnxModel;
    private labels;
    private modelUrl;
    private tokenizerUrl;
    constructor(config: PipelineConfig, labels?: string[]);
    initialize(): Promise<void>;
    setLabels(labels: string[]): void;
    run(input: string | string[], options?: TextClassificationOptions): Promise<TextClassificationResult | TextClassificationResult[]>;
    protected preprocess(input: string | string[]): Promise<EdgeFlowTensor[]>;
    private runInference;
    protected postprocess(outputs: EdgeFlowTensor[], options?: TextClassificationOptions): Promise<TextClassificationResult>;
}
export declare class SentimentAnalysisPipeline extends TextClassificationPipeline {
    constructor(config: PipelineConfig);
    analyze(text: string | string[], options?: TextClassificationOptions): Promise<TextClassificationResult | TextClassificationResult[]>;
}
export declare function createTextClassificationPipeline(config?: Partial<PipelineConfig>): TextClassificationPipeline;
export declare function createSentimentAnalysisPipeline(config?: Partial<PipelineConfig>): SentimentAnalysisPipeline;
//# sourceMappingURL=text-classification.d.ts.map