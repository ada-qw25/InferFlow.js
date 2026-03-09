/**
 * edgeFlow.js - Question Answering Pipeline
 *
 * Extract answers from context given a question using real ONNX QA models.
 */
import { BasePipeline, PipelineResult } from './base.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { Tokenizer } from '../utils/tokenizer.js';
export interface QAInput {
    question: string;
    context: string;
}
export interface QuestionAnsweringOptions extends PipelineOptions {
    maxAnswerLength?: number;
    maxQuestionLength?: number;
    topK?: number;
    threshold?: number;
    handleImpossible?: boolean;
}
export interface QuestionAnsweringResult extends PipelineResult {
    answer: string;
    score: number;
    start: number;
    end: number;
}
export declare class QuestionAnsweringPipeline extends BasePipeline<QAInput | QAInput[], QuestionAnsweringResult | QuestionAnsweringResult[]> {
    private tokenizer;
    private onnxModel;
    private modelUrl;
    private tokenizerUrl;
    constructor(config?: PipelineConfig);
    initialize(): Promise<void>;
    setTokenizer(tokenizer: Tokenizer): void;
    run(input: QAInput | QAInput[], options?: QuestionAnsweringOptions): Promise<QuestionAnsweringResult | QuestionAnsweringResult[]>;
    private answerQuestion;
    private tokenOffsetToCharOffset;
    protected preprocess(input: QAInput | QAInput[]): Promise<EdgeFlowTensor[]>;
    protected postprocess(outputs: EdgeFlowTensor[], _options?: PipelineOptions): Promise<QuestionAnsweringResult | QuestionAnsweringResult[]>;
}
export declare function createQuestionAnsweringPipeline(config?: PipelineConfig): QuestionAnsweringPipeline;
//# sourceMappingURL=question-answering.d.ts.map