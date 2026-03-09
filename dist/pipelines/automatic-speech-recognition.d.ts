/**
 * edgeFlow.js - Automatic Speech Recognition Pipeline
 *
 * Transcribe audio to text using Whisper ONNX models (encoder + decoder).
 */
import { BasePipeline, PipelineResult } from './base.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { type AudioInput } from '../utils/preprocessor.js';
import { Tokenizer } from '../utils/tokenizer.js';
export interface ASROptions extends PipelineOptions {
    language?: string;
    task?: 'transcribe' | 'translate';
    returnTimestamps?: boolean | 'word' | 'chunk';
    maxDuration?: number;
    chunkDuration?: number;
    chunkOverlap?: number;
}
export interface WordTimestamp {
    word: string;
    start: number;
    end: number;
    confidence?: number;
}
export interface ChunkTimestamp {
    text: string;
    start: number;
    end: number;
}
export interface ASRResult extends PipelineResult {
    text: string;
    language?: string;
    words?: WordTimestamp[];
    chunks?: ChunkTimestamp[];
}
export declare class AutomaticSpeechRecognitionPipeline extends BasePipeline<AudioInput | AudioInput[], ASRResult | ASRResult[]> {
    private audioPreprocessor;
    private tokenizer;
    private encoderModel;
    private decoderModel;
    private encoderUrl;
    private decoderUrl;
    private tokenizerUrl;
    constructor(config?: PipelineConfig);
    initialize(): Promise<void>;
    setTokenizer(tokenizer: Tokenizer): void;
    run(input: AudioInput | AudioInput[], options?: PipelineOptions): Promise<ASRResult | ASRResult[]>;
    private transcribeSingle;
    private buildInitialTokens;
    private getLanguageToken;
    /**
     * Autoregressive decoder loop similar to text-generation.
     * Feeds encoder hidden states + growing token sequence to decoder.
     */
    private autoregressiveDecode;
    private extractTimestamps;
    processLongAudio(audio: AudioInput, options?: ASROptions): Promise<ASRResult>;
    protected preprocess(input: AudioInput | AudioInput[]): Promise<EdgeFlowTensor[]>;
    protected postprocess(outputs: EdgeFlowTensor[], options?: PipelineOptions): Promise<ASRResult | ASRResult[]>;
    private decodeOutput;
}
export declare function createASRPipeline(config?: PipelineConfig): AutomaticSpeechRecognitionPipeline;
//# sourceMappingURL=automatic-speech-recognition.d.ts.map