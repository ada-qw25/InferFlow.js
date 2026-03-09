/**
 * edgeFlow.js - Question Answering Pipeline
 *
 * Extract answers from context given a question using real ONNX QA models.
 */
import { BasePipeline, registerPipeline } from './base.js';
import { EdgeFlowTensor, softmax } from '../core/tensor.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { loadModelData } from '../utils/model-loader.js';
import { loadModelFromBuffer, runInferenceNamed } from '../core/runtime.js';
// ============================================================================
// Default Model (DistilBERT fine-tuned on SQuAD)
// ============================================================================
const DEFAULT_MODELS = {
    model: 'https://huggingface.co/Xenova/distilbert-base-cased-distilled-squad/resolve/main/onnx/model_quantized.onnx',
    tokenizer: 'https://huggingface.co/Xenova/distilbert-base-cased-distilled-squad/resolve/main/tokenizer.json',
};
// ============================================================================
// Question Answering Pipeline
// ============================================================================
export class QuestionAnsweringPipeline extends BasePipeline {
    tokenizer = null;
    onnxModel = null;
    modelUrl;
    tokenizerUrl;
    constructor(config) {
        super(config ?? {
            task: 'question-answering',
            model: 'default',
        });
        this.modelUrl = (config?.model && config.model !== 'default') ? config.model : DEFAULT_MODELS.model;
        this.tokenizerUrl = DEFAULT_MODELS.tokenizer;
    }
    async initialize() {
        await super.initialize();
        if (!this.tokenizer) {
            this.tokenizer = await Tokenizer.fromUrl(this.tokenizerUrl);
        }
        if (!this.onnxModel) {
            const modelData = await loadModelData(this.modelUrl, { cache: this.config.cache ?? true });
            this.onnxModel = await loadModelFromBuffer(modelData);
        }
    }
    setTokenizer(tokenizer) {
        this.tokenizer = tokenizer;
    }
    async run(input, options) {
        await this.initialize();
        const inputs = Array.isArray(input) ? input : [input];
        const results = await Promise.all(inputs.map(i => this.answerQuestion(i, options ?? {})));
        return Array.isArray(input) ? results : results[0];
    }
    async answerQuestion(input, options) {
        const startTime = performance.now();
        const { question, context } = input;
        const maxAnswerLength = options.maxAnswerLength ?? 30;
        const encoded = this.tokenizer.encode(question, {
            textPair: context,
            addSpecialTokens: true,
            maxLength: 512,
            truncation: true,
            returnAttentionMask: true,
            returnTokenTypeIds: true,
        });
        const inputIds = new EdgeFlowTensor(BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))), [1, encoded.inputIds.length], 'int64');
        const attentionMask = new EdgeFlowTensor(BigInt64Array.from(encoded.attentionMask.map(m => BigInt(m))), [1, encoded.attentionMask.length], 'int64');
        const namedInputs = new Map();
        namedInputs.set('input_ids', inputIds);
        namedInputs.set('attention_mask', attentionMask);
        const outputs = await runInferenceNamed(this.onnxModel, namedInputs);
        if (outputs.length < 2) {
            return { answer: '', score: 0, start: 0, end: 0, processingTime: performance.now() - startTime };
        }
        const startLogits = outputs[0].toFloat32Array();
        const endLogits = outputs[1].toFloat32Array();
        const seqLen = startLogits.length;
        const startProbs = softmax(new EdgeFlowTensor(new Float32Array(startLogits), [seqLen], 'float32')).toFloat32Array();
        const endProbs = softmax(new EdgeFlowTensor(new Float32Array(endLogits), [seqLen], 'float32')).toFloat32Array();
        // Find best start/end token positions
        let bestStartIdx = 0;
        let bestEndIdx = 0;
        let bestScore = 0;
        for (let s = 0; s < seqLen; s++) {
            for (let e = s; e < Math.min(s + maxAnswerLength, seqLen); e++) {
                const score = (startProbs[s] ?? 0) * (endProbs[e] ?? 0);
                if (score > bestScore) {
                    bestScore = score;
                    bestStartIdx = s;
                    bestEndIdx = e;
                }
            }
        }
        // Decode the answer span back to text
        const answerTokenIds = encoded.inputIds.slice(bestStartIdx, bestEndIdx + 1);
        const answer = this.tokenizer.decode(answerTokenIds, true);
        // Map token positions back to character offsets in context
        const charStart = this.tokenOffsetToCharOffset(context, question, encoded.inputIds, bestStartIdx);
        const charEnd = this.tokenOffsetToCharOffset(context, question, encoded.inputIds, bestEndIdx) + 1;
        return {
            answer: answer || '',
            score: bestScore,
            start: charStart,
            end: charEnd,
            processingTime: performance.now() - startTime,
        };
    }
    tokenOffsetToCharOffset(context, _question, inputIds, tokenIdx) {
        // Approximate mapping: decode tokens up to this index and measure length
        // For a production implementation you'd use the tokenizer's offset mapping.
        const decoded = this.tokenizer.decode(inputIds.slice(0, tokenIdx + 1), true);
        const contextStart = context.indexOf(decoded.trim().split(' ').pop() ?? '');
        return contextStart >= 0 ? contextStart : 0;
    }
    async preprocess(input) {
        const qaInput = Array.isArray(input) ? input[0] : input;
        const encoded = this.tokenizer.encode(qaInput.question, {
            textPair: qaInput.context,
            addSpecialTokens: true,
            maxLength: 512,
            truncation: true,
            returnAttentionMask: true,
            returnTokenTypeIds: true,
        });
        return [
            new EdgeFlowTensor(BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))), [1, encoded.inputIds.length], 'int64'),
            new EdgeFlowTensor(BigInt64Array.from(encoded.attentionMask.map(m => BigInt(m))), [1, encoded.attentionMask.length], 'int64'),
        ];
    }
    async postprocess(outputs, _options) {
        if (outputs.length < 2) {
            return { answer: '', score: 0, start: 0, end: 0 };
        }
        const startLogits = outputs[0].toFloat32Array();
        const endLogits = outputs[1].toFloat32Array();
        const seqLen = startLogits.length;
        const startProbs = softmax(new EdgeFlowTensor(startLogits, [seqLen], 'float32')).toFloat32Array();
        const endProbs = softmax(new EdgeFlowTensor(endLogits, [seqLen], 'float32')).toFloat32Array();
        let bestStart = 0;
        let bestEnd = 0;
        let bestScore = 0;
        for (let start = 0; start < seqLen; start++) {
            for (let end = start; end < Math.min(start + 30, seqLen); end++) {
                const score = (startProbs[start] ?? 0) * (endProbs[end] ?? 0);
                if (score > bestScore) {
                    bestScore = score;
                    bestStart = start;
                    bestEnd = end;
                }
            }
        }
        return {
            answer: '',
            score: bestScore,
            start: bestStart,
            end: bestEnd,
        };
    }
}
// ============================================================================
// Factory
// ============================================================================
export function createQuestionAnsweringPipeline(config) {
    return new QuestionAnsweringPipeline(config);
}
registerPipeline('question-answering', (config) => new QuestionAnsweringPipeline(config));
//# sourceMappingURL=question-answering.js.map