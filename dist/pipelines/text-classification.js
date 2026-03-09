/**
 * edgeFlow.js - Text Classification Pipeline
 *
 * High-level API for text classification tasks including
 * sentiment analysis, topic classification, etc.
 */
import { EdgeFlowTensor, softmax } from '../core/tensor.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { loadModelData } from '../utils/model-loader.js';
import { loadModelFromBuffer, runInferenceNamed } from '../core/runtime.js';
import { BasePipeline, registerPipeline, SENTIMENT_LABELS, } from './base.js';
// ============================================================================
// Default Model (DistilBERT fine-tuned on SST-2)
// ============================================================================
const DEFAULT_MODELS = {
    model: 'https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model_quantized.onnx',
    tokenizer: 'https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/tokenizer.json',
};
const DEFAULT_SST2_LABELS = ['NEGATIVE', 'POSITIVE'];
export class TextClassificationPipeline extends BasePipeline {
    tokenizer = null;
    onnxModel = null;
    labels;
    modelUrl;
    tokenizerUrl;
    constructor(config, labels) {
        super(config);
        this.labels = labels ?? DEFAULT_SST2_LABELS;
        this.modelUrl = config.model !== 'default' ? config.model : DEFAULT_MODELS.model;
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
    setLabels(labels) {
        this.labels = labels;
    }
    async run(input, options) {
        const isBatch = Array.isArray(input);
        const inputs = isBatch ? input : [input];
        await this.initialize();
        const startTime = performance.now();
        const results = [];
        for (const text of inputs) {
            const tensorInputs = await this.preprocess(text);
            const outputs = await this.runInference(tensorInputs);
            const result = await this.postprocess(outputs, options);
            results.push(result);
        }
        const processingTime = performance.now() - startTime;
        for (const result of results) {
            result.processingTime = processingTime / results.length;
        }
        return isBatch ? results : results[0];
    }
    async preprocess(input) {
        const text = Array.isArray(input) ? input[0] : input;
        const encoded = this.tokenizer.encode(text, {
            maxLength: 128,
            padding: 'max_length',
            truncation: true,
        });
        const inputIds = new EdgeFlowTensor(BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))), [1, encoded.inputIds.length], 'int64');
        const attentionMask = new EdgeFlowTensor(BigInt64Array.from(encoded.attentionMask.map(m => BigInt(m))), [1, encoded.attentionMask.length], 'int64');
        return [inputIds, attentionMask];
    }
    async runInference(inputs) {
        const namedInputs = new Map();
        namedInputs.set('input_ids', inputs[0]);
        namedInputs.set('attention_mask', inputs[1]);
        const outputs = await runInferenceNamed(this.onnxModel, namedInputs);
        return outputs;
    }
    async postprocess(outputs, options) {
        const logits = outputs[0];
        if (!logits) {
            return { label: 'unknown', score: 0 };
        }
        const probs = softmax(logits, -1);
        const probsArray = probs.toFloat32Array();
        let maxIdx = 0;
        let maxScore = probsArray[0] ?? 0;
        for (let i = 1; i < probsArray.length; i++) {
            if ((probsArray[i] ?? 0) > maxScore) {
                maxScore = probsArray[i] ?? 0;
                maxIdx = i;
            }
        }
        const label = options?.labels?.[maxIdx] ?? this.labels[maxIdx] ?? `class_${maxIdx}`;
        return {
            label,
            score: maxScore,
        };
    }
}
// ============================================================================
// Sentiment Analysis Pipeline
// ============================================================================
export class SentimentAnalysisPipeline extends TextClassificationPipeline {
    constructor(config) {
        super(config, SENTIMENT_LABELS);
    }
    async analyze(text, options) {
        return this.run(text, options);
    }
}
// ============================================================================
// Factory Functions
// ============================================================================
export function createTextClassificationPipeline(config = {}) {
    return new TextClassificationPipeline({
        task: 'text-classification',
        model: config.model ?? 'default',
        runtime: config.runtime,
        cache: config.cache ?? true,
        quantization: config.quantization,
    });
}
export function createSentimentAnalysisPipeline(config = {}) {
    return new SentimentAnalysisPipeline({
        task: 'sentiment-analysis',
        model: config.model ?? 'default',
        runtime: config.runtime,
        cache: config.cache ?? true,
        quantization: config.quantization,
    });
}
registerPipeline('text-classification', (config) => new TextClassificationPipeline(config));
registerPipeline('sentiment-analysis', (config) => new SentimentAnalysisPipeline(config));
//# sourceMappingURL=text-classification.js.map