/**
 * edgeFlow.js - Feature Extraction Pipeline
 *
 * Extract embeddings/features from text using sentence-transformer models.
 */
import { EdgeFlowTensor } from '../core/tensor.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { loadModelData } from '../utils/model-loader.js';
import { loadModelFromBuffer, runInferenceNamed } from '../core/runtime.js';
import { BasePipeline, registerPipeline, } from './base.js';
// ============================================================================
// Default Model (all-MiniLM-L6-v2, 384-dim sentence embeddings)
// ============================================================================
const DEFAULT_MODELS = {
    model: 'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx',
    tokenizer: 'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json',
};
const DEFAULT_EMBEDDING_DIM = 384;
export class FeatureExtractionPipeline extends BasePipeline {
    tokenizer = null;
    onnxModel = null;
    embeddingDim;
    modelUrl;
    tokenizerUrl;
    constructor(config, embeddingDim = DEFAULT_EMBEDDING_DIM) {
        super(config);
        this.embeddingDim = embeddingDim;
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
        const tokenTypeIds = new EdgeFlowTensor(BigInt64Array.from(encoded.inputIds.map(() => BigInt(0))), [1, encoded.inputIds.length], 'int64');
        return [inputIds, attentionMask, tokenTypeIds];
    }
    async runInference(inputs) {
        const namedInputs = new Map();
        namedInputs.set('input_ids', inputs[0]);
        namedInputs.set('attention_mask', inputs[1]);
        namedInputs.set('token_type_ids', inputs[2]);
        const outputs = await runInferenceNamed(this.onnxModel, namedInputs);
        return outputs;
    }
    async postprocess(outputs, options) {
        const hiddenStates = outputs[0];
        if (!hiddenStates) {
            return { embeddings: [] };
        }
        const pooling = options?.pooling ?? 'mean';
        const normalize = options?.normalize ?? true;
        let embeddings;
        switch (pooling) {
            case 'cls':
                embeddings = this.extractCLSEmbedding(hiddenStates);
                break;
            case 'max':
                embeddings = this.maxPooling(hiddenStates);
                break;
            case 'none':
                embeddings = hiddenStates.toArray();
                break;
            case 'mean':
            default:
                embeddings = this.meanPooling(hiddenStates);
                break;
        }
        if (normalize) {
            embeddings = this.normalizeVector(embeddings);
        }
        if (options?.outputDim && options.outputDim < embeddings.length) {
            embeddings = embeddings.slice(0, options.outputDim);
        }
        return { embeddings };
    }
    extractCLSEmbedding(hiddenStates) {
        const data = hiddenStates.toFloat32Array();
        const embeddingDim = hiddenStates.shape[2] ?? this.embeddingDim;
        return Array.from(data.slice(0, embeddingDim));
    }
    meanPooling(hiddenStates) {
        const data = hiddenStates.toFloat32Array();
        const seqLen = hiddenStates.shape[1] ?? 1;
        const embeddingDim = hiddenStates.shape[2] ?? this.embeddingDim;
        const result = new Float32Array(embeddingDim);
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < embeddingDim; j++) {
                result[j] = (result[j] ?? 0) + (data[i * embeddingDim + j] ?? 0) / seqLen;
            }
        }
        return Array.from(result);
    }
    maxPooling(hiddenStates) {
        const data = hiddenStates.toFloat32Array();
        const seqLen = hiddenStates.shape[1] ?? 1;
        const embeddingDim = hiddenStates.shape[2] ?? this.embeddingDim;
        const result = new Array(embeddingDim).fill(-Infinity);
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < embeddingDim; j++) {
                const val = data[i * embeddingDim + j] ?? 0;
                if (val > (result[j] ?? -Infinity)) {
                    result[j] = val;
                }
            }
        }
        return result;
    }
    normalizeVector(vec) {
        let norm = 0;
        for (const v of vec) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        if (norm === 0)
            return vec;
        return vec.map(v => v / norm);
    }
}
// ============================================================================
// Factory Function
// ============================================================================
export function createFeatureExtractionPipeline(config = {}) {
    return new FeatureExtractionPipeline({
        task: 'feature-extraction',
        model: config.model ?? 'default',
        runtime: config.runtime,
        cache: config.cache ?? true,
        quantization: config.quantization,
    });
}
registerPipeline('feature-extraction', (config) => new FeatureExtractionPipeline(config));
//# sourceMappingURL=feature-extraction.js.map