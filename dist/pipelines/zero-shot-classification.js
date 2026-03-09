/**
 * edgeFlow.js - Zero-shot Classification Pipeline
 *
 * Classify text into any set of labels without fine-tuning,
 * using a real NLI (Natural Language Inference) model.
 */
import { BasePipeline, registerPipeline } from './base.js';
import { EdgeFlowTensor, softmax } from '../core/tensor.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { loadModelData } from '../utils/model-loader.js';
import { loadModelFromBuffer, runInferenceNamed } from '../core/runtime.js';
// ============================================================================
// Default Model (DistilBART fine-tuned on MNLI)
// ============================================================================
const DEFAULT_MODELS = {
    model: 'https://huggingface.co/Xenova/nli-deberta-v3-small/resolve/main/onnx/model_quantized.onnx',
    tokenizer: 'https://huggingface.co/Xenova/nli-deberta-v3-small/resolve/main/tokenizer.json',
};
// NLI output indices: [contradiction, neutral, entailment]
const ENTAILMENT_IDX = 2;
// ============================================================================
// Zero-shot Classification Pipeline
// ============================================================================
export class ZeroShotClassificationPipeline extends BasePipeline {
    tokenizer = null;
    onnxModel = null;
    hypothesisTemplate = 'This text is about {label}.';
    modelUrl;
    tokenizerUrl;
    constructor(config) {
        super(config ?? {
            task: 'zero-shot-classification',
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
    async classify(text, candidateLabels, options) {
        return this.run({ text, candidateLabels }, options);
    }
    async run(input, options) {
        await this.initialize();
        const { text, candidateLabels } = input;
        const opts = options ?? {};
        const texts = Array.isArray(text) ? text : [text];
        const template = opts.hypothesisTemplate ?? this.hypothesisTemplate;
        const multiLabel = opts.multiLabel ?? false;
        const results = await Promise.all(texts.map(t => this.classifySingle(t, candidateLabels, template, multiLabel)));
        return Array.isArray(text) ? results : results[0];
    }
    async classifySingle(text, candidateLabels, template, multiLabel) {
        const startTime = performance.now();
        const hypotheses = candidateLabels.map(label => template.replace('{label}', label));
        const scores = [];
        for (const hypothesis of hypotheses) {
            const score = await this.scoreHypothesis(text, hypothesis);
            scores.push(score);
        }
        let normalizedScores;
        if (multiLabel) {
            normalizedScores = scores.map(s => 1 / (1 + Math.exp(-s)));
        }
        else {
            const tensor = new EdgeFlowTensor(new Float32Array(scores), [scores.length], 'float32');
            normalizedScores = Array.from(softmax(tensor).toFloat32Array());
        }
        const indexed = candidateLabels.map((label, i) => ({
            label,
            score: normalizedScores[i] ?? 0,
        }));
        indexed.sort((a, b) => b.score - a.score);
        return {
            sequence: text,
            labels: indexed.map(i => i.label),
            scores: indexed.map(i => i.score),
            processingTime: performance.now() - startTime,
        };
    }
    /**
     * Score a single hypothesis using the real NLI ONNX model.
     * Returns the entailment logit.
     */
    async scoreHypothesis(premise, hypothesis) {
        const encoded = this.tokenizer.encode(premise, {
            textPair: hypothesis,
            addSpecialTokens: true,
            maxLength: 512,
            truncation: true,
            returnAttentionMask: true,
        });
        const inputIds = new EdgeFlowTensor(BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))), [1, encoded.inputIds.length], 'int64');
        const attentionMask = new EdgeFlowTensor(BigInt64Array.from(encoded.attentionMask.map(m => BigInt(m))), [1, encoded.attentionMask.length], 'int64');
        const namedInputs = new Map();
        namedInputs.set('input_ids', inputIds);
        namedInputs.set('attention_mask', attentionMask);
        const outputs = await runInferenceNamed(this.onnxModel, namedInputs);
        const logits = outputs[0].toFloat32Array();
        // Return entailment logit (index 2 in [contradiction, neutral, entailment])
        return logits[ENTAILMENT_IDX] ?? 0;
    }
    async preprocess(input) {
        const { text, candidateLabels } = input;
        const firstText = Array.isArray(text) ? text[0] ?? '' : text;
        const firstLabel = candidateLabels[0] ?? '';
        const encoded = this.tokenizer.encode(firstText, {
            textPair: this.hypothesisTemplate.replace('{label}', firstLabel),
            addSpecialTokens: true,
            maxLength: 512,
        });
        return [new EdgeFlowTensor(BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))), [1, encoded.inputIds.length], 'int64')];
    }
    async postprocess(_outputs, _options) {
        return {
            sequence: '',
            labels: [],
            scores: [],
        };
    }
}
// ============================================================================
// Factory
// ============================================================================
export function createZeroShotClassificationPipeline(config) {
    return new ZeroShotClassificationPipeline(config);
}
registerPipeline('zero-shot-classification', (config) => new ZeroShotClassificationPipeline(config));
//# sourceMappingURL=zero-shot-classification.js.map