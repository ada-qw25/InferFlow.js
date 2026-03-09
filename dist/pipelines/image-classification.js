/**
 * edgeFlow.js - Image Classification Pipeline
 *
 * Classify images into categories using vision models.
 */
import { softmax } from '../core/tensor.js';
import { createImagePreprocessor } from '../utils/preprocessor.js';
import { loadModelData } from '../utils/model-loader.js';
import { loadModelFromBuffer, runInference } from '../core/runtime.js';
import { BasePipeline, registerPipeline, IMAGENET_LABELS, } from './base.js';
// ============================================================================
// Default Model (MobileViT-small, quantized)
// ============================================================================
const DEFAULT_MODELS = {
    model: 'https://huggingface.co/Xenova/mobilevit-small/resolve/main/onnx/model_quantized.onnx',
};
export class ImageClassificationPipeline extends BasePipeline {
    preprocessor = null;
    onnxModel = null;
    labels;
    modelUrl;
    constructor(config, labels, _numClasses = 1000) {
        super(config);
        this.labels = labels ?? IMAGENET_LABELS;
        this.modelUrl = config.model !== 'default' ? config.model : DEFAULT_MODELS.model;
    }
    async initialize() {
        await super.initialize();
        if (!this.preprocessor) {
            this.preprocessor = createImagePreprocessor('imagenet');
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
        for (const image of inputs) {
            const tensorInputs = await this.preprocess(image);
            const outputs = await this.runModelInference(tensorInputs);
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
        const image = Array.isArray(input) ? input[0] : input;
        const tensor = await this.preprocessor.process(image);
        if (tensor.shape.length === 3) {
            return [tensor.reshape([1, ...tensor.shape])];
        }
        return [tensor];
    }
    async runModelInference(inputs) {
        const outputs = await runInference(this.onnxModel, inputs);
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
        return { label, score: maxScore };
    }
}
// ============================================================================
// Factory Function
// ============================================================================
export function createImageClassificationPipeline(config = {}, labels) {
    return new ImageClassificationPipeline({
        task: 'image-classification',
        model: config.model ?? 'default',
        runtime: config.runtime,
        cache: config.cache ?? true,
        quantization: config.quantization,
    }, labels);
}
registerPipeline('image-classification', (config) => new ImageClassificationPipeline(config));
//# sourceMappingURL=image-classification.js.map