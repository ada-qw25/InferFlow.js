/**
 * edgeFlow.js - Image Classification Pipeline
 * 
 * Classify images into categories using vision models.
 */

import {
  PipelineConfig,
  PipelineOptions,
  LoadedModel,
} from '../core/types.js';
import { EdgeFlowTensor, softmax } from '../core/tensor.js';
import { ImagePreprocessor, createImagePreprocessor } from '../utils/preprocessor.js';
import { loadModelData } from '../utils/model-loader.js';
import { loadModelFromBuffer, runInference } from '../core/runtime.js';
import {
  BasePipeline,
  ImageClassificationResult,
  registerPipeline,
  IMAGENET_LABELS,
} from './base.js';

// ============================================================================
// Default Model (MobileViT-small, quantized)
// ============================================================================

const DEFAULT_MODELS = {
  model: 'https://huggingface.co/Xenova/mobilevit-small/resolve/main/onnx/model_quantized.onnx',
};


// ============================================================================
// Image Classification Pipeline
// ============================================================================

export interface ImageClassificationOptions extends PipelineOptions {
  returnAllScores?: boolean;
  labels?: string[];
  topK?: number;
}

export type ImageInput =
  | HTMLImageElement
  | HTMLCanvasElement
  | ImageBitmap
  | ImageData
  | string;

export class ImageClassificationPipeline extends BasePipeline<
  ImageInput | ImageInput[],
  ImageClassificationResult | ImageClassificationResult[]
> {
  private preprocessor: ImagePreprocessor | null = null;
  private onnxModel: LoadedModel | null = null;
  private labels: string[];
  private modelUrl: string;

  constructor(
    config: PipelineConfig,
    labels?: string[],
    _numClasses: number = 1000
  ) {
    super(config);
    this.labels = labels ?? IMAGENET_LABELS;
    this.modelUrl = config.model !== 'default' ? config.model : DEFAULT_MODELS.model;
  }

  override async initialize(): Promise<void> {
    await super.initialize();

    if (!this.preprocessor) {
      this.preprocessor = createImagePreprocessor('imagenet');
    }

    if (!this.onnxModel) {
      const modelData = await loadModelData(this.modelUrl, { cache: this.config.cache ?? true });
      this.onnxModel = await loadModelFromBuffer(modelData);
    }
  }

  setLabels(labels: string[]): void {
    this.labels = labels;
  }

  override async run(
    input: ImageInput | ImageInput[],
    options?: ImageClassificationOptions
  ): Promise<ImageClassificationResult | ImageClassificationResult[]> {
    const isBatch = Array.isArray(input);
    const inputs = isBatch ? input : [input];

    await this.initialize();

    const startTime = performance.now();
    const results: ImageClassificationResult[] = [];

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

    return isBatch ? results : results[0]!;
  }

  protected override async preprocess(input: ImageInput | ImageInput[]): Promise<EdgeFlowTensor[]> {
    const image = Array.isArray(input) ? input[0]! : input;
    const tensor = await this.preprocessor!.process(image);

    if (tensor.shape.length === 3) {
      return [tensor.reshape([1, ...tensor.shape])];
    }
    return [tensor];
  }

  private async runModelInference(inputs: EdgeFlowTensor[]): Promise<EdgeFlowTensor[]> {
    const outputs = await runInference(this.onnxModel!, inputs);
    return outputs as EdgeFlowTensor[];
  }

  protected override async postprocess(
    outputs: EdgeFlowTensor[],
    options?: ImageClassificationOptions
  ): Promise<ImageClassificationResult> {
    const logits = outputs[0];
    if (!logits) {
      return { label: 'unknown', score: 0 };
    }

    const probs = softmax(logits, -1) as EdgeFlowTensor;
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

export function createImageClassificationPipeline(
  config: Partial<PipelineConfig> = {},
  labels?: string[]
): ImageClassificationPipeline {
  return new ImageClassificationPipeline(
    {
      task: 'image-classification',
      model: config.model ?? 'default',
      runtime: config.runtime,
      cache: config.cache ?? true,
      quantization: config.quantization,
    },
    labels
  );
}

registerPipeline('image-classification', (config) => new ImageClassificationPipeline(config));
