/**
 * edgeFlow.js - Feature Extraction Pipeline
 * 
 * Extract embeddings/features from text using sentence-transformer models.
 */

import {
  PipelineConfig,
  PipelineOptions,
  LoadedModel,
} from '../core/types.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { loadModelData } from '../utils/model-loader.js';
import { loadModelFromBuffer, runInferenceNamed } from '../core/runtime.js';
import {
  BasePipeline,
  FeatureExtractionResult,
  registerPipeline,
} from './base.js';

// ============================================================================
// Default Model (all-MiniLM-L6-v2, 384-dim sentence embeddings)
// ============================================================================

const DEFAULT_MODELS = {
  model: 'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx',
  tokenizer: 'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json',
};

const DEFAULT_EMBEDDING_DIM = 384;

// ============================================================================
// Feature Extraction Pipeline
// ============================================================================

export interface FeatureExtractionOptions extends PipelineOptions {
  pooling?: 'mean' | 'max' | 'cls' | 'none';
  normalize?: boolean;
  outputDim?: number;
}

export class FeatureExtractionPipeline extends BasePipeline<
  string | string[],
  FeatureExtractionResult | FeatureExtractionResult[]
> {
  private tokenizer: Tokenizer | null = null;
  private onnxModel: LoadedModel | null = null;
  private embeddingDim: number;
  private modelUrl: string;
  private tokenizerUrl: string;

  constructor(config: PipelineConfig, embeddingDim: number = DEFAULT_EMBEDDING_DIM) {
    super(config);
    this.embeddingDim = embeddingDim;
    this.modelUrl = config.model !== 'default' ? config.model : DEFAULT_MODELS.model;
    this.tokenizerUrl = DEFAULT_MODELS.tokenizer;
  }

  override async initialize(): Promise<void> {
    await super.initialize();

    if (!this.tokenizer) {
      this.tokenizer = await Tokenizer.fromUrl(this.tokenizerUrl);
    }

    if (!this.onnxModel) {
      const modelData = await loadModelData(this.modelUrl, { cache: this.config.cache ?? true });
      this.onnxModel = await loadModelFromBuffer(modelData);
    }
  }

  override async run(
    input: string | string[],
    options?: FeatureExtractionOptions
  ): Promise<FeatureExtractionResult | FeatureExtractionResult[]> {
    const isBatch = Array.isArray(input);
    const inputs = isBatch ? input : [input];
    
    await this.initialize();
    
    const startTime = performance.now();
    const results: FeatureExtractionResult[] = [];

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

    return isBatch ? results : results[0]!;
  }

  protected override async preprocess(input: string | string[]): Promise<EdgeFlowTensor[]> {
    const text = Array.isArray(input) ? input[0]! : input;
    
    const encoded = this.tokenizer!.encode(text, {
      maxLength: 128,
      padding: 'max_length',
      truncation: true,
    });

    const inputIds = new EdgeFlowTensor(
      BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))),
      [1, encoded.inputIds.length],
      'int64'
    );

    const attentionMask = new EdgeFlowTensor(
      BigInt64Array.from(encoded.attentionMask.map(m => BigInt(m))),
      [1, encoded.attentionMask.length],
      'int64'
    );

    const tokenTypeIds = new EdgeFlowTensor(
      BigInt64Array.from(encoded.inputIds.map(() => BigInt(0))),
      [1, encoded.inputIds.length],
      'int64'
    );

    return [inputIds, attentionMask, tokenTypeIds];
  }

  private async runInference(inputs: EdgeFlowTensor[]): Promise<EdgeFlowTensor[]> {
    const namedInputs = new Map<string, EdgeFlowTensor>();
    namedInputs.set('input_ids', inputs[0]!);
    namedInputs.set('attention_mask', inputs[1]!);
    namedInputs.set('token_type_ids', inputs[2]!);

    const outputs = await runInferenceNamed(this.onnxModel!, namedInputs);
    return outputs as EdgeFlowTensor[];
  }

  protected override async postprocess(
    outputs: EdgeFlowTensor[],
    options?: FeatureExtractionOptions
  ): Promise<FeatureExtractionResult> {
    const hiddenStates = outputs[0];
    if (!hiddenStates) {
      return { embeddings: [] };
    }

    const pooling = options?.pooling ?? 'mean';
    const normalize = options?.normalize ?? true;

    let embeddings: number[];

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

  private extractCLSEmbedding(hiddenStates: EdgeFlowTensor): number[] {
    const data = hiddenStates.toFloat32Array();
    const embeddingDim = hiddenStates.shape[2] ?? this.embeddingDim;
    return Array.from(data.slice(0, embeddingDim));
  }

  private meanPooling(hiddenStates: EdgeFlowTensor): number[] {
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

  private maxPooling(hiddenStates: EdgeFlowTensor): number[] {
    const data = hiddenStates.toFloat32Array();
    const seqLen = hiddenStates.shape[1] ?? 1;
    const embeddingDim = hiddenStates.shape[2] ?? this.embeddingDim;
    
    const result = new Array(embeddingDim).fill(-Infinity) as number[];
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

  private normalizeVector(vec: number[]): number[] {
    let norm = 0;
    for (const v of vec) {
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (norm === 0) return vec;
    return vec.map(v => v / norm);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createFeatureExtractionPipeline(
  config: Partial<PipelineConfig> = {}
): FeatureExtractionPipeline {
  return new FeatureExtractionPipeline({
    task: 'feature-extraction',
    model: config.model ?? 'default',
    runtime: config.runtime,
    cache: config.cache ?? true,
    quantization: config.quantization,
  });
}

registerPipeline('feature-extraction', (config) => new FeatureExtractionPipeline(config));
