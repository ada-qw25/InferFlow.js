/**
 * edgeFlow.js - transformers.js Adapter Backend
 *
 * Wraps transformers.js (by Hugging Face) as an inference backend, giving
 * users access to 1000+ HuggingFace models while adding edgeFlow.js's
 * orchestration layer (scheduling, caching, memory management, workers).
 *
 * @example
 * ```typescript
 * import { useTransformersBackend } from 'edgeflowjs';
 * import { pipeline as tfPipeline } from '@xenova/transformers';
 *
 * // Register the adapter
 * useTransformersBackend();
 *
 * // Now use edgeFlow.js pipeline API — inference delegates to transformers.js
 * const classifier = await pipeline('text-classification', {
 *   model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
 * });
 *
 * // edgeFlow.js handles scheduling, batching, memory, caching
 * const results = await classifier.runBatch(thousandsOfTexts);
 * ```
 */

import {
  Runtime,
  RuntimeType,
  RuntimeCapabilities,
  LoadedModel,
  ModelLoadOptions,
  ModelMetadata,
  Tensor,
  EdgeFlowError,
  ErrorCodes,
} from '../core/types.js';
import { LoadedModelImpl } from '../core/runtime.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { getMemoryManager } from '../core/memory.js';
import { registerRuntime } from '../core/runtime.js';

// ---------------------------------------------------------------------------
// Types for the transformers.js interop
// ---------------------------------------------------------------------------

/**
 * Minimal interface for a transformers.js pipeline instance.
 * We avoid importing @xenova/transformers directly so edgeFlow.js
 * does not add it as a hard dependency.
 */
interface TransformersPipelineInstance {
  (input: unknown, options?: unknown): Promise<unknown>;
  dispose?: () => Promise<void> | void;
}

/**
 * A factory that creates a transformers.js pipeline.
 * Users pass this so we don't hard-depend on the library.
 */
export type TransformersPipelineFactory = (
  task: string,
  model?: string,
  options?: Record<string, unknown>,
) => Promise<TransformersPipelineInstance>;

/**
 * Options for configuring the transformers.js adapter.
 */
export interface TransformersAdapterOptions {
  /** The pipeline factory from transformers.js (e.g. the `pipeline` function) */
  pipelineFactory: TransformersPipelineFactory;
  /** Default device ('webgpu' | 'wasm' | 'cpu') — passed to transformers.js */
  device?: string;
  /** Default dtype ('fp32' | 'fp16' | 'q8' | 'q4') */
  dtype?: string;
  /** Cache directory (browser IndexedDB path) */
  cacheDir?: string;
}

// ---------------------------------------------------------------------------
// Session store: maps model IDs to transformers.js pipeline instances
// ---------------------------------------------------------------------------

const sessionStore = new Map<string, {
  instance: TransformersPipelineInstance;
  task: string;
  model: string;
}>();

let adapterOptions: TransformersAdapterOptions | null = null;

// ---------------------------------------------------------------------------
// Runtime implementation
// ---------------------------------------------------------------------------

export class TransformersAdapterRuntime implements Runtime {
  readonly name: RuntimeType = 'wasm'; // registers under the wasm slot

  get capabilities(): RuntimeCapabilities {
    return {
      concurrency: true,
      quantization: true,
      float16: true,
      dynamicShapes: true,
      maxBatchSize: 128,
      availableMemory: 1024 * 1024 * 1024,
    };
  }

  async isAvailable(): Promise<boolean> {
    return adapterOptions?.pipelineFactory != null;
  }

  async initialize(): Promise<void> {
    if (!adapterOptions?.pipelineFactory) {
      throw new EdgeFlowError(
        'TransformersAdapterRuntime requires a pipelineFactory. ' +
        'Call useTransformersBackend({ pipelineFactory }) first.',
        ErrorCodes.RUNTIME_INIT_FAILED,
      );
    }
  }

  async loadModel(
    modelData: ArrayBuffer,
    options: ModelLoadOptions = {},
  ): Promise<LoadedModel> {
    // modelData is unused — transformers.js downloads its own models.
    // Instead the model identifier comes via metadata.name or the URL.
    const modelName = options.metadata?.name ?? 'default';

    const metadata: ModelMetadata = {
      name: modelName,
      version: '1.0.0',
      inputs: [{ name: 'input', dtype: 'float32', shape: [-1] }],
      outputs: [{ name: 'output', dtype: 'float32', shape: [-1] }],
      sizeBytes: modelData.byteLength || 0,
      quantization: options.quantization ?? 'float32',
      format: 'onnx',
    };

    const modelId = `tjs_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;

    const model = new LoadedModelImpl(metadata, this.name, () => {
      const session = sessionStore.get(modelId);
      if (session?.instance.dispose) {
        session.instance.dispose();
      }
      sessionStore.delete(modelId);
    });

    getMemoryManager().trackModel(model, () => model.dispose());
    return model;
  }

  /**
   * Load a transformers.js pipeline by task + model name
   * (called by the higher-level adapter pipeline, not via the
   * standard loadModel path).
   */
  async loadPipeline(
    task: string,
    model: string,
    pipelineOptions?: Record<string, unknown>,
  ): Promise<string> {
    if (!adapterOptions?.pipelineFactory) {
      throw new EdgeFlowError(
        'Adapter not initialised',
        ErrorCodes.RUNTIME_NOT_INITIALIZED,
      );
    }

    const opts: Record<string, unknown> = { ...pipelineOptions };
    if (adapterOptions.device) opts['device'] = adapterOptions.device;
    if (adapterOptions.dtype) opts['dtype'] = adapterOptions.dtype;

    const instance = await adapterOptions.pipelineFactory(task, model, opts);
    const modelId = `tjs_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
    sessionStore.set(modelId, { instance, task, model });

    return modelId;
  }

  /**
   * Run inference by passing the raw input to the transformers.js pipeline.
   * The result is returned as a single EdgeFlowTensor wrapping the JSON-encoded output
   * (since transformers.js returns task-specific objects, not raw tensors).
   */
  async run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]> {
    const session = sessionStore.get(model.id);
    if (!session) {
      throw new EdgeFlowError(
        `No transformers.js session for model ${model.id}`,
        ErrorCodes.MODEL_NOT_LOADED,
      );
    }

    // Reconstruct input from tensor (simple: use the float data as-is)
    const inputData = inputs[0]?.toFloat32Array() ?? new Float32Array(0);
    const result = await session.instance(inputData);

    // Wrap the result in a tensor — downstream pipelines can interpret it
    const resultArray = Array.isArray(result)
      ? new Float32Array(result.flat(Infinity) as number[])
      : new Float32Array([0]);

    return [new EdgeFlowTensor(resultArray, [resultArray.length], 'float32')];
  }

  /**
   * High-level: run the transformers.js pipeline directly with arbitrary input.
   * Returns the raw result object (not a tensor).
   */
  async runDirect(
    modelId: string,
    input: unknown,
    options?: Record<string, unknown>,
  ): Promise<unknown> {
    const session = sessionStore.get(modelId);
    if (!session) {
      throw new EdgeFlowError(
        `No transformers.js session for model ${modelId}`,
        ErrorCodes.MODEL_NOT_LOADED,
      );
    }
    return session.instance(input, options);
  }

  dispose(): void {
    for (const [id, session] of sessionStore) {
      if (session.instance.dispose) {
        session.instance.dispose();
      }
      sessionStore.delete(id);
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

let adapterRuntime: TransformersAdapterRuntime | null = null;

/**
 * Register the transformers.js adapter as the default inference backend.
 *
 * @example
 * ```typescript
 * import { pipeline } from '@xenova/transformers';
 * import { useTransformersBackend } from 'edgeflowjs';
 *
 * useTransformersBackend({
 *   pipelineFactory: pipeline,
 *   device: 'webgpu',
 *   dtype: 'fp16',
 * });
 * ```
 */
export function useTransformersBackend(options: TransformersAdapterOptions): void {
  adapterOptions = options;
  adapterRuntime = new TransformersAdapterRuntime();
  registerRuntime('wasm', () => adapterRuntime!);
}

/**
 * Get the adapter runtime instance (for advanced use).
 */
export function getTransformersAdapter(): TransformersAdapterRuntime | null {
  return adapterRuntime;
}
