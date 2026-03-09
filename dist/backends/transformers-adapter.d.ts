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
import { Runtime, RuntimeType, RuntimeCapabilities, LoadedModel, ModelLoadOptions, Tensor } from '../core/types.js';
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
export type TransformersPipelineFactory = (task: string, model?: string, options?: Record<string, unknown>) => Promise<TransformersPipelineInstance>;
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
export declare class TransformersAdapterRuntime implements Runtime {
    readonly name: RuntimeType;
    get capabilities(): RuntimeCapabilities;
    isAvailable(): Promise<boolean>;
    initialize(): Promise<void>;
    loadModel(modelData: ArrayBuffer, options?: ModelLoadOptions): Promise<LoadedModel>;
    /**
     * Load a transformers.js pipeline by task + model name
     * (called by the higher-level adapter pipeline, not via the
     * standard loadModel path).
     */
    loadPipeline(task: string, model: string, pipelineOptions?: Record<string, unknown>): Promise<string>;
    /**
     * Run inference by passing the raw input to the transformers.js pipeline.
     * The result is returned as a single EdgeFlowTensor wrapping the JSON-encoded output
     * (since transformers.js returns task-specific objects, not raw tensors).
     */
    run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]>;
    /**
     * High-level: run the transformers.js pipeline directly with arbitrary input.
     * Returns the raw result object (not a tensor).
     */
    runDirect(modelId: string, input: unknown, options?: Record<string, unknown>): Promise<unknown>;
    dispose(): void;
}
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
export declare function useTransformersBackend(options: TransformersAdapterOptions): void;
/**
 * Get the adapter runtime instance (for advanced use).
 */
export declare function getTransformersAdapter(): TransformersAdapterRuntime | null;
export {};
//# sourceMappingURL=transformers-adapter.d.ts.map