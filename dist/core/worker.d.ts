/**
 * edgeFlow.js - Web Worker Support
 *
 * Run inference in a Web Worker to avoid blocking the main thread.
 */
import type { Tensor, RuntimeType } from './types.js';
/**
 * Worker message types
 */
export type WorkerMessageType = 'init' | 'load_model' | 'run_inference' | 'dispose' | 'ready' | 'result' | 'error' | 'progress';
/**
 * Worker message structure
 */
export interface WorkerMessage {
    id: string;
    type: WorkerMessageType;
    payload?: unknown;
}
/**
 * Worker request for loading a model
 */
export interface LoadModelRequest {
    url: string;
    options?: {
        runtime?: RuntimeType;
        cache?: boolean;
    };
}
/**
 * Worker request for running inference
 */
export interface InferenceRequest {
    modelId: string;
    inputs: SerializedTensor[];
}
/**
 * Serialized tensor for transfer
 */
export interface SerializedTensor {
    data: ArrayBuffer;
    shape: number[];
    dtype: string;
}
/**
 * Worker pool options
 */
export interface WorkerPoolOptions {
    /** Number of workers (default: navigator.hardwareConcurrency or 4) */
    numWorkers?: number;
    /** Worker script URL (default: auto-detect) */
    workerUrl?: string;
}
/**
 * Serialize a tensor for transfer to worker
 */
export declare function serializeTensor(tensor: Tensor): SerializedTensor;
/**
 * Deserialize a tensor from worker.
 * Uses a lazy import to avoid circular dependency issues.
 */
export declare function deserializeTensor(serialized: SerializedTensor): Promise<Tensor>;
/**
 * Synchronous deserialisation used internally where async is not feasible.
 * Requires EdgeFlowTensor to be passed in to avoid require().
 */
export declare function deserializeTensorSync(serialized: SerializedTensor, TensorClass: new (data: Float32Array, shape: number[], dtype: string) => Tensor): Tensor;
export type WorkerHealthState = 'alive' | 'dead' | 'restarting';
/**
 * InferenceWorker - Wrapper for a single Web Worker with auto-restart
 */
export declare class InferenceWorker {
    private worker;
    private pendingRequests;
    private isReady;
    private readyPromise;
    private readyResolve;
    private workerUrl;
    private _health;
    private restartAttempts;
    constructor(workerUrl?: string);
    get health(): WorkerHealthState;
    /**
     * Initialize the worker
     */
    private initWorker;
    /**
     * Handle worker crash: reject pending, mark dead, attempt restart
     */
    private handleCrash;
    /**
     * Restart the worker with exponential backoff
     */
    private attemptRestart;
    /**
     * Restart: terminate old, create new
     */
    restart(): void;
    /**
     * Create worker code as blob URL
     */
    private createWorkerBlob;
    /**
     * Handle worker message
     */
    private handleMessage;
    /**
     * Send a request to the worker
     */
    private sendRequest;
    /**
     * Initialize the worker
     */
    init(): Promise<void>;
    /**
     * Load a model
     */
    loadModel(url: string, options?: {
        runtime?: RuntimeType;
        cache?: boolean;
    }): Promise<string>;
    /**
     * Run inference
     */
    runInference(modelId: string, inputs: Tensor[]): Promise<Tensor[]>;
    /**
     * Dispose a model
     */
    dispose(modelId: string): Promise<void>;
    /**
     * Terminate the worker
     */
    terminate(): void;
}
/**
 * WorkerPool - Manage multiple workers for parallel inference.
 * Automatically falls back to healthy workers when one is dead.
 */
export declare class WorkerPool {
    private workers;
    private currentIndex;
    private modelAssignments;
    private poolOptions;
    constructor(options?: WorkerPoolOptions);
    /**
     * Get next healthy worker (round-robin, skipping dead ones)
     */
    private getNextHealthyWorker;
    /**
     * Get worker for a specific model, falling back to any healthy worker
     */
    private getWorkerForModel;
    /**
     * Replace a worker at a given index with a fresh one
     */
    replaceWorker(index: number): void;
    /**
     * Initialize all workers
     */
    init(): Promise<void>;
    /**
     * Load a model on a worker
     */
    loadModel(url: string, options?: {
        runtime?: RuntimeType;
        cache?: boolean;
    }): Promise<string>;
    /**
     * Run inference (auto-retries on a healthy worker if assigned one is dead)
     */
    runInference(modelId: string, inputs: Tensor[]): Promise<Tensor[]>;
    /**
     * Run inference on multiple inputs in parallel
     */
    runBatch(modelId: string, batchInputs: Tensor[][]): Promise<Tensor[][]>;
    /**
     * Dispose a model
     */
    dispose(modelId: string): Promise<void>;
    /**
     * Terminate all workers
     */
    terminate(): void;
    /**
     * Get number of workers
     */
    get size(): number;
}
/**
 * Get or create global worker pool
 */
export declare function getWorkerPool(options?: WorkerPoolOptions): WorkerPool;
/**
 * Run inference in a worker
 */
export declare function runInWorker(modelUrl: string, inputs: Tensor[], options?: {
    cache?: boolean;
}): Promise<Tensor[]>;
/**
 * Check if Web Workers are supported
 */
export declare function isWorkerSupported(): boolean;
//# sourceMappingURL=worker.d.ts.map