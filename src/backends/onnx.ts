/**
 * edgeFlow.js - ONNX Runtime Backend
 * 
 * Uses onnxruntime-web for real ONNX model inference.
 * onnxruntime-web is an optional peer dependency loaded dynamically.
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
  DataType,
} from '../core/types.js';
import { LoadedModelImpl } from '../core/runtime.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { getMemoryManager } from '../core/memory.js';

// Lazy-loaded onnxruntime-web module
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ort: any = null;

async function getOrt(): Promise<any> {
  if (ort) return ort;
  try {
    // Import the WASM-only sub-path so Vite rewrites the bare specifier
    // to ort.wasm.bundle.min.mjs. This avoids loading the JSEP/WebGPU
    // worker module (jsep.mjs) that ort.bundle.min.mjs eagerly fetches
    // whenever navigator.gpu exists — which causes a 404 in dev servers
    // that restrict ES module imports from /public.
    ort = await import('onnxruntime-web/wasm');
    return ort;
  } catch {
    return null;
  }
}

/**
 * Check whether onnxruntime-web is importable.
 */
export async function isOnnxAvailable(): Promise<boolean> {
  return (await getOrt()) != null;
}

// ============================================================================
// ONNX Session Storage
// ============================================================================

interface ONNXSessionData {
  session: any; // ort.InferenceSession
  inputNames: string[];
  outputNames: string[];
}

const sessionStore: Map<string, ONNXSessionData> = new Map();

// ============================================================================
// ONNX Runtime Implementation
// ============================================================================

/**
 * ONNXRuntime - Real ONNX model inference using onnxruntime-web
 */
export class ONNXRuntime implements Runtime {
  readonly name: RuntimeType = 'wasm'; // Register as wasm since it's the fallback
  
  private initialized = false;
  private executionProvider: 'webgpu' | 'wasm' = 'wasm';

  get capabilities(): RuntimeCapabilities {
    return {
      concurrency: true,
      quantization: true,
      float16: this.executionProvider === 'webgpu',
      dynamicShapes: true,
      maxBatchSize: 32,
      availableMemory: 512 * 1024 * 1024, // 512MB
    };
  }

  /**
   * Check if ONNX Runtime is available (peer dependency installed)
   */
  async isAvailable(): Promise<boolean> {
    return isOnnxAvailable();
  }

  /**
   * Initialize the ONNX runtime
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    const ortModule = await getOrt();
    if (!ortModule) {
      throw new EdgeFlowError(
        'onnxruntime-web is not installed. Install it with: npm install onnxruntime-web',
        ErrorCodes.RUNTIME_NOT_AVAILABLE
      );
    }

    // Configure WASM backend for browser use.
    // numThreads=1 disables multi-threading so ort only needs the plain
    // .wasm binary — the worker .mjs file is never requested, which avoids
    // Vite's restriction on importing files from /public as ES modules.
    // Consumers should copy onnxruntime-web/dist/*.wasm to public/ort/.
    if (typeof window !== 'undefined' && ortModule.env?.wasm) {
      (ortModule.env.wasm as any).wasmPaths = '/ort/';
      (ortModule.env.wasm as any).numThreads = 1;
    }

    this.initialized = true;
  }

  /**
   * Load a model from ArrayBuffer
   */
  async loadModel(
    modelData: ArrayBuffer,
    options: ModelLoadOptions = {}
  ): Promise<LoadedModel> {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      const ortModule = await getOrt();
      if (!ortModule) {
        throw new Error('onnxruntime-web is not installed');
      }

      // WASM-only execution provider — WebGPU acceleration can be added
      // later via the dedicated WebGPURuntime backend.
      const sessionOptions = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      };

      const modelBytes = new Uint8Array(modelData);

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const session: any = await ortModule.InferenceSession.create(modelBytes, sessionOptions);
      
      // Get input/output names
      const inputNames = session.inputNames;
      const outputNames = session.outputNames;

      // Generate model ID
      const modelId = `onnx_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

      // Store session
      sessionStore.set(modelId, {
        session,
        inputNames: [...inputNames],
        outputNames: [...outputNames],
      });

      // Create metadata
      const metadata: ModelMetadata = {
        name: options.metadata?.name ?? 'onnx-model',
        version: '1.0.0',
        inputs: inputNames.map((name: string) => ({
          name,
          dtype: 'float32' as DataType,
          shape: [-1], // Dynamic shape
        })),
        outputs: outputNames.map((name: string) => ({
          name,
          dtype: 'float32' as DataType,
          shape: [-1],
        })),
        sizeBytes: modelData.byteLength,
        quantization: options.quantization ?? 'float32',
        format: 'onnx',
      };

      // Create model instance
      const model = new LoadedModelImpl(
        metadata,
        'wasm',
        () => this.unloadModel(modelId)
      );

      // Override the ID to match our stored session
      Object.defineProperty(model, 'id', { value: modelId, writable: false });

      // Track in memory manager
      getMemoryManager().trackModel(model, () => model.dispose());

      return model;
    } catch (error) {
      throw new EdgeFlowError(
        `Failed to load ONNX model: ${error instanceof Error ? error.message : String(error)}`,
        ErrorCodes.MODEL_LOAD_FAILED,
        { error }
      );
    }
  }

  /**
   * Run inference
   */
  async run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]> {
    const sessionData = sessionStore.get(model.id);
    if (!sessionData) {
      throw new EdgeFlowError(
        `ONNX session not found for model ${model.id}`,
        ErrorCodes.MODEL_NOT_LOADED,
        { modelId: model.id }
      );
    }

    const { session, inputNames, outputNames } = sessionData;

    try {
      const ortModule = await getOrt();
      const feeds: Record<string, any> = {};
      
      for (let i = 0; i < Math.min(inputs.length, inputNames.length); i++) {
        const inputName = inputNames[i];
        const inputTensor = inputs[i] as EdgeFlowTensor;
        
        if (inputName && inputTensor) {
          const dtype = inputTensor.dtype;
          let ortTensor: any;
          
          if (dtype === 'int64') {
            const data = inputTensor.data as unknown as BigInt64Array;
            ortTensor = new ortModule.Tensor('int64', data, inputTensor.shape as number[]);
          } else if (dtype === 'int32') {
            const data = inputTensor.data as Int32Array;
            ortTensor = new ortModule.Tensor('int32', data, inputTensor.shape as number[]);
          } else {
            const data = inputTensor.toFloat32Array();
            ortTensor = new ortModule.Tensor('float32', data, inputTensor.shape as number[]);
          }
          
          feeds[inputName] = ortTensor;
        }
      }

      const results = await session.run(feeds);

      // Convert outputs to EdgeFlowTensor
      const outputs: Tensor[] = [];
      
      for (const outputName of outputNames) {
        const ortTensor = results[outputName];
        if (ortTensor) {
          const data = ortTensor.data as Float32Array;
          const shape = Array.from(ortTensor.dims).map(d => Number(d));
          outputs.push(new EdgeFlowTensor(new Float32Array(data), shape, 'float32'));
        }
      }

      return outputs;
    } catch (error) {
      throw new EdgeFlowError(
        `ONNX inference failed: ${error instanceof Error ? error.message : String(error)}`,
        ErrorCodes.INFERENCE_FAILED,
        { modelId: model.id, error }
      );
    }
  }

  /**
   * Run inference with named inputs
   */
  async runNamed(model: LoadedModel, namedInputs: Map<string, Tensor>): Promise<Tensor[]> {
    const sessionData = sessionStore.get(model.id);
    if (!sessionData) {
      throw new EdgeFlowError(
        `ONNX session not found for model ${model.id}`,
        ErrorCodes.MODEL_NOT_LOADED,
        { modelId: model.id }
      );
    }

    const { session, inputNames, outputNames } = sessionData;

    try {
      const ortModule = await getOrt();
      const feeds: Record<string, any> = {};
      
      for (const [inputName, inputTensor] of namedInputs) {
        const tensor = inputTensor as EdgeFlowTensor;
        const dtype = tensor.dtype;
        let ortTensor: any;
        
        if (dtype === 'int64') {
          const data = tensor.data as unknown as BigInt64Array;
          ortTensor = new ortModule.Tensor('int64', data, tensor.shape as number[]);
        } else if (dtype === 'int32') {
          const data = tensor.data as Int32Array;
          ortTensor = new ortModule.Tensor('int32', data, tensor.shape as number[]);
        } else {
          const data = tensor.toFloat32Array();
          ortTensor = new ortModule.Tensor('float32', data, tensor.shape as number[]);
        }
        
        feeds[inputName] = ortTensor;
      }

      const results = await session.run(feeds);

      // Convert outputs to EdgeFlowTensor
      const outputs: Tensor[] = [];
      
      for (const outputName of outputNames) {
        const ortTensor = results[outputName];
        if (ortTensor) {
          const data = ortTensor.data as Float32Array;
          const shape = Array.from(ortTensor.dims).map(d => Number(d));
          outputs.push(new EdgeFlowTensor(new Float32Array(data), shape, 'float32'));
        }
      }

      return outputs;
    } catch (error) {
      throw new EdgeFlowError(
        `ONNX inference failed: ${error instanceof Error ? error.message : String(error)}`,
        ErrorCodes.INFERENCE_FAILED,
        { modelId: model.id, expectedInputs: inputNames, providedInputs: Array.from(namedInputs.keys()), error }
      );
    }
  }

  /**
   * Unload a model
   */
  private async unloadModel(modelId: string): Promise<void> {
    const sessionData = sessionStore.get(modelId);
    if (sessionData) {
      // Release session will be handled by GC
      sessionStore.delete(modelId);
    }
  }

  /**
   * Dispose the runtime
   */
  dispose(): void {
    // Clear all sessions
    sessionStore.clear();
    this.initialized = false;
  }
}

/**
 * Create ONNX runtime factory
 */
export function createONNXRuntime(): Runtime {
  return new ONNXRuntime();
}
