/**
 * edgeFlow.js - ONNX Runtime Backend
 * 
 * Uses onnxruntime-web for real ONNX model inference.
 */

import * as ort from 'onnxruntime-web';
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
   * Check if ONNX Runtime is available
   */
  async isAvailable(): Promise<boolean> {
    return true;
  }

  /**
   * Initialize the ONNX runtime
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Configure WASM paths for CDN loading (required for browser deployment)
    if (typeof window !== 'undefined') {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
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
      // Create session options with multiple execution providers
      // ONNX Runtime will try them in order and use the first available one
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: ['webgpu', 'wasm'], // Try WebGPU first, fallback to WASM
        graphOptimizationLevel: 'all',
      };

      // Create inference session (convert ArrayBuffer to Uint8Array)
      const modelBytes = new Uint8Array(modelData);
      
      let session: ort.InferenceSession;
      try {
        session = await ort.InferenceSession.create(modelBytes, sessionOptions);
        console.log('[ONNX] Session created (tried WebGPU → WASM)');
      } catch (e) {
        // If WebGPU fails, try WASM only
        console.log('[ONNX] WebGPU not available, falling back to WASM. Reason:', e instanceof Error ? e.message : e);
        const wasmOptions: ort.InferenceSession.SessionOptions = {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        };
        session = await ort.InferenceSession.create(modelBytes, wasmOptions);
        console.log('[ONNX] Session created with WASM backend');
      }
      
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
        inputs: inputNames.map(name => ({
          name,
          dtype: 'float32' as DataType,
          shape: [-1], // Dynamic shape
        })),
        outputs: outputNames.map(name => ({
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
      // Prepare input feeds
      const feeds: Record<string, any> = {};
      
      for (let i = 0; i < Math.min(inputs.length, inputNames.length); i++) {
        const inputName = inputNames[i];
        const inputTensor = inputs[i] as EdgeFlowTensor;
        
        if (inputName && inputTensor) {
          // Convert to ONNX tensor with correct dtype
          const dtype = inputTensor.dtype;
          let ortTensor: any;
          
          if (dtype === 'int64') {
            // Get raw BigInt64Array data directly
            const data = inputTensor.data as unknown as BigInt64Array;
            ortTensor = new ort.Tensor('int64', data, inputTensor.shape as number[]);
          } else if (dtype === 'int32') {
            const data = inputTensor.data as Int32Array;
            ortTensor = new ort.Tensor('int32', data, inputTensor.shape as number[]);
          } else {
            const data = inputTensor.toFloat32Array();
            ortTensor = new ort.Tensor('float32', data, inputTensor.shape as number[]);
          }
          
          feeds[inputName] = ortTensor;
        }
      }

      // Run inference
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
      // Prepare input feeds from named inputs
      const feeds: Record<string, any> = {};
      
      // Log expected vs provided inputs for debugging
      console.log('[ONNX] Model expects inputs:', inputNames);
      console.log('[ONNX] Provided inputs:', Array.from(namedInputs.keys()));
      
      for (const [inputName, inputTensor] of namedInputs) {
        const tensor = inputTensor as EdgeFlowTensor;
        const dtype = tensor.dtype;
        let ortTensor: any;
        
        if (dtype === 'int64') {
          const data = tensor.data as unknown as BigInt64Array;
          ortTensor = new ort.Tensor('int64', data, tensor.shape as number[]);
        } else if (dtype === 'int32') {
          const data = tensor.data as Int32Array;
          ortTensor = new ort.Tensor('int32', data, tensor.shape as number[]);
        } else {
          const data = tensor.toFloat32Array();
          ortTensor = new ort.Tensor('float32', data, tensor.shape as number[]);
        }
        
        feeds[inputName] = ortTensor;
        console.log(`[ONNX] Input '${inputName}': shape=${tensor.shape}, dtype=${dtype}`);
      }

      // Run inference
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
      // Log detailed error info
      console.error('[ONNX] Inference failed. Model expects:', inputNames);
      console.error('[ONNX] Provided:', Array.from(namedInputs.keys()));
      throw new EdgeFlowError(
        `ONNX inference failed: ${error instanceof Error ? error.message : String(error)}`,
        ErrorCodes.INFERENCE_FAILED,
        { modelId: model.id, error }
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
