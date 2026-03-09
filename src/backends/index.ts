/**
 * edgeFlow.js - Backend Exports
 */

// WebGPU Backend (planned - skeleton only)
export { WebGPURuntime, createWebGPURuntime } from './webgpu.js';

// WebNN Backend (planned - skeleton only)
export { WebNNRuntime, createWebNNRuntime } from './webnn.js';

// WASM Backend (basic tensor ops)
export { WASMRuntime, createWASMRuntime } from './wasm.js';

// ONNX Runtime Backend (real model inference)
export { ONNXRuntime, createONNXRuntime, isOnnxAvailable } from './onnx.js';

// transformers.js Adapter Backend
export {
  TransformersAdapterRuntime,
  useTransformersBackend,
  getTransformersAdapter,
  type TransformersAdapterOptions,
  type TransformersPipelineFactory,
} from './transformers-adapter.js';

// Re-export types
export type { Runtime, RuntimeType, RuntimeCapabilities } from '../core/types.js';

import { registerRuntime } from '../core/runtime.js';
import { createONNXRuntime } from './onnx.js';

/**
 * Register all available backends.
 *
 * Always registers the ONNX Runtime factory synchronously so there is no
 * async race between registration and the first pipeline() call.
 * `ONNXRuntime.isAvailable()` is called lazily by RuntimeManager when it
 * selects a backend, so if onnxruntime-web is not installed the runtime is
 * simply skipped at that point.
 */
export function registerAllBackends(): void {
  registerRuntime('wasm', createONNXRuntime);
}

/**
 * Auto-register backends on module load (synchronous — no race condition).
 */
registerAllBackends();
