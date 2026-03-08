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
export { ONNXRuntime, createONNXRuntime } from './onnx.js';

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

/**
 * Initialize all backends with the runtime manager
 */
import { registerRuntime } from '../core/runtime.js';
import { createONNXRuntime } from './onnx.js';

/**
 * Register all available backends.
 * 
 * Only ONNX Runtime is registered by default as it is the only backend
 * that performs real inference. ONNX Runtime supports WebGPU and WebNN
 * via its own execution providers, so GPU acceleration is still available.
 * 
 * The WebGPU and WebNN backends are planned for future custom shader /
 * native graph support and can be registered manually if needed.
 */
export function registerAllBackends(): void {
  registerRuntime('wasm', createONNXRuntime);
}

/**
 * Auto-register backends on module load
 */
registerAllBackends();
