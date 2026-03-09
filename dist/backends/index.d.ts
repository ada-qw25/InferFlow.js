/**
 * edgeFlow.js - Backend Exports
 */
export { WebGPURuntime, createWebGPURuntime } from './webgpu.js';
export { WebNNRuntime, createWebNNRuntime } from './webnn.js';
export { WASMRuntime, createWASMRuntime } from './wasm.js';
export { ONNXRuntime, createONNXRuntime, isOnnxAvailable } from './onnx.js';
export { TransformersAdapterRuntime, useTransformersBackend, getTransformersAdapter, type TransformersAdapterOptions, type TransformersPipelineFactory, } from './transformers-adapter.js';
export type { Runtime, RuntimeType, RuntimeCapabilities } from '../core/types.js';
/**
 * Register all available backends.
 *
 * Always registers the ONNX Runtime factory synchronously so there is no
 * async race between registration and the first pipeline() call.
 * `ONNXRuntime.isAvailable()` is called lazily by RuntimeManager when it
 * selects a backend, so if onnxruntime-web is not installed the runtime is
 * simply skipped at that point.
 */
export declare function registerAllBackends(): void;
//# sourceMappingURL=index.d.ts.map