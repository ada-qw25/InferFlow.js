/**
 * edgeFlow.js - Core Module Exports
 */
export * from './types.js';
export { EdgeFlowTensor, tensor, zeros, ones, full, random, randn, arange, linspace, eye, add, sub, mul, div, matmul, softmax, relu, sigmoid, tanh, sum, mean, argmax, concat, } from './tensor.js';
export { InferenceScheduler, getScheduler, setScheduler, configureScheduler, } from './scheduler.js';
export { MemoryManager, MemoryScope, ModelCache, withMemoryScope, withMemoryScopeSync, getMemoryManager, getMemoryStats, release, gc, } from './memory.js';
export { RuntimeManager, LoadedModelImpl, loadModel, loadModelFromBuffer, runInference, runBatchInference, getRuntimeManager, registerRuntime, getBestRuntime, getAvailableRuntimes, } from './runtime.js';
export { registerPlugin, getPluginPipeline, getPluginMiddleware, listPlugins, unregisterPlugin, type EdgeFlowPlugin, type PluginPipelineEntry, type PluginBackendEntry, type PluginMiddleware, } from './plugin.js';
export { getDeviceProfile, recommendQuantization, recommendModelVariant, resetDeviceProfile, type DeviceProfile, type DeviceTier, type ModelRecommendation, } from './device-profiler.js';
export { compose, parallel, type CompositionStage, type CompositionResult, type ComposedPipeline, } from './composer.js';
export { InferenceWorker, WorkerPool, getWorkerPool, runInWorker, isWorkerSupported, serializeTensor, deserializeTensor, type WorkerMessage, type WorkerMessageType, type LoadModelRequest, type InferenceRequest, type SerializedTensor, type WorkerPoolOptions, } from './worker.js';
//# sourceMappingURL=index.d.ts.map