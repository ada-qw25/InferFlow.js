/**
 * edgeFlow.js - ONNX Runtime Backend
 *
 * Uses onnxruntime-web for real ONNX model inference.
 * onnxruntime-web is an optional peer dependency loaded dynamically.
 */
import { EdgeFlowError, ErrorCodes, } from '../core/types.js';
import { LoadedModelImpl } from '../core/runtime.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { getMemoryManager } from '../core/memory.js';
// Lazy-loaded onnxruntime-web module
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ort = null;
async function getOrt() {
    if (ort)
        return ort;
    try {
        // Import the WASM-only sub-path so Vite rewrites the bare specifier
        // to ort.wasm.bundle.min.mjs. This avoids loading the JSEP/WebGPU
        // worker module (jsep.mjs) that ort.bundle.min.mjs eagerly fetches
        // whenever navigator.gpu exists — which causes a 404 in dev servers
        // that restrict ES module imports from /public.
        ort = await import('onnxruntime-web/wasm');
        return ort;
    }
    catch {
        return null;
    }
}
/**
 * Check whether onnxruntime-web is importable.
 */
export async function isOnnxAvailable() {
    return (await getOrt()) != null;
}
const sessionStore = new Map();
// ============================================================================
// ONNX Runtime Implementation
// ============================================================================
/**
 * ONNXRuntime - Real ONNX model inference using onnxruntime-web
 */
export class ONNXRuntime {
    name = 'wasm'; // Register as wasm since it's the fallback
    initialized = false;
    executionProvider = 'wasm';
    get capabilities() {
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
    async isAvailable() {
        return isOnnxAvailable();
    }
    /**
     * Initialize the ONNX runtime
     */
    async initialize() {
        if (this.initialized)
            return;
        const ortModule = await getOrt();
        if (!ortModule) {
            throw new EdgeFlowError('onnxruntime-web is not installed. Install it with: npm install onnxruntime-web', ErrorCodes.RUNTIME_NOT_AVAILABLE);
        }
        // Configure WASM backend for browser use.
        // numThreads=1 disables multi-threading so ort only needs the plain
        // .wasm binary — the worker .mjs file is never requested, which avoids
        // Vite's restriction on importing files from /public as ES modules.
        // Consumers should copy onnxruntime-web/dist/*.wasm to public/ort/.
        if (typeof window !== 'undefined' && ortModule.env?.wasm) {
            ortModule.env.wasm.wasmPaths = '/ort/';
            ortModule.env.wasm.numThreads = 1;
        }
        this.initialized = true;
    }
    /**
     * Load a model from ArrayBuffer
     */
    async loadModel(modelData, options = {}) {
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
            const session = await ortModule.InferenceSession.create(modelBytes, sessionOptions);
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
            const metadata = {
                name: options.metadata?.name ?? 'onnx-model',
                version: '1.0.0',
                inputs: inputNames.map((name) => ({
                    name,
                    dtype: 'float32',
                    shape: [-1], // Dynamic shape
                })),
                outputs: outputNames.map((name) => ({
                    name,
                    dtype: 'float32',
                    shape: [-1],
                })),
                sizeBytes: modelData.byteLength,
                quantization: options.quantization ?? 'float32',
                format: 'onnx',
            };
            // Create model instance
            const model = new LoadedModelImpl(metadata, 'wasm', () => this.unloadModel(modelId));
            // Override the ID to match our stored session
            Object.defineProperty(model, 'id', { value: modelId, writable: false });
            // Track in memory manager
            getMemoryManager().trackModel(model, () => model.dispose());
            return model;
        }
        catch (error) {
            throw new EdgeFlowError(`Failed to load ONNX model: ${error instanceof Error ? error.message : String(error)}`, ErrorCodes.MODEL_LOAD_FAILED, { error });
        }
    }
    /**
     * Run inference
     */
    async run(model, inputs) {
        const sessionData = sessionStore.get(model.id);
        if (!sessionData) {
            throw new EdgeFlowError(`ONNX session not found for model ${model.id}`, ErrorCodes.MODEL_NOT_LOADED, { modelId: model.id });
        }
        const { session, inputNames, outputNames } = sessionData;
        try {
            const ortModule = await getOrt();
            const feeds = {};
            for (let i = 0; i < Math.min(inputs.length, inputNames.length); i++) {
                const inputName = inputNames[i];
                const inputTensor = inputs[i];
                if (inputName && inputTensor) {
                    const dtype = inputTensor.dtype;
                    let ortTensor;
                    if (dtype === 'int64') {
                        const data = inputTensor.data;
                        ortTensor = new ortModule.Tensor('int64', data, inputTensor.shape);
                    }
                    else if (dtype === 'int32') {
                        const data = inputTensor.data;
                        ortTensor = new ortModule.Tensor('int32', data, inputTensor.shape);
                    }
                    else {
                        const data = inputTensor.toFloat32Array();
                        ortTensor = new ortModule.Tensor('float32', data, inputTensor.shape);
                    }
                    feeds[inputName] = ortTensor;
                }
            }
            const results = await session.run(feeds);
            // Convert outputs to EdgeFlowTensor
            const outputs = [];
            for (const outputName of outputNames) {
                const ortTensor = results[outputName];
                if (ortTensor) {
                    const data = ortTensor.data;
                    const shape = Array.from(ortTensor.dims).map(d => Number(d));
                    outputs.push(new EdgeFlowTensor(new Float32Array(data), shape, 'float32'));
                }
            }
            return outputs;
        }
        catch (error) {
            throw new EdgeFlowError(`ONNX inference failed: ${error instanceof Error ? error.message : String(error)}`, ErrorCodes.INFERENCE_FAILED, { modelId: model.id, error });
        }
    }
    /**
     * Run inference with named inputs
     */
    async runNamed(model, namedInputs) {
        const sessionData = sessionStore.get(model.id);
        if (!sessionData) {
            throw new EdgeFlowError(`ONNX session not found for model ${model.id}`, ErrorCodes.MODEL_NOT_LOADED, { modelId: model.id });
        }
        const { session, inputNames, outputNames } = sessionData;
        try {
            const ortModule = await getOrt();
            const feeds = {};
            for (const [inputName, inputTensor] of namedInputs) {
                const tensor = inputTensor;
                const dtype = tensor.dtype;
                let ortTensor;
                if (dtype === 'int64') {
                    const data = tensor.data;
                    ortTensor = new ortModule.Tensor('int64', data, tensor.shape);
                }
                else if (dtype === 'int32') {
                    const data = tensor.data;
                    ortTensor = new ortModule.Tensor('int32', data, tensor.shape);
                }
                else {
                    const data = tensor.toFloat32Array();
                    ortTensor = new ortModule.Tensor('float32', data, tensor.shape);
                }
                feeds[inputName] = ortTensor;
            }
            const results = await session.run(feeds);
            // Convert outputs to EdgeFlowTensor
            const outputs = [];
            for (const outputName of outputNames) {
                const ortTensor = results[outputName];
                if (ortTensor) {
                    const data = ortTensor.data;
                    const shape = Array.from(ortTensor.dims).map(d => Number(d));
                    outputs.push(new EdgeFlowTensor(new Float32Array(data), shape, 'float32'));
                }
            }
            return outputs;
        }
        catch (error) {
            throw new EdgeFlowError(`ONNX inference failed: ${error instanceof Error ? error.message : String(error)}`, ErrorCodes.INFERENCE_FAILED, { modelId: model.id, expectedInputs: inputNames, providedInputs: Array.from(namedInputs.keys()), error });
        }
    }
    /**
     * Unload a model
     */
    async unloadModel(modelId) {
        const sessionData = sessionStore.get(modelId);
        if (sessionData) {
            // Release session will be handled by GC
            sessionStore.delete(modelId);
        }
    }
    /**
     * Dispose the runtime
     */
    dispose() {
        // Clear all sessions
        sessionStore.clear();
        this.initialized = false;
    }
}
/**
 * Create ONNX runtime factory
 */
export function createONNXRuntime() {
    return new ONNXRuntime();
}
//# sourceMappingURL=onnx.js.map