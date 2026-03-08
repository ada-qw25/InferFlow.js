/**
 * edgeFlow.js - Image Segmentation Pipeline
 *
 * Interactive image segmentation using SAM (Segment Anything Model).
 * Supports point prompts and bounding box prompts.
 */
import { EdgeFlowTensor } from '../core/tensor.js';
import { BasePipeline, registerPipeline } from './base.js';
import { loadModel, loadModelFromBuffer, runInference, runInferenceNamed } from '../core/runtime.js';
// ============================================================================
// Default Model URLs (SlimSAM - quantized for browser)
// ============================================================================
const DEFAULT_SAM_MODELS = {
    encoder: 'https://huggingface.co/Xenova/slimsam-77-uniform/resolve/main/onnx/vision_encoder_quantized.onnx',
    decoder: 'https://huggingface.co/Xenova/slimsam-77-uniform/resolve/main/onnx/prompt_encoder_mask_decoder_quantized.onnx',
};
// ============================================================================
// Image Segmentation Pipeline
// ============================================================================
/**
 * ImageSegmentationPipeline - Interactive image segmentation
 *
 * Uses SAM-style models for point/box prompted segmentation.
 *
 * @example
 * ```typescript
 * const segmenter = createImageSegmentationPipeline();
 *
 * // Load models with progress callback
 * await segmenter.loadModels((progress) => {
 *   console.log(`Loading ${progress.model}: ${progress.progress}%`);
 * });
 *
 * // Set image and segment
 * await segmenter.setImage(imageElement);
 * const result = await segmenter.segment({
 *   points: [{ x: 0.5, y: 0.5, label: 1 }]
 * });
 * ```
 */
export class ImageSegmentationPipeline extends BasePipeline {
    encoderModel = null;
    decoderModel = null;
    imageEmbedding = null;
    imagePositionalEmbedding = null;
    currentImageSize = null;
    resizedImageSize = null;
    inputSize = 1024; // SAM default input size
    modelsLoaded = false;
    // Custom model URLs
    encoderUrl;
    decoderUrl;
    constructor(config) {
        super(config);
        this.encoderUrl = DEFAULT_SAM_MODELS.encoder;
        this.decoderUrl = DEFAULT_SAM_MODELS.decoder;
    }
    /**
     * Check if models are loaded
     */
    get isModelsLoaded() {
        return this.modelsLoaded;
    }
    /**
     * Set custom model URLs
     */
    setModelUrls(encoder, decoder) {
        this.encoderUrl = encoder;
        this.decoderUrl = decoder;
    }
    /**
     * Load both encoder and decoder models with progress callback
     */
    async loadModels(onProgress) {
        if (this.modelsLoaded)
            return;
        // Load encoder
        onProgress?.({ model: 'encoder', loaded: 0, total: 100, progress: 0 });
        const encoderData = await this.fetchModelWithProgress(this.encoderUrl, (loaded, total) => {
            onProgress?.({
                model: 'encoder',
                loaded,
                total,
                progress: Math.round((loaded / total) * 100),
            });
        });
        this.encoderModel = await loadModelFromBuffer(encoderData, {
            runtime: 'wasm', // Uses ONNXRuntime which auto-detects WebGPU internally
        });
        // Load decoder
        onProgress?.({ model: 'decoder', loaded: 0, total: 100, progress: 0 });
        const decoderData = await this.fetchModelWithProgress(this.decoderUrl, (loaded, total) => {
            onProgress?.({
                model: 'decoder',
                loaded,
                total,
                progress: Math.round((loaded / total) * 100),
            });
        });
        this.decoderModel = await loadModelFromBuffer(decoderData, {
            runtime: 'wasm', // Uses ONNXRuntime which auto-detects WebGPU internally
        });
        this.modelsLoaded = true;
    }
    /**
     * Fetch model with progress tracking
     */
    async fetchModelWithProgress(url, onProgress) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
        }
        const contentLength = response.headers.get('content-length');
        const total = contentLength ? parseInt(contentLength, 10) : 0;
        if (!response.body) {
            // Fallback if no streaming support
            const buffer = await response.arrayBuffer();
            onProgress(buffer.byteLength, buffer.byteLength);
            return buffer;
        }
        const reader = response.body.getReader();
        const chunks = [];
        let loaded = 0;
        while (true) {
            const { done, value } = await reader.read();
            if (done)
                break;
            chunks.push(value);
            loaded += value.length;
            onProgress(loaded, total || loaded);
        }
        // Combine chunks into ArrayBuffer
        const buffer = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
            buffer.set(chunk, offset);
            offset += chunk.length;
        }
        return buffer.buffer;
    }
    /**
     * Initialize pipeline (override to skip default model loading)
     */
    async initialize() {
        if (this.isReady)
            return;
        // Don't call super.initialize() - we handle model loading separately
        this.isReady = true;
    }
    /**
     * Load encoder model (processes the image once)
     */
    async loadEncoder(modelUrl) {
        this.encoderModel = await loadModel(modelUrl, {
            runtime: 'wasm',
        });
    }
    /**
     * Load decoder model (processes prompts to generate masks)
     */
    async loadDecoder(modelUrl) {
        this.decoderModel = await loadModel(modelUrl, {
            runtime: 'wasm',
        });
    }
    /**
     * Set and encode the image (call once per image)
     */
    async setImage(image) {
        if (!this.modelsLoaded) {
            throw new Error('Models not loaded. Call loadModels() first.');
        }
        // Get image data
        const imageData = await this.loadImage(image);
        this.currentImageSize = {
            width: imageData.width,
            height: imageData.height,
        };
        // Preprocess image for SAM
        const { tensor: inputTensor, resizedSize } = this.preprocessImage(imageData);
        this.resizedImageSize = resizedSize;
        // Run encoder
        if (this.encoderModel) {
            const outputs = await runInference(this.encoderModel, [inputTensor]);
            // SlimSAM encoder outputs: [image_embeddings, image_positional_embeddings]
            this.imageEmbedding = outputs[0];
            this.imagePositionalEmbedding = outputs[1];
            console.log('[SAM] Encoder outputs:', outputs.length);
            console.log('[SAM] image_embeddings shape:', this.imageEmbedding.shape);
            if (this.imagePositionalEmbedding) {
                console.log('[SAM] image_positional_embeddings shape:', this.imagePositionalEmbedding.shape);
            }
        }
        else {
            throw new Error('Encoder model not loaded');
        }
    }
    /**
     * Segment the image with given prompts
     */
    async segment(options = {}) {
        if (!this.imageEmbedding || !this.currentImageSize || !this.resizedImageSize) {
            throw new Error('No image set. Call setImage() first.');
        }
        if (!this.decoderModel) {
            throw new Error('Decoder model not loaded');
        }
        const startTime = performance.now();
        const { points = [], boxes = [], maskThreshold = 0.0, returnAllMasks = false } = options;
        // Prepare inputs for decoder
        const decoderInputs = this.prepareDecoderInputs(points, boxes);
        // Add image embeddings to inputs
        decoderInputs.set('image_embeddings', this.imageEmbedding);
        // Add positional embeddings (required by SlimSAM)
        if (this.imagePositionalEmbedding) {
            decoderInputs.set('image_positional_embeddings', this.imagePositionalEmbedding);
        }
        else {
            throw new Error('image_positional_embeddings not available from encoder');
        }
        // Run decoder model with named inputs
        const outputs = await runInferenceNamed(this.decoderModel, decoderInputs);
        // SAM decoder outputs: [masks, iou_predictions]
        const masks = outputs[0];
        const scores = outputs[1];
        // Post-process masks
        const result = this.postprocessMasks(masks, scores, maskThreshold, returnAllMasks);
        result.processingTime = performance.now() - startTime;
        return result;
    }
    /**
     * Run segmentation (implements BasePipeline interface)
     */
    async run(input, options) {
        await this.setImage(input);
        return this.segment(options);
    }
    /**
     * Load image from various sources
     */
    async loadImage(input) {
        // Handle different input types
        if (typeof input === 'string') {
            // URL or base64
            return this.loadImageFromUrl(input);
        }
        else if (input instanceof HTMLImageElement) {
            return this.imageElementToImageData(input);
        }
        else if (input instanceof HTMLCanvasElement) {
            return this.canvasToImageData(input);
        }
        else if (input instanceof ImageData) {
            return input;
        }
        else if (typeof ImageBitmap !== 'undefined' && input instanceof ImageBitmap) {
            return this.imageBitmapToImageData(input);
        }
        throw new Error('Unsupported image input type');
    }
    /**
     * Load image from URL
     */
    async loadImageFromUrl(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                resolve(ctx.getImageData(0, 0, img.width, img.height));
            };
            img.onerror = reject;
            img.src = url;
        });
    }
    /**
     * Convert HTMLImageElement to ImageData
     */
    imageElementToImageData(img) {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth || img.width;
        canvas.height = img.naturalHeight || img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        return ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    /**
     * Convert canvas to ImageData
     */
    canvasToImageData(canvas) {
        const ctx = canvas.getContext('2d');
        return ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    /**
     * Convert ImageBitmap to ImageData
     */
    imageBitmapToImageData(bitmap) {
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(bitmap, 0, 0);
        return ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    /**
     * Preprocess image for SAM
     */
    preprocessImage(imageData) {
        const { width, height } = imageData;
        // Calculate resize dimensions (longest side = inputSize)
        const scale = this.inputSize / Math.max(width, height);
        const newWidth = Math.round(width * scale);
        const newHeight = Math.round(height * scale);
        // Create resized canvas with padding
        const canvas = document.createElement('canvas');
        canvas.width = this.inputSize;
        canvas.height = this.inputSize;
        const ctx = canvas.getContext('2d');
        // Fill with padding color (SAM uses pixel mean)
        ctx.fillStyle = `rgb(123.675, 116.28, 103.53)`;
        ctx.fillRect(0, 0, this.inputSize, this.inputSize);
        // Draw resized image (top-left aligned)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageData, 0, 0);
        ctx.drawImage(tempCanvas, 0, 0, newWidth, newHeight);
        // Get pixel data
        const resizedData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
        // Convert to tensor (NCHW format, normalized with ImageNet mean/std)
        const tensorData = new Float32Array(3 * this.inputSize * this.inputSize);
        const mean = [123.675, 116.28, 103.53];
        const std = [58.395, 57.12, 57.375];
        for (let i = 0; i < this.inputSize * this.inputSize; i++) {
            const pixelIdx = i * 4;
            tensorData[i] = (resizedData.data[pixelIdx] - mean[0]) / std[0]; // R
            tensorData[this.inputSize * this.inputSize + i] =
                (resizedData.data[pixelIdx + 1] - mean[1]) / std[1]; // G
            tensorData[2 * this.inputSize * this.inputSize + i] =
                (resizedData.data[pixelIdx + 2] - mean[2]) / std[2]; // B
        }
        return {
            tensor: new EdgeFlowTensor(tensorData, [1, 3, this.inputSize, this.inputSize], 'float32'),
            resizedSize: { width: newWidth, height: newHeight },
        };
    }
    /**
     * Prepare decoder inputs (prompts) for SlimSAM
     *
     * SlimSAM prompt_encoder_mask_decoder expects these named inputs:
     * - image_embeddings: [1, 256, 64, 64]
     * - point_coords: [batch, num_points, 2]
     * - point_labels: [batch, num_points]
     * - mask_input: [batch, 1, 256, 256]
     * - has_mask_input: [batch, 1]
     * - orig_im_size: [2]
     * - position_ids: [batch, num_points]
     */
    prepareDecoderInputs(points, boxes) {
        const { width: resizedW, height: resizedH } = this.resizedImageSize;
        // Scale factors for converting normalized coords to resized image coords
        const scaleX = resizedW;
        const scaleY = resizedH;
        const allPoints = [];
        const allLabels = [];
        // Add point prompts
        for (const point of points) {
            allPoints.push(point.x * scaleX, point.y * scaleY);
            allLabels.push(point.label);
        }
        // Add box prompts (as two corner points)
        for (const box of boxes) {
            // Top-left corner (label 2)
            allPoints.push(box.x1 * scaleX, box.y1 * scaleY);
            allLabels.push(2);
            // Bottom-right corner (label 3)
            allPoints.push(box.x2 * scaleX, box.y2 * scaleY);
            allLabels.push(3);
        }
        // Default point if no prompts (center of image)
        if (allPoints.length === 0) {
            allPoints.push(resizedW / 2, resizedH / 2);
            allLabels.push(1);
        }
        const numPoints = allLabels.length;
        const inputs = new Map();
        // input_points: [1, 1, num_points, 2] - SlimSAM format (float32)
        inputs.set('input_points', new EdgeFlowTensor(new Float32Array(allPoints), [1, 1, numPoints, 2], 'float32'));
        // input_labels: [1, 1, num_points] - SlimSAM format (int64)
        inputs.set('input_labels', new EdgeFlowTensor(BigInt64Array.from(allLabels.map(l => BigInt(l))), [1, 1, numPoints], 'int64'));
        // Note: image_embeddings and image_positional_embeddings are added in segment()
        // SlimSAM decoder only needs: image_embeddings, image_positional_embeddings, input_points, input_labels
        return inputs;
    }
    /**
     * Post-process masks from decoder output
     */
    postprocessMasks(masks, scores, threshold, returnAllMasks) {
        const { width, height } = this.currentImageSize;
        const scoresData = scores.toFloat32Array();
        const masksData = masks.toFloat32Array();
        // SAM outputs multiple masks (usually 3)
        const numMasks = scoresData.length;
        const maskShape = masks.shape; // [1, num_masks, H, W]
        const maskH = maskShape[2] ?? height;
        const maskW = maskShape[3] ?? width;
        // Find best mask by score
        let bestIdx = 0;
        let bestScore = scoresData[0] ?? 0;
        for (let i = 1; i < numMasks; i++) {
            if ((scoresData[i] ?? 0) > bestScore) {
                bestScore = scoresData[i] ?? 0;
                bestIdx = i;
            }
        }
        // Extract and resize the best mask to original image size
        const outputMask = this.resizeMask(masksData, bestIdx, maskW, maskH, width, height, threshold);
        const result = {
            mask: outputMask,
            width,
            height,
            score: bestScore,
        };
        if (returnAllMasks && numMasks > 1) {
            result.allMasks = [];
            for (let m = 0; m < numMasks; m++) {
                const mask = this.resizeMask(masksData, m, maskW, maskH, width, height, threshold);
                result.allMasks.push({
                    mask,
                    score: scoresData[m] ?? 0,
                });
            }
        }
        return result;
    }
    /**
     * Resize mask from model output size to original image size
     */
    resizeMask(masksData, maskIdx, srcW, srcH, dstW, dstH, threshold) {
        const outputMask = new Uint8Array(dstW * dstH);
        const maskOffset = maskIdx * srcW * srcH;
        // Bilinear interpolation for resizing
        for (let y = 0; y < dstH; y++) {
            for (let x = 0; x < dstW; x++) {
                // Map to source coordinates
                const srcX = (x / dstW) * srcW;
                const srcY = (y / dstH) * srcH;
                // Bilinear interpolation
                const x0 = Math.floor(srcX);
                const x1 = Math.min(x0 + 1, srcW - 1);
                const y0 = Math.floor(srcY);
                const y1 = Math.min(y0 + 1, srcH - 1);
                const xFrac = srcX - x0;
                const yFrac = srcY - y0;
                const v00 = masksData[maskOffset + y0 * srcW + x0] ?? 0;
                const v01 = masksData[maskOffset + y0 * srcW + x1] ?? 0;
                const v10 = masksData[maskOffset + y1 * srcW + x0] ?? 0;
                const v11 = masksData[maskOffset + y1 * srcW + x1] ?? 0;
                const value = v00 * (1 - xFrac) * (1 - yFrac) +
                    v01 * xFrac * (1 - yFrac) +
                    v10 * (1 - xFrac) * yFrac +
                    v11 * xFrac * yFrac;
                // Apply sigmoid and threshold
                const sigmoid = 1 / (1 + Math.exp(-value));
                outputMask[y * dstW + x] = sigmoid > threshold ? 255 : 0;
            }
        }
        return outputMask;
    }
    /**
     * Clear the current image embedding
     */
    clearImage() {
        this.imageEmbedding = null;
        this.imagePositionalEmbedding = null;
        this.currentImageSize = null;
        this.resizedImageSize = null;
    }
    /**
     * Preprocess (required by BasePipeline)
     */
    async preprocess(input) {
        const imageData = await this.loadImage(input);
        const { tensor } = this.preprocessImage(imageData);
        return [tensor];
    }
    /**
     * Postprocess (required by BasePipeline)
     */
    async postprocess(_outputs, _options) {
        // This is handled in segment() method
        return {
            mask: new Uint8Array(0),
            width: 0,
            height: 0,
            score: 0,
        };
    }
    /**
     * Dispose resources
     */
    dispose() {
        super.dispose();
        this.encoderModel?.dispose();
        this.decoderModel?.dispose();
        this.imageEmbedding = null;
        this.imagePositionalEmbedding = null;
        this.currentImageSize = null;
        this.resizedImageSize = null;
        this.modelsLoaded = false;
    }
}
// ============================================================================
// Factory Function
// ============================================================================
/**
 * Create image segmentation pipeline
 */
export function createImageSegmentationPipeline(config = {}) {
    return new ImageSegmentationPipeline({
        task: 'image-segmentation',
        model: config.model ?? 'slimsam',
        runtime: config.runtime,
        cache: config.cache ?? true,
        quantization: config.quantization,
    });
}
// Register pipeline
registerPipeline('image-segmentation', (config) => new ImageSegmentationPipeline(config));
//# sourceMappingURL=image-segmentation.js.map