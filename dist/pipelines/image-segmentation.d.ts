/**
 * edgeFlow.js - Image Segmentation Pipeline
 *
 * Interactive image segmentation using SAM (Segment Anything Model).
 * Supports point prompts and bounding box prompts.
 */
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { BasePipeline, PipelineResult } from './base.js';
/**
 * Point prompt for segmentation
 */
export interface PointPrompt {
    /** X coordinate (0-1 normalized) */
    x: number;
    /** Y coordinate (0-1 normalized) */
    y: number;
    /** 1 for foreground (include), 0 for background (exclude) */
    label: 0 | 1;
}
/**
 * Box prompt for segmentation
 */
export interface BoxPrompt {
    /** Top-left X (0-1 normalized) */
    x1: number;
    /** Top-left Y (0-1 normalized) */
    y1: number;
    /** Bottom-right X (0-1 normalized) */
    x2: number;
    /** Bottom-right Y (0-1 normalized) */
    y2: number;
}
/**
 * Model loading progress callback
 */
export interface ModelLoadProgress {
    /** Model name (encoder or decoder) */
    model: 'encoder' | 'decoder';
    /** Bytes loaded */
    loaded: number;
    /** Total bytes */
    total: number;
    /** Progress percentage (0-100) */
    progress: number;
}
/**
 * Segmentation options
 */
export interface ImageSegmentationOptions extends PipelineOptions {
    /** Point prompts */
    points?: PointPrompt[];
    /** Box prompts */
    boxes?: BoxPrompt[];
    /** Return all masks or just the best one */
    returnAllMasks?: boolean;
    /** Mask threshold (0-1) */
    maskThreshold?: number;
}
/**
 * Segmentation result
 */
export interface ImageSegmentationResult extends PipelineResult {
    /** Segmentation mask (Uint8Array, 0 or 255) */
    mask: Uint8Array;
    /** Mask width */
    width: number;
    /** Mask height */
    height: number;
    /** Confidence score */
    score: number;
    /** All masks if returnAllMasks is true */
    allMasks?: Array<{
        mask: Uint8Array;
        score: number;
    }>;
}
/**
 * Image input types
 */
export type ImageInput = HTMLImageElement | HTMLCanvasElement | ImageBitmap | ImageData | string;
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
export declare class ImageSegmentationPipeline extends BasePipeline<ImageInput, ImageSegmentationResult> {
    private encoderModel;
    private decoderModel;
    private imageEmbedding;
    private currentImageSize;
    private resizedImageSize;
    private inputSize;
    private modelsLoaded;
    private encoderUrl;
    private decoderUrl;
    constructor(config: PipelineConfig);
    /**
     * Check if models are loaded
     */
    get isModelsLoaded(): boolean;
    /**
     * Set custom model URLs
     */
    setModelUrls(encoder: string, decoder: string): void;
    /**
     * Load both encoder and decoder models with progress callback
     */
    loadModels(onProgress?: (progress: ModelLoadProgress) => void): Promise<void>;
    /**
     * Fetch model with progress tracking
     */
    private fetchModelWithProgress;
    /**
     * Initialize pipeline (override to skip default model loading)
     */
    initialize(): Promise<void>;
    /**
     * Load encoder model (processes the image once)
     */
    loadEncoder(modelUrl: string): Promise<void>;
    /**
     * Load decoder model (processes prompts to generate masks)
     */
    loadDecoder(modelUrl: string): Promise<void>;
    /**
     * Set and encode the image (call once per image)
     */
    setImage(image: ImageInput): Promise<void>;
    /**
     * Segment the image with given prompts
     */
    segment(options?: ImageSegmentationOptions): Promise<ImageSegmentationResult>;
    /**
     * Run segmentation (implements BasePipeline interface)
     */
    run(input: ImageInput, options?: ImageSegmentationOptions): Promise<ImageSegmentationResult>;
    /**
     * Load image from various sources
     */
    private loadImage;
    /**
     * Load image from URL
     */
    private loadImageFromUrl;
    /**
     * Convert HTMLImageElement to ImageData
     */
    private imageElementToImageData;
    /**
     * Convert canvas to ImageData
     */
    private canvasToImageData;
    /**
     * Convert ImageBitmap to ImageData
     */
    private imageBitmapToImageData;
    /**
     * Preprocess image for SAM
     */
    private preprocessImage;
    /**
     * Prepare decoder inputs (prompts)
     */
    private prepareDecoderInputs;
    /**
     * Post-process masks from decoder output
     */
    private postprocessMasks;
    /**
     * Resize mask from model output size to original image size
     */
    private resizeMask;
    /**
     * Clear the current image embedding
     */
    clearImage(): void;
    /**
     * Preprocess (required by BasePipeline)
     */
    protected preprocess(input: ImageInput): Promise<EdgeFlowTensor[]>;
    /**
     * Postprocess (required by BasePipeline)
     */
    protected postprocess(_outputs: EdgeFlowTensor[], _options?: PipelineOptions): Promise<ImageSegmentationResult>;
    /**
     * Dispose resources
     */
    dispose(): void;
}
/**
 * Create image segmentation pipeline
 */
export declare function createImageSegmentationPipeline(config?: Partial<PipelineConfig>): ImageSegmentationPipeline;
//# sourceMappingURL=image-segmentation.d.ts.map