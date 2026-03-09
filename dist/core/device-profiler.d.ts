/**
 * edgeFlow.js - Device Profiler
 *
 * Automatically profiles the current device and recommends optimal model
 * variants (quantization level, batch size, execution provider).
 *
 * @example
 * ```typescript
 * import { getDeviceProfile, recommendQuantization } from 'edgeflowjs';
 *
 * const profile = await getDeviceProfile();
 * console.log(profile.tier); // 'high' | 'medium' | 'low'
 *
 * const quant = recommendQuantization(profile);
 * console.log(quant); // 'fp16' | 'int8' | 'int4'
 * ```
 */
import type { QuantizationType } from './types.js';
/**
 * Device capability tier.
 */
export type DeviceTier = 'high' | 'medium' | 'low';
/**
 * Profiled device information.
 */
export interface DeviceProfile {
    /** Capability tier */
    tier: DeviceTier;
    /** Number of logical CPU cores */
    cores: number;
    /** Device memory in GiB (navigator.deviceMemory, may be null) */
    memoryGiB: number | null;
    /** Whether WebGPU is available */
    webgpu: boolean;
    /** Whether WebNN is available */
    webnn: boolean;
    /** Recommended max batch size */
    recommendedBatchSize: number;
    /** Recommended concurrency limit */
    recommendedConcurrency: number;
    /** Whether the device is mobile */
    mobile: boolean;
    /** Raw GPU adapter info (if WebGPU available) */
    gpuInfo?: string;
}
/**
 * Model variant recommendation.
 */
export interface ModelRecommendation {
    /** Recommended quantization */
    quantization: QuantizationType;
    /** Recommended execution provider */
    executionProvider: 'webgpu' | 'wasm';
    /** Recommended batch size */
    batchSize: number;
    /** Whether to enable worker-based inference */
    useWorker: boolean;
}
/**
 * Profile the current device. Results are cached after the first call.
 */
export declare function getDeviceProfile(): Promise<DeviceProfile>;
/**
 * Recommend the best quantization level for the current device.
 */
export declare function recommendQuantization(profile: DeviceProfile): QuantizationType;
/**
 * Get full model variant recommendations for the current device.
 */
export declare function recommendModelVariant(): Promise<ModelRecommendation>;
/**
 * Reset the cached profile (useful for testing).
 */
export declare function resetDeviceProfile(): void;
//# sourceMappingURL=device-profiler.d.ts.map