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
// ---------------------------------------------------------------------------
// Profiling
// ---------------------------------------------------------------------------
let cachedProfile = null;
/**
 * Profile the current device. Results are cached after the first call.
 */
export async function getDeviceProfile() {
    if (cachedProfile)
        return cachedProfile;
    const cores = typeof navigator !== 'undefined'
        ? navigator.hardwareConcurrency ?? 2
        : 2;
    const memoryGiB = typeof navigator !== 'undefined' && 'deviceMemory' in navigator
        ? navigator.deviceMemory ?? null
        : null;
    const mobile = typeof navigator !== 'undefined'
        ? /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)
        : false;
    let webgpu = false;
    let gpuInfo;
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            webgpu = adapter != null;
            if (adapter && typeof adapter === 'object') {
                try {
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    const info = adapter['info'];
                    if (info) {
                        gpuInfo = `${info['vendor'] ?? ''} ${info['architecture'] ?? ''}`.trim() || undefined;
                    }
                }
                catch {
                    // info not available
                }
            }
        }
        catch {
            // WebGPU not available
        }
    }
    let webnn = false;
    if (typeof navigator !== 'undefined' && 'ml' in navigator) {
        try {
            const ml = navigator.ml;
            if (ml) {
                const ctx = await ml.createContext();
                webnn = ctx != null;
            }
        }
        catch {
            // WebNN not available
        }
    }
    // Determine tier
    let tier;
    if (webgpu && cores >= 8 && (memoryGiB === null || memoryGiB >= 8)) {
        tier = 'high';
    }
    else if (cores >= 4 && (memoryGiB === null || memoryGiB >= 4)) {
        tier = 'medium';
    }
    else {
        tier = 'low';
    }
    // Mobile devices get capped even if specs look good
    if (mobile && tier === 'high') {
        tier = 'medium';
    }
    const recommendedBatchSize = tier === 'high' ? 32 : tier === 'medium' ? 8 : 1;
    const recommendedConcurrency = tier === 'high' ? 4 : tier === 'medium' ? 2 : 1;
    cachedProfile = {
        tier,
        cores,
        memoryGiB,
        webgpu,
        webnn,
        recommendedBatchSize,
        recommendedConcurrency,
        mobile,
        gpuInfo,
    };
    return cachedProfile;
}
/**
 * Recommend the best quantization level for the current device.
 */
export function recommendQuantization(profile) {
    if (profile.tier === 'high' && profile.webgpu)
        return 'float16';
    if (profile.tier === 'medium')
        return 'int8';
    return 'int8'; // low-tier: most aggressive
}
/**
 * Get full model variant recommendations for the current device.
 */
export async function recommendModelVariant() {
    const profile = await getDeviceProfile();
    return {
        quantization: recommendQuantization(profile),
        executionProvider: profile.webgpu ? 'webgpu' : 'wasm',
        batchSize: profile.recommendedBatchSize,
        useWorker: profile.cores >= 4,
    };
}
/**
 * Reset the cached profile (useful for testing).
 */
export function resetDeviceProfile() {
    cachedProfile = null;
}
//# sourceMappingURL=device-profiler.js.map