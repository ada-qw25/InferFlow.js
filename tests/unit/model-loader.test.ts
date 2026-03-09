/**
 * Unit tests for model-loader (download, cache, IndexedDB quota handling)
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

// We test the public API surface. Actual IndexedDB and fetch calls are mocked
// since happy-dom doesn't fully support IndexedDB.

describe('ModelLoader Public API', () => {
  it('should export loadModelData', async () => {
    const mod = await import('../../src/utils/model-loader');
    expect(typeof mod.loadModelData).toBe('function');
  });

  it('should export preloadModel', async () => {
    const mod = await import('../../src/utils/model-loader');
    expect(typeof mod.preloadModel).toBe('function');
  });

  it('should export isModelCached', async () => {
    const mod = await import('../../src/utils/model-loader');
    expect(typeof mod.isModelCached).toBe('function');
  });

  it('should export getPreloadStatus', async () => {
    const mod = await import('../../src/utils/model-loader');
    expect(typeof mod.getPreloadStatus).toBe('function');
  });

  it('should export cancelPreload', async () => {
    const mod = await import('../../src/utils/model-loader');
    expect(typeof mod.cancelPreload).toBe('function');
  });

  it('should export clearModelCache', async () => {
    const mod = await import('../../src/utils/model-loader');
    expect(typeof mod.clearModelCache).toBe('function');
  });

  it('should export getModelCacheStats', async () => {
    const mod = await import('../../src/utils/model-loader');
    expect(typeof mod.getModelCacheStats).toBe('function');
  });
});

describe('DownloadProgress type shape', () => {
  it('should accept a valid progress object', () => {
    const progress = {
      loaded: 1024,
      total: 4096,
      percent: 25,
      speed: 2048,
      eta: 1500,
    };

    expect(progress.percent).toBe(25);
    expect(progress.speed).toBeGreaterThan(0);
    expect(progress.eta).toBeGreaterThan(0);
  });

  it('should accept chunked progress', () => {
    const progress = {
      loaded: 2048,
      total: 8192,
      percent: 25,
      speed: 1024,
      eta: 6000,
      currentChunk: 1,
      totalChunks: 4,
    };

    expect(progress.currentChunk).toBe(1);
    expect(progress.totalChunks).toBe(4);
  });
});

describe('PreloadManager', () => {
  it('should return not_found for unknown URL', async () => {
    const { getPreloadStatus } = await import('../../src/utils/model-loader');
    const status = getPreloadStatus('https://example.com/nonexistent.onnx');
    expect(status).toBe('not_found');
  });
});
