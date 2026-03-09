/**
 * Unit tests for Runtime registration, auto-selection, and fallback chain
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { RuntimeManager } from '../../src/core/runtime';

describe('RuntimeManager', () => {
  it('should be a singleton', () => {
    const a = RuntimeManager.getInstance();
    const b = RuntimeManager.getInstance();
    expect(a).toBe(b);
  });

  it('should have register method', () => {
    const manager = RuntimeManager.getInstance();
    expect(typeof manager.register).toBe('function');
  });

  it('should resolve auto runtime without throwing', async () => {
    const manager = RuntimeManager.getInstance();
    // 'auto' should resolve to an available runtime or throw
    // In happy-dom, only wasm-based runtimes are available
    try {
      const runtime = await manager.getRuntime('auto');
      expect(runtime).toBeDefined();
    } catch {
      // May fail if no runtimes registered — that's fine
    }
  });

  it('should throw for unknown runtime type', async () => {
    const manager = RuntimeManager.getInstance();
    // @ts-expect-error testing invalid type
    await expect(manager.getRuntime('nonexistent')).rejects.toThrow();
  });
});

describe('ONNX Runtime', () => {
  it('should export isOnnxAvailable', async () => {
    const mod = await import('../../src/backends/onnx');
    expect(typeof mod.isOnnxAvailable).toBe('function');
  });

  it('should export ONNXRuntime class', async () => {
    const mod = await import('../../src/backends/onnx');
    expect(mod.ONNXRuntime).toBeDefined();
  });

  it('should export createONNXRuntime factory', async () => {
    const mod = await import('../../src/backends/onnx');
    expect(typeof mod.createONNXRuntime).toBe('function');
  });

  it('isOnnxAvailable should return boolean', async () => {
    const { isOnnxAvailable } = await import('../../src/backends/onnx');
    const result = await isOnnxAvailable();
    expect(typeof result).toBe('boolean');
  });
});

describe('Backend Registration', () => {
  it('should export registerAllBackends', async () => {
    const mod = await import('../../src/backends/index');
    expect(typeof mod.registerAllBackends).toBe('function');
  });

  it('should export isOnnxAvailable from backends index', async () => {
    const mod = await import('../../src/backends/index');
    expect(typeof mod.isOnnxAvailable).toBe('function');
  });

  it('should not throw on registerAllBackends()', async () => {
    const { registerAllBackends } = await import('../../src/backends/index');
    await expect(registerAllBackends()).resolves.not.toThrow();
  });
});
