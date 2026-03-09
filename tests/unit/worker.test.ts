/**
 * Unit tests for InferenceWorker and WorkerPool
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Worker is not available in happy-dom, so we test the serialization layer
// and the pool logic that doesn't require a real Worker instance.
import {
  serializeTensor,
  deserializeTensorSync,
  type SerializedTensor,
} from '../../src/core/worker';
import { EdgeFlowTensor } from '../../src/core/tensor';

describe('Tensor Serialization', () => {
  it('should serialize a tensor to transferable format', () => {
    const tensor = new EdgeFlowTensor([1, 2, 3, 4], [2, 2]);
    const serialized = serializeTensor(tensor);

    expect(serialized.data).toBeInstanceOf(ArrayBuffer);
    expect(serialized.shape).toEqual([2, 2]);
    expect(serialized.dtype).toBe('float32');
    expect(serialized.data.byteLength).toBe(4 * 4);
  });

  it('should produce a detached copy of the ArrayBuffer', () => {
    const tensor = new EdgeFlowTensor([10, 20], [2]);
    const serialized = serializeTensor(tensor);

    const view = new Float32Array(serialized.data);
    expect(view[0]).toBe(10);
    expect(view[1]).toBe(20);
  });

  it('should deserialize back to tensor synchronously', () => {
    const original = new EdgeFlowTensor([5, 6, 7], [3]);
    const serialized = serializeTensor(original);

    const restored = deserializeTensorSync(serialized, EdgeFlowTensor);

    expect(restored.shape).toEqual([3]);
    expect(restored.dtype).toBe('float32');
    expect(restored.toFloat32Array()[0]).toBeCloseTo(5);
    expect(restored.toFloat32Array()[2]).toBeCloseTo(7);
  });

  it('should handle large tensors', () => {
    const data = new Array(10000).fill(0).map((_, i) => i);
    const tensor = new EdgeFlowTensor(data, [100, 100]);
    const serialized = serializeTensor(tensor);

    expect(serialized.shape).toEqual([100, 100]);
    expect(serialized.data.byteLength).toBe(10000 * 4);
  });

  it('should handle 1-element tensor', () => {
    const tensor = new EdgeFlowTensor([42], [1]);
    const serialized = serializeTensor(tensor);

    const restored = deserializeTensorSync(serialized, EdgeFlowTensor);
    expect(restored.toFloat32Array()[0]).toBeCloseTo(42);
  });
});

describe('WorkerHealthState (import check)', () => {
  it('should export WorkerHealthState type', async () => {
    const mod = await import('../../src/core/worker');
    // InferenceWorker and WorkerPool are exported
    expect(mod.InferenceWorker).toBeDefined();
    expect(mod.WorkerPool).toBeDefined();
  });
});
