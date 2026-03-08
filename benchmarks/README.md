# edgeFlow.js Benchmarks

This directory contains performance benchmarks for edgeFlow.js.

## Running Benchmarks

```bash
npm install
npm run build
npm run test -- --run tests/unit/
```

> **Note:** A dedicated `npm run benchmark` script with browser-based benchmarks is planned. The unit tests include basic tensor and scheduler performance validation.

## Benchmark Types

### 1. Tensor Operations

- Tensor creation and disposal
- Shape transformation (reshape, transpose)
- Math operations (add, matmul, softmax)

### 2. Scheduler Throughput

- Priority queue ordering under load
- Concurrent task execution
- Task cancellation overhead

### 3. Model Loading

- Cached vs uncached loads (IndexedDB)
- Chunked download with resume
- Preloading pipeline

### 4. Inference Latency

- Text generation (TinyLlama) end-to-end
- Image segmentation (SlimSAM) encode + decode

## How edgeFlow.js Adds Value

edgeFlow.js is not a replacement for inference engines like ONNX Runtime or transformers.js. It is an **orchestration layer** that adds production features on top of them:

| Scenario | Without edgeFlow.js | With edgeFlow.js |
|----------|---------------------|------------------|
| 5 concurrent model calls | Uncontrolled, may OOM | Scheduled with concurrency limits |
| Repeated inference on same input | Recomputed every time | Cached results (LRU/TTL) |
| Large model download interrupted | Start from scratch | Resume from last chunk |
| Memory leak from undisposed tensors | Silent leak | Detected and warned |

> All benchmark claims will be backed by reproducible scripts before the 1.0 release.

## Custom Benchmarks

```typescript
import { runBenchmark, benchmarkSuite } from 'edgeflowjs/tools';

const result = await runBenchmark(
  async () => {
    await model.run(input);
  },
  {
    warmupRuns: 5,
    runs: 20,
    verbose: true,
  }
);

console.log(`Average: ${result.avgTime.toFixed(2)}ms`);
console.log(`Throughput: ${result.throughput.toFixed(2)} ops/sec`);

const results = await benchmarkSuite({
  'small-model': async () => smallModel.run(input),
  'large-model': async () => largeModel.run(input),
});
```
