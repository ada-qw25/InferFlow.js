# Architecture Overview

edgeFlow.js is a **production orchestration layer** for browser ML inference. It does not compete with inference engines like ONNX Runtime or transformers.js — it wraps them and adds the features real applications need.

## Layer Diagram

```
┌───────────────────────────────────────────────────────────┐
│                     Your Application                       │
├───────────────────────────────────────────────────────────┤
│                      edgeFlow.js                           │
│  ┌────────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐  │
│  │  Scheduler  │ │ Memory   │ │  Composer  │ │  Plugin  │  │
│  │  (priority, │ │ Manager  │ │  (chain /  │ │  System  │  │
│  │  retry,     │ │ (scopes, │ │  parallel) │ │          │  │
│  │  circuit    │ │  GC,     │ │            │ │          │  │
│  │  breaker)   │ │  leak    │ │            │ │          │  │
│  │             │ │  detect) │ │            │ │          │  │
│  └────────────┘ └──────────┘ └───────────┘ └──────────┘  │
│  ┌────────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐  │
│  │  Worker    │ │  Cache   │ │  Device   │ │  Monitor  │  │
│  │  Pool      │ │  (LRU,   │ │  Profiler │ │  (perf,   │  │
│  │            │ │  IndexDB)│ │           │ │  alerts)  │  │
│  └────────────┘ └──────────┘ └───────────┘ └──────────┘  │
├───────────────────────────────────────────────────────────┤
│              Inference Backends (pluggable)                │
│  ┌────────────┐ ┌─────────────────────┐ ┌──────────────┐ │
│  │ ONNX       │ │ transformers.js     │ │ Custom       │ │
│  │ Runtime    │ │ Adapter             │ │ Backend      │ │
│  └────────────┘ └─────────────────────┘ └──────────────┘ │
└───────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Backend Agnosticism

edgeFlow.js does not lock you into a single inference engine. The `Runtime` interface is intentionally minimal:

```typescript
interface Runtime {
  name: RuntimeType;
  capabilities: RuntimeCapabilities;
  initialize(): Promise<void>;
  isAvailable(): Promise<boolean>;
  loadModel(data: ArrayBuffer, options?): Promise<LoadedModel>;
  run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]>;
  dispose(): void;
}
```

Any engine that can implement this interface can be plugged in. The built-in ONNX Runtime backend and the transformers.js adapter are both implementations of this interface.

### Scheduler-First Architecture

Every inference call goes through the `InferenceScheduler`:

1. Tasks are enqueued with a priority (`critical`, `high`, `normal`, `low`).
2. The scheduler respects per-model and global concurrency limits.
3. Failed tasks are optionally retried with exponential backoff.
4. A circuit breaker per model prevents cascading failures.

### Memory Scopes

Inspired by RAII patterns, `withMemoryScope()` ensures tensors and models are automatically disposed:

```typescript
const result = await withMemoryScope(async (scope) => {
  const a = scope.track(tensor([1, 2, 3]));
  const b = scope.track(tensor([4, 5, 6]));
  const output = add(a, b);
  return scope.keep(output); // keep this, dispose the rest
});
```

### Plugin Extensibility

Third-party plugins can register new pipeline tasks, backends, and middleware without modifying edgeFlow.js source. See [Plugin System](/guide/plugins).
