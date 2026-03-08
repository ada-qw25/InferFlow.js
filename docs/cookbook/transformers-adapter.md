# transformers.js Adapter

Use edgeFlow.js as an orchestration layer on top of [transformers.js](https://huggingface.co/docs/transformers.js) to access 1000+ HuggingFace models with scheduling, caching, and memory management.

## Installation

```bash
npm install edgeflowjs @xenova/transformers
```

## Setup

```typescript
import { pipeline as tfPipeline } from '@xenova/transformers';
import { useTransformersBackend, pipeline, configureScheduler } from 'edgeflowjs';

// Register transformers.js as the inference backend
useTransformersBackend({
  pipelineFactory: tfPipeline,
  device: 'webgpu',    // GPU acceleration
  dtype: 'fp16',       // Half precision for speed
});

// Optional: configure edgeFlow.js scheduling
configureScheduler({
  maxConcurrentTasks: 4,
  maxConcurrentPerModel: 1,
  maxRetries: 2,
  circuitBreaker: true,
});
```

## Usage

After setup, use the standard edgeFlow.js API. All inference calls go through transformers.js but with edgeFlow.js orchestration:

```typescript
// Sentiment analysis with scheduling + caching
const classifier = await pipeline('text-classification', {
  model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
});

const result = await classifier.run('I love edgeFlow.js!');
```

## Why Use the Adapter?

| Scenario | transformers.js alone | With edgeFlow.js adapter |
|----------|----------------------|--------------------------|
| Run 5 models at once | Uncontrolled memory | Scheduled with limits |
| Same input repeated | Recomputed | Cached |
| Model download interrupted | Restart | Resume from last chunk |
| Task cancellation | Not possible | `task.cancel()` |
| Performance monitoring | Manual | Built-in dashboard |

## Advanced: Direct Pipeline Access

For advanced use, access the transformers.js pipeline directly:

```typescript
import { getTransformersAdapter } from 'edgeflowjs';

const adapter = getTransformersAdapter();
if (adapter) {
  const modelId = await adapter.loadPipeline(
    'text-classification',
    'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
  );

  const result = await adapter.runDirect(modelId, 'Hello world');
}
```
