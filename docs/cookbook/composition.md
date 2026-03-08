# Pipeline Composition

Chain multiple ML models together to build complex workflows. No other browser ML framework supports this natively.

## Sequential Composition

Process data through a sequence of models. Each stage's output feeds the next stage's input.

```typescript
import { compose } from 'edgeflowjs';

const analyzer = compose([
  { task: 'automatic-speech-recognition' },
  {
    task: 'text-classification',
    transform: (asrResult: any) => asrResult.text,
  },
]);

const { output, stages, totalTime } = await analyzer.run(audioBlob);
// stages[0] = ASR result
// stages[1] = classification result
// output    = final classification
```

## Parallel Composition

Run multiple models on the same input simultaneously:

```typescript
import { parallel } from 'edgeflowjs';

const multiAnalysis = parallel([
  { task: 'text-classification' },
  { task: 'feature-extraction' },
  {
    task: 'zero-shot-classification',
    transform: (text) => ({
      text,
      candidateLabels: ['tech', 'sports', 'politics'],
    }),
  },
]);

const { outputs, totalTime } = await multiAnalysis.run('Breaking news today');
// outputs[0] = classification result
// outputs[1] = embedding result
// outputs[2] = zero-shot result
```

## Transform Functions

Use `transform` to reshape data between stages:

```typescript
compose([
  { task: 'image-segmentation' },
  {
    task: 'image-classification',
    transform: (segResult: any) => {
      // Extract the largest segment and classify it
      return segResult.masks[0].croppedImage;
    },
  },
]);
```

## Resource Management

Composed pipelines support `dispose()` to clean up all underlying models:

```typescript
const pipeline = compose([...]);
const result = await pipeline.run(input);

// When done
pipeline.dispose(); // disposes all stage pipelines
```

## API

| Function | Description |
|----------|-------------|
| `compose(stages)` | Chain stages sequentially (output → input) |
| `parallel(stages)` | Run stages concurrently on the same input |
| `ComposedPipeline.run(input)` | Execute the full chain |
| `ComposedPipeline.dispose()` | Clean up all pipelines |
| `ComposedPipeline.length` | Number of stages |
