# Plugin System

edgeFlow.js supports plugins that register custom pipelines, backends, and middleware at runtime.

## Creating a Plugin

```typescript
import { registerPlugin, type EdgeFlowPlugin } from 'edgeflowjs';

const myPlugin: EdgeFlowPlugin = {
  name: 'edgeflow-plugin-whisper',
  version: '1.0.0',

  pipelines: {
    'whisper-transcribe': {
      factory: (config) => new WhisperPipeline(config),
      description: 'Transcribe audio using Whisper',
    },
  },

  setup() {
    console.log('Whisper plugin loaded');
  },
};

await registerPlugin(myPlugin);
```

After registration, the pipeline is available via:

```typescript
const transcriber = await pipeline('whisper-transcribe');
```

## Plugin Structure

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Unique plugin identifier |
| `version` | `string` | Semver version |
| `pipelines` | `Record<string, PluginPipelineEntry>` | Pipeline factories keyed by task name |
| `backends` | `Record<string, PluginBackendEntry>` | Backend factories keyed by runtime name |
| `middleware` | `PluginMiddleware[]` | Before/after inference hooks |
| `setup` | `() => void \| Promise<void>` | One-time initialisation |

## Middleware

Middleware runs before and/or after every inference call:

```typescript
registerPlugin({
  name: 'logger',
  version: '1.0.0',
  middleware: [{
    name: 'inference-logger',
    before: (ctx) => {
      console.log(`Running inference on ${ctx.modelId}`);
      return ctx.inputs;
    },
    after: (ctx) => {
      console.log(`Inference complete on ${ctx.modelId}`);
      return ctx.outputs;
    },
  }],
});
```

## Managing Plugins

```typescript
import { listPlugins, unregisterPlugin } from 'edgeflowjs';

console.log(listPlugins());
// [{ name: 'edgeflow-plugin-whisper', version: '1.0.0' }]

unregisterPlugin('edgeflow-plugin-whisper');
```
