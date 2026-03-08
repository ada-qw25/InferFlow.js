# Device Profiling

edgeFlow.js can automatically profile the current device and recommend optimal model variants.

## Quick Start

```typescript
import { getDeviceProfile, recommendModelVariant } from 'edgeflowjs';

const profile = await getDeviceProfile();
console.log(profile.tier);   // 'high' | 'medium' | 'low'
console.log(profile.webgpu); // true | false
console.log(profile.cores);  // e.g. 8

const rec = await recommendModelVariant();
console.log(rec.quantization);      // 'float16' | 'int8'
console.log(rec.executionProvider);  // 'webgpu' | 'wasm'
console.log(rec.batchSize);          // e.g. 32
```

## Device Tiers

| Tier | Criteria | Example Devices |
|------|----------|-----------------|
| **high** | WebGPU + 8+ cores + 8+ GB RAM | Desktop with dedicated GPU |
| **medium** | 4+ cores + 4+ GB RAM | Modern laptop, high-end mobile |
| **low** | Everything else | Older devices, low-end mobile |

## Using with Pipelines

```typescript
const profile = await getDeviceProfile();
const rec = await recommendModelVariant();

const classifier = await pipeline('text-classification', {
  model: `my-model-${rec.quantization}`,
  runtime: rec.executionProvider === 'webgpu' ? 'webgpu' : 'wasm',
});
```

## API

| Function | Description |
|----------|-------------|
| `getDeviceProfile()` | Returns `DeviceProfile` with tier, cores, memory, GPU info |
| `recommendQuantization(profile)` | Returns best `QuantizationType` for the given profile |
| `recommendModelVariant()` | Returns full `ModelRecommendation` (quant, provider, batch, worker) |
| `resetDeviceProfile()` | Clears the cached profile (useful for testing) |
