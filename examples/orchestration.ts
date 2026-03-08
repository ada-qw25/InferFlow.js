/**
 * edgeFlow.js — Orchestration Example
 *
 * Demonstrates what makes edgeFlow.js unique: production orchestration
 * features that no other browser ML framework provides.
 */

import {
  pipeline,
  configureScheduler,
  getScheduler,
  getMemoryStats,
  withMemoryScope,
  preloadModel,
  getPreloadStatus,
  isModelCached,
  loadModelData,
  type TextGenerationPipeline,
} from 'edgeflowjs';

// ---------------------------------------------------------------------------
// 1. Concurrent Model Management
// ---------------------------------------------------------------------------

async function concurrentModelsExample() {
  console.log('--- Concurrent Model Management ---');

  // Limit concurrency to prevent OOM on constrained devices
  configureScheduler({
    maxConcurrentTasks: 4,
    maxConcurrentPerModel: 1,
  });

  const scheduler = getScheduler();

  // Schedule 10 tasks — scheduler ensures only 4 run at a time
  const tasks = Array.from({ length: 10 }, (_, i) =>
    scheduler.schedule(
      `model-${i % 3}`, // distribute across 3 "models"
      async () => {
        await new Promise((r) => setTimeout(r, 100));
        return `result-${i}`;
      },
      i < 3 ? 'high' : 'normal',
    ),
  );

  const results = await Promise.all(tasks.map((t) => t.wait()));
  console.log(`Completed ${results.length} tasks`);
  console.log('Stats:', scheduler.getStats());
}

// ---------------------------------------------------------------------------
// 2. Smart Model Caching & Preloading
// ---------------------------------------------------------------------------

async function cachingExample() {
  console.log('\n--- Smart Caching & Preloading ---');

  const modelUrl = 'https://huggingface.co/example/model/resolve/main/model.onnx';

  // Preload models in the background while the user interacts with the UI
  preloadModel(modelUrl, { priority: 10 });
  console.log('Preload status:', getPreloadStatus(modelUrl));

  // Check if a model is already cached (IndexedDB)
  const cached = await isModelCached(modelUrl);
  console.log('Is cached:', cached);

  // Download with resume support — if interrupted, picks up where it left off
  try {
    const data = await loadModelData(modelUrl, {
      resumable: true,
      chunkSize: 5 * 1024 * 1024,
      onProgress: (p) => {
        console.log(`Download: ${p.percent.toFixed(1)}% — ${(p.speed / 1e6).toFixed(1)} MB/s`);
      },
    });
    console.log(`Downloaded ${data.byteLength} bytes`);
  } catch {
    console.log('Download failed (expected in example)');
  }
}

// ---------------------------------------------------------------------------
// 3. Memory Scopes — Automatic Cleanup
// ---------------------------------------------------------------------------

async function memoryScopeExample() {
  console.log('\n--- Memory Scopes ---');

  const before = getMemoryStats();
  console.log('Before:', before);

  await withMemoryScope(async (scope) => {
    // Everything tracked in the scope is automatically disposed on exit
    const gen = (await pipeline('text-generation')) as TextGenerationPipeline;
    scope.track(gen);

    // Nested scopes for fine-grained control
    const innerResult = await withMemoryScope(async (inner) => {
      // inner resources cleaned up first, then outer
      return 'inner-done';
    });

    console.log('Inner scope result:', innerResult);
  });

  const after = getMemoryStats();
  console.log('After:', after);
}

// ---------------------------------------------------------------------------
// 4. Task Cancellation & Timeouts
// ---------------------------------------------------------------------------

async function cancellationExample() {
  console.log('\n--- Cancellation & Timeouts ---');

  const scheduler = getScheduler();

  // Schedule with timeout — auto-fails if takes too long
  const timedTask = scheduler.scheduleWithTimeout(
    'slow-model',
    async () => {
      await new Promise((r) => setTimeout(r, 60_000));
      return 'done';
    },
    5_000, // 5 second timeout
    'normal',
  );

  // Cancel programmatically
  const cancelableTask = scheduler.schedule(
    'model-x',
    async () => {
      await new Promise((r) => setTimeout(r, 10_000));
      return 'result';
    },
  );

  // User navigates away — cancel pending work
  cancelableTask.cancel();
  console.log('Cancelled task status:', cancelableTask.status);

  // Timeout will fire for timedTask
  try {
    await timedTask.wait();
  } catch (e) {
    console.log('Timed out as expected:', (e as Error).message);
  }
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== edgeFlow.js Orchestration Demo ===\n');
  await concurrentModelsExample();
  await cachingExample();
  await memoryScopeExample();
  await cancellationExample();
}

main().catch(console.error);
