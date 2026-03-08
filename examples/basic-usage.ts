/**
 * edgeFlow.js — Basic Usage Example
 *
 * Demonstrates the core APIs: pipeline creation, text generation,
 * image segmentation, scheduling, and memory management.
 */

import {
  pipeline,
  getScheduler,
  getMemoryStats,
  withMemoryScope,
  configureScheduler,
  gc,
} from 'edgeflowjs';

// ---------------------------------------------------------------------------
// 1. Text Generation (production-ready pipeline)
// ---------------------------------------------------------------------------

async function textGenerationExample() {
  const generator = await pipeline('text-generation');

  // Simple generation
  const result = await generator.run('Once upon a time', {
    maxNewTokens: 50,
    temperature: 0.8,
  });

  console.log('Generated:', result);

  // Streaming
  for await (const event of generator.stream('Hello, ')) {
    process.stdout.write(event.token);
    if (event.done) break;
  }

  generator.dispose();
}

// ---------------------------------------------------------------------------
// 2. Task Scheduling
// ---------------------------------------------------------------------------

async function schedulerExample() {
  configureScheduler({
    maxConcurrentTasks: 2,
    maxConcurrentPerModel: 1,
    defaultTimeout: 30_000,
  });

  const scheduler = getScheduler();

  // Schedule tasks with different priorities
  const high = scheduler.schedule('model-a', async () => 'high-result', 'high');
  const low = scheduler.schedule('model-a', async () => 'low-result', 'low');
  const critical = scheduler.schedule(
    'model-b',
    async () => 'critical-result',
    'critical',
  );

  const results = await Promise.all([high.wait(), low.wait(), critical.wait()]);
  console.log('Scheduler results:', results);

  // Cancel a pending task
  const task = scheduler.schedule('model-a', async () => 'will-cancel');
  task.cancel();
  console.log('Task status:', task.status); // 'cancelled'
}

// ---------------------------------------------------------------------------
// 3. Memory Management
// ---------------------------------------------------------------------------

async function memoryExample() {
  const result = await withMemoryScope(async (scope) => {
    const gen = await pipeline('text-generation');
    scope.track(gen);

    const output = await gen.run('test', { maxNewTokens: 10 });
    return output;
    // gen is auto-disposed when scope exits
  });

  console.log('Scoped result:', result);
  console.log('Memory stats:', getMemoryStats());

  gc();
}

// ---------------------------------------------------------------------------
// Run all examples
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== edgeFlow.js Basic Usage ===\n');

  await textGenerationExample();
  await schedulerExample();
  await memoryExample();
}

main().catch(console.error);
