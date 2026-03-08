/**
 * edgeFlow.js - Pipeline Composer
 *
 * Chain multiple pipelines together to build complex multi-model workflows.
 * Each stage's output is transformed and fed as input to the next stage.
 *
 * @example
 * ```typescript
 * import { compose } from 'edgeflowjs';
 *
 * const speechTranslator = compose([
 *   { task: 'automatic-speech-recognition' },
 *   { task: 'translation', options: { srcLang: 'en', tgtLang: 'zh' } },
 * ]);
 *
 * const result = await speechTranslator.run(audioBlob);
 * // result.stages = [asrResult, translationResult]
 * // result.output  = final translation text
 * ```
 */

import { pipeline, type PipelineFactoryOptions } from '../pipelines/index.js';
import type { PipelineTask } from './types.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * A single stage in a composed pipeline.
 */
export interface CompositionStage {
  /** The pipeline task to run */
  task: PipelineTask | (string & {});
  /** Model override for this stage */
  model?: string;
  /** Extra options forwarded to `pipeline()` */
  options?: PipelineFactoryOptions;
  /**
   * Optional transform applied to the previous stage's output before it is
   * passed as input to this stage. If omitted, the raw output is forwarded.
   */
  transform?: (previousOutput: unknown) => unknown;
  /**
   * Options forwarded to the pipeline's `run()` call.
   */
  runOptions?: Record<string, unknown>;
}

/**
 * Result from running a composed pipeline.
 */
export interface CompositionResult {
  /** The final output from the last stage */
  output: unknown;
  /** Intermediate results for every stage (index-aligned with stages) */
  stages: unknown[];
  /** Total wall-clock time in milliseconds */
  totalTime: number;
  /** Per-stage timing */
  stageTimes: number[];
}

/**
 * A composed (chained) pipeline.
 */
export interface ComposedPipeline {
  /** Execute the full chain with the given initial input */
  run(input: unknown): Promise<CompositionResult>;
  /** Dispose all underlying pipeline instances */
  dispose(): void;
  /** Number of stages */
  readonly length: number;
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/**
 * Compose multiple pipeline stages into a single sequential chain.
 *
 * The output of each stage is fed as the input to the next stage. Use the
 * optional `transform` hook in a stage to reshape data between stages.
 *
 * All pipelines are lazily initialised on the first `run()` call and cached
 * for subsequent calls.
 *
 * @param stages - Ordered list of pipeline stages
 * @returns A composed pipeline that can be run end-to-end
 *
 * @example
 * ```typescript
 * const ocrPipeline = compose([
 *   { task: 'image-to-text' },
 *   {
 *     task: 'text-classification',
 *     transform: (ocrResult: any) => ocrResult.text,
 *   },
 * ]);
 *
 * const { output, stages, totalTime } = await ocrPipeline.run(imageElement);
 * ```
 */
export function compose(stages: CompositionStage[]): ComposedPipeline {
  if (stages.length === 0) {
    throw new Error('[edgeFlow.js] compose() requires at least one stage');
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let pipelineInstances: any[] | null = null;

  async function ensureInitialised() {
    if (pipelineInstances) return pipelineInstances;

    pipelineInstances = await Promise.all(
      stages.map((stage) =>
        pipeline(stage.task as Parameters<typeof pipeline>[0], {
          model: stage.model,
          ...stage.options,
        }),
      ),
    );
    return pipelineInstances;
  }

  return {
    get length() {
      return stages.length;
    },

    async run(input: unknown): Promise<CompositionResult> {
      const instances = await ensureInitialised();
      const stageResults: unknown[] = [];
      const stageTimes: number[] = [];
      let current = input;
      const wallStart = performance.now();

      for (let i = 0; i < stages.length; i++) {
        const stage = stages[i]!;
        const inst = instances[i]!;

        // Apply transform from previous stage output if provided
        if (stage.transform) {
          current = stage.transform(current);
        }

        const t0 = performance.now();
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        current = await (inst as any).run(current, stage.runOptions);
        stageTimes.push(performance.now() - t0);
        stageResults.push(current);
      }

      return {
        output: current,
        stages: stageResults,
        totalTime: performance.now() - wallStart,
        stageTimes,
      };
    },

    dispose() {
      if (pipelineInstances) {
        for (const inst of pipelineInstances) {
          if (inst && typeof inst.dispose === 'function') {
            inst.dispose();
          }
        }
        pipelineInstances = null;
      }
    },
  };
}

/**
 * Run stages in parallel (fan-out) and collect all results.
 *
 * Unlike `compose` (which is sequential), `parallel` runs every stage
 * independently with the same input and returns an array of results.
 *
 * @example
 * ```typescript
 * const analyzer = parallel([
 *   { task: 'text-classification' },
 *   { task: 'feature-extraction' },
 *   { task: 'zero-shot-classification',
 *     transform: (text) => ({ text, candidateLabels: ['news', 'sports'] }) },
 * ]);
 *
 * const results = await analyzer.run('Breaking: team wins championship');
 * ```
 */
export function parallel(
  stages: CompositionStage[],
): {
  run(input: unknown): Promise<{ outputs: unknown[]; totalTime: number }>;
  dispose(): void;
} {
  if (stages.length === 0) {
    throw new Error('[edgeFlow.js] parallel() requires at least one stage');
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let pipelineInstances: any[] | null = null;

  async function ensureInitialised() {
    if (pipelineInstances) return pipelineInstances;
    pipelineInstances = await Promise.all(
      stages.map((s) =>
        pipeline(s.task as Parameters<typeof pipeline>[0], {
          model: s.model,
          ...s.options,
        }),
      ),
    );
    return pipelineInstances;
  }

  return {
    async run(input: unknown) {
      const instances = await ensureInitialised();
      const t0 = performance.now();

      const outputs = await Promise.all(
        stages.map((stage, i) => {
          const stageInput = stage.transform ? stage.transform(input) : input;
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          return (instances[i] as any).run(stageInput, stage.runOptions);
        }),
      );

      return { outputs, totalTime: performance.now() - t0 };
    },

    dispose() {
      if (pipelineInstances) {
        for (const inst of pipelineInstances) {
          if (inst && typeof inst.dispose === 'function') {
            inst.dispose();
          }
        }
        pipelineInstances = null;
      }
    },
  };
}
