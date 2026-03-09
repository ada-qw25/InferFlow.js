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
import { pipeline } from '../pipelines/index.js';
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
export function compose(stages) {
    if (stages.length === 0) {
        throw new Error('[edgeFlow.js] compose() requires at least one stage');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let pipelineInstances = null;
    async function ensureInitialised() {
        if (pipelineInstances)
            return pipelineInstances;
        pipelineInstances = await Promise.all(stages.map((stage) => pipeline(stage.task, {
            model: stage.model,
            ...stage.options,
        })));
        return pipelineInstances;
    }
    return {
        get length() {
            return stages.length;
        },
        async run(input) {
            const instances = await ensureInitialised();
            const stageResults = [];
            const stageTimes = [];
            let current = input;
            const wallStart = performance.now();
            for (let i = 0; i < stages.length; i++) {
                const stage = stages[i];
                const inst = instances[i];
                // Apply transform from previous stage output if provided
                if (stage.transform) {
                    current = stage.transform(current);
                }
                const t0 = performance.now();
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                current = await inst.run(current, stage.runOptions);
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
export function parallel(stages) {
    if (stages.length === 0) {
        throw new Error('[edgeFlow.js] parallel() requires at least one stage');
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let pipelineInstances = null;
    async function ensureInitialised() {
        if (pipelineInstances)
            return pipelineInstances;
        pipelineInstances = await Promise.all(stages.map((s) => pipeline(s.task, {
            model: s.model,
            ...s.options,
        })));
        return pipelineInstances;
    }
    return {
        async run(input) {
            const instances = await ensureInitialised();
            const t0 = performance.now();
            const outputs = await Promise.all(stages.map((stage, i) => {
                const stageInput = stage.transform ? stage.transform(input) : input;
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                return instances[i].run(stageInput, stage.runOptions);
            }));
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
//# sourceMappingURL=composer.js.map