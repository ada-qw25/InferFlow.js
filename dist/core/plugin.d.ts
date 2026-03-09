/**
 * edgeFlow.js - Plugin System
 *
 * Register custom pipelines, backends, and middleware via plugins.
 *
 * @example
 * ```typescript
 * import { registerPlugin } from 'edgeflowjs';
 *
 * registerPlugin({
 *   name: 'edgeflow-plugin-whisper',
 *   version: '1.0.0',
 *   pipelines: {
 *     'whisper-transcribe': {
 *       factory: (config) => new WhisperPipeline(config),
 *     },
 *   },
 * });
 *
 * // Now available via pipeline('whisper-transcribe')
 * ```
 */
import type { PipelineConfig, Runtime } from './types.js';
/**
 * A pipeline factory registered by a plugin.
 */
export interface PluginPipelineEntry {
    /** Factory that creates a pipeline instance */
    factory: (config: PipelineConfig) => any;
    /** Optional description */
    description?: string;
}
/**
 * A backend registered by a plugin.
 */
export interface PluginBackendEntry {
    /** Factory that creates a runtime instance */
    factory: () => Runtime;
    /** Optional description */
    description?: string;
}
/**
 * Middleware that runs before/after inference.
 */
export interface PluginMiddleware {
    /** Unique name */
    name: string;
    /** Called before inference with (model, inputs). Return modified inputs. */
    before?: (ctx: {
        modelId: string;
        inputs: any;
    }) => any | Promise<any>;
    /** Called after inference with (model, outputs). Return modified outputs. */
    after?: (ctx: {
        modelId: string;
        outputs: any;
    }) => any | Promise<any>;
}
/**
 * Plugin definition.
 */
export interface EdgeFlowPlugin {
    /** Unique plugin name (e.g. 'edgeflow-plugin-whisper') */
    name: string;
    /** Plugin version (semver) */
    version: string;
    /** Pipelines contributed by this plugin */
    pipelines?: Record<string, PluginPipelineEntry>;
    /** Backends contributed by this plugin */
    backends?: Record<string, PluginBackendEntry>;
    /** Middleware contributed by this plugin */
    middleware?: PluginMiddleware[];
    /** Called once when the plugin is registered */
    setup?: () => void | Promise<void>;
}
/**
 * Register a plugin. Pipelines and backends are made available immediately.
 */
export declare function registerPlugin(plugin: EdgeFlowPlugin): Promise<void>;
/**
 * Look up a pipeline factory registered by any plugin.
 * Returns undefined if no plugin provides this task.
 */
export declare function getPluginPipeline(task: string): PluginPipelineEntry | undefined;
/**
 * Get all registered middleware.
 */
export declare function getPluginMiddleware(): ReadonlyArray<PluginMiddleware>;
/**
 * List all registered plugins.
 */
export declare function listPlugins(): Array<{
    name: string;
    version: string;
}>;
/**
 * Unregister a plugin by name.
 */
export declare function unregisterPlugin(name: string): boolean;
//# sourceMappingURL=plugin.d.ts.map