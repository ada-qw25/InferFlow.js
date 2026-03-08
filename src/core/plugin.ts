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

import type { PipelineConfig, Runtime, RuntimeType } from './types.js';
import { registerRuntime } from './runtime.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * A pipeline factory registered by a plugin.
 */
export interface PluginPipelineEntry {
  /** Factory that creates a pipeline instance */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  before?: (ctx: { modelId: string; inputs: any }) => any | Promise<any>;
  /** Called after inference with (model, outputs). Return modified outputs. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  after?: (ctx: { modelId: string; outputs: any }) => any | Promise<any>;
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

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

const registeredPlugins = new Map<string, EdgeFlowPlugin>();
const pluginPipelines = new Map<string, PluginPipelineEntry>();
const pluginMiddleware: PluginMiddleware[] = [];

/**
 * Register a plugin. Pipelines and backends are made available immediately.
 */
export async function registerPlugin(plugin: EdgeFlowPlugin): Promise<void> {
  if (registeredPlugins.has(plugin.name)) {
    console.warn(`[edgeFlow.js] Plugin "${plugin.name}" is already registered — skipping.`);
    return;
  }

  // Run setup hook
  if (plugin.setup) {
    await plugin.setup();
  }

  // Register pipelines
  if (plugin.pipelines) {
    for (const [task, entry] of Object.entries(plugin.pipelines)) {
      pluginPipelines.set(task, entry);
    }
  }

  // Register backends
  if (plugin.backends) {
    for (const [name, entry] of Object.entries(plugin.backends)) {
      registerRuntime(name as RuntimeType, entry.factory);
    }
  }

  // Register middleware
  if (plugin.middleware) {
    pluginMiddleware.push(...plugin.middleware);
  }

  registeredPlugins.set(plugin.name, plugin);
}

/**
 * Look up a pipeline factory registered by any plugin.
 * Returns undefined if no plugin provides this task.
 */
export function getPluginPipeline(task: string): PluginPipelineEntry | undefined {
  return pluginPipelines.get(task);
}

/**
 * Get all registered middleware.
 */
export function getPluginMiddleware(): ReadonlyArray<PluginMiddleware> {
  return pluginMiddleware;
}

/**
 * List all registered plugins.
 */
export function listPlugins(): Array<{ name: string; version: string }> {
  return Array.from(registeredPlugins.values()).map(p => ({
    name: p.name,
    version: p.version,
  }));
}

/**
 * Unregister a plugin by name.
 */
export function unregisterPlugin(name: string): boolean {
  const plugin = registeredPlugins.get(name);
  if (!plugin) return false;

  // Remove pipelines
  if (plugin.pipelines) {
    for (const task of Object.keys(plugin.pipelines)) {
      pluginPipelines.delete(task);
    }
  }

  // Remove middleware
  if (plugin.middleware) {
    for (const mw of plugin.middleware) {
      const idx = pluginMiddleware.indexOf(mw);
      if (idx !== -1) pluginMiddleware.splice(idx, 1);
    }
  }

  registeredPlugins.delete(name);
  return true;
}
