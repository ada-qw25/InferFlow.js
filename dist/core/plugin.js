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
import { registerRuntime } from './runtime.js';
// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
const registeredPlugins = new Map();
const pluginPipelines = new Map();
const pluginMiddleware = [];
/**
 * Register a plugin. Pipelines and backends are made available immediately.
 */
export async function registerPlugin(plugin) {
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
            registerRuntime(name, entry.factory);
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
export function getPluginPipeline(task) {
    return pluginPipelines.get(task);
}
/**
 * Get all registered middleware.
 */
export function getPluginMiddleware() {
    return pluginMiddleware;
}
/**
 * List all registered plugins.
 */
export function listPlugins() {
    return Array.from(registeredPlugins.values()).map(p => ({
        name: p.name,
        version: p.version,
    }));
}
/**
 * Unregister a plugin by name.
 */
export function unregisterPlugin(name) {
    const plugin = registeredPlugins.get(name);
    if (!plugin)
        return false;
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
            if (idx !== -1)
                pluginMiddleware.splice(idx, 1);
        }
    }
    registeredPlugins.delete(name);
    return true;
}
//# sourceMappingURL=plugin.js.map