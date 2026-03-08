import { defineConfig } from 'vitepress';

export default defineConfig({
  title: 'edgeFlow.js',
  description: 'Production runtime for browser ML inference',
  base: '/',

  themeConfig: {
    logo: '/logo.svg',
    nav: [
      { text: 'Guide', link: '/guide/installation' },
      { text: 'API', link: '/api/pipeline' },
      { text: 'Cookbook', link: '/cookbook/transformers-adapter' },
      {
        text: 'v0.1.0',
        items: [
          { text: 'Changelog', link: '/changelog' },
          { text: 'GitHub', link: 'https://github.com/s-zx/edgeflow.js' },
        ],
      },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Quick Start', link: '/guide/quickstart' },
            { text: 'Core Concepts', link: '/guide/concepts' },
          ],
        },
        {
          text: 'Architecture',
          items: [
            { text: 'Overview', link: '/guide/architecture' },
            { text: 'Plugin System', link: '/guide/plugins' },
            { text: 'Device Profiling', link: '/guide/device-profiling' },
          ],
        },
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'pipeline()', link: '/api/pipeline' },
            { text: 'compose() / parallel()', link: '/api/composer' },
            { text: 'Tensor', link: '/api/tensor' },
            { text: 'Tokenizer', link: '/api/tokenizer' },
            { text: 'Model Loader', link: '/api/model-loader' },
            { text: 'Scheduler', link: '/api/scheduler' },
            { text: 'Memory', link: '/api/memory' },
          ],
        },
      ],
      '/cookbook/': [
        {
          text: 'Recipes',
          items: [
            { text: 'transformers.js Adapter', link: '/cookbook/transformers-adapter' },
            { text: 'Pipeline Composition', link: '/cookbook/composition' },
            { text: 'Offline-First App', link: '/cookbook/offline' },
            { text: 'Multi-Model Dashboard', link: '/cookbook/multi-model' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/s-zx/edgeflow.js' },
      { icon: 'npm', link: 'https://www.npmjs.com/package/edgeflowjs' },
    ],

    search: { provider: 'local' },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright 2026 edgeFlow.js Contributors',
    },
  },
});
