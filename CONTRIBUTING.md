# Contributing to edgeFlow.js

Thank you for your interest in contributing to edgeFlow.js! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/s-zx/edgeflow.js.git
cd edgeflow.js

# Install dependencies
npm install

# Build the project
npm run build

# Run tests
npm run test:unit

# Start development mode (watch)
npm run dev
```

## Project Structure

```
src/
├── core/           # Runtime, scheduler, memory, tensor, types
├── backends/       # ONNX Runtime (production), WebGPU/WebNN (planned)
├── pipelines/      # Task pipelines (text-generation, image-segmentation, etc.)
├── utils/          # Tokenizer, preprocessor, cache, model-loader, hub
└── tools/          # Quantization, benchmark, debugger, monitor
```

## How to Contribute

### Reporting Bugs

Open an issue using the bug report template. Include:
- A minimal code reproduction
- Browser and OS information
- edgeFlow.js version

### Suggesting Features

Open an issue using the feature request template describing:
- The problem you're trying to solve
- Your proposed solution
- Alternatives you've considered

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run checks: `npm run lint && npx tsc --noEmit && npm run test:unit`
5. Commit with a descriptive message
6. Push and open a pull request

### Good First Issues

Look for issues labeled `good first issue`. These are scoped tasks ideal for newcomers:
- Adding tests for uncovered modules
- Improving error messages
- Adding examples
- Documentation improvements

## Code Standards

- **TypeScript strict mode** — all strict options are enabled
- **No `any`** — use proper types; `unknown` if truly dynamic
- **ESM only** — use `.js` extensions in imports
- **No console.log in library code** — use the event system or `console.warn` for important warnings
- **Dispose pattern** — all resources must be disposable to prevent memory leaks

## Testing

```bash
npm run test:unit        # Run unit tests
npm run test:integration # Run integration tests
npm run test:coverage    # Generate coverage report
npm run test:watch       # Watch mode
```

Tests use [Vitest](https://vitest.dev/). Place tests in:
- `tests/unit/` — for isolated unit tests
- `tests/integration/` — for pipeline/backend integration tests
- `tests/e2e/` — for browser-based tests

## Architecture Decisions

edgeFlow.js is designed as an **orchestration layer**, not an inference engine. Key principles:

1. **Backend agnostic** — work with any inference engine (ONNX Runtime, transformers.js, custom)
2. **Production-first** — scheduling, memory management, error recovery matter more than model count
3. **Honest API** — experimental features are clearly labeled, not presented as production-ready
4. **Plugin-friendly** — custom pipelines and backends can be registered at runtime

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
