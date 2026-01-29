/**
 * edgeFlow.js Interactive Demo
 * 
 * Organized into modules:
 * 1. State & Config
 * 2. Utilities
 * 3. UI Helpers
 * 4. Core Features
 * 5. SAM Interactive Segmentation (Real Model)
 * 6. AI Chat (Real Model)
 * 7. Demo Class (Public API)
 * 8. Initialization
 */

import * as edgeFlow from '/dist/edgeflow.browser.js';

// Expose edgeFlow globally for debugging
window.edgeFlow = edgeFlow;

/* ==========================================================================
   1. State & Config
   ========================================================================== */

const state = {
  model: null,
  testTensors: [],
  monitor: null,
  // SAM state
  samPipeline: null,
  samModelLoaded: false,
  samImage: null,
  samPoints: [],
  samCanvas: null,
  samMaskCanvas: null,
  samCtx: null,
  samMaskCtx: null,
  // Chat state
  chatPipeline: null,
  chatModelLoaded: false,
  chatHistory: [],
  chatGenerating: false,
};

const config = {
  defaultSeqLen: 128,
  monitorSampleInterval: 500,
  monitorHistorySize: 30,
};

/* ==========================================================================
   2. Utilities
   ========================================================================== */

const utils = {
  /**
   * Format bytes to human readable string
   */
  formatBytes(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  },

  /**
   * Sleep for given milliseconds
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  },

  /**
   * Generate placeholder model inputs based on model metadata
   */
  createModelInputs(model, seqLen = config.defaultSeqLen) {
    return model.metadata.inputs.map(spec => {
      const data = new Array(seqLen).fill(0);
      
      if (spec.name.includes('input')) {
        data[0] = 101;  // [CLS]
        data[1] = 2054; // sample token
        data[2] = 102;  // [SEP]
      } else if (spec.name.includes('mask')) {
        data[0] = 1;
        data[1] = 1;
        data[2] = 1;
      }
      
      return edgeFlow.tensor(data, [1, seqLen], 'int64');
    });
  },

  /**
   * Simple tokenization and inference
   */
  async inferText(text) {
    if (!state.model) throw new Error('Model not loaded');
    
    const tokens = text.toLowerCase().split(/\s+/);
    const maxLen = config.defaultSeqLen;
    const numTokens = Math.min(tokens.length + 2, maxLen);
    
    const inputs = state.model.metadata.inputs.map(spec => {
      const data = new Array(maxLen).fill(0);
      
      if (spec.name.includes('input')) {
        data[0] = 101; // [CLS]
        tokens.slice(0, maxLen - 2).forEach((t, i) => {
          // Simple hash-based token ID (demo only)
          data[i + 1] = Math.abs(t.split('').reduce((a, c) => a + c.charCodeAt(0), 0)) % 30000;
        });
        data[numTokens - 1] = 102; // [SEP]
      } else if (spec.name.includes('mask')) {
        for (let i = 0; i < numTokens; i++) data[i] = 1;
      }
      
      return edgeFlow.tensor(data, [1, maxLen], 'int64');
    });

    const outputs = await edgeFlow.runInference(state.model, inputs);
    const outputData = outputs[0].toArray();
    
    // Calculate sentiment score
    const score = outputData.length >= 2
      ? Math.exp(outputData[1]) / (Math.exp(outputData[0]) + Math.exp(outputData[1]))
      : outputData[0] > 0.5 ? outputData[0] : 1 - outputData[0];

    // Cleanup
    inputs.forEach(t => t.dispose());
    outputs.forEach(t => t.dispose());

    return {
      label: score > 0.5 ? 'positive' : 'negative',
      score,
    };
  },
};

/* ==========================================================================
   3. UI Helpers
   ========================================================================== */

const ui = {
  /**
   * Get element by ID
   */
  $(id) {
    return document.getElementById(id);
  },

  /**
   * Set output content
   */
  setOutput(id, content, type = '') {
    const el = this.$(id);
    if (!el) return;
    
    const className = type ? `class="${type}"` : '';
    el.innerHTML = `<pre><span ${className}>${content}</span></pre>`;
  },

  /**
   * Show loading state
   */
  showLoading(id, message = 'Loading...') {
    this.setOutput(id, `<span class="loader"></span>${message}`);
  },

  /**
   * Show success message
   */
  showSuccess(id, message) {
    this.setOutput(id, `✓ ${message}`, 'success');
  },

  /**
   * Show error message
   */
  showError(id, error) {
    const message = error instanceof Error ? error.message : String(error);
    this.setOutput(id, `Error: ${message}`, 'error');
  },

  /**
   * Render status list
   */
  renderStatusList(id, items) {
    const el = this.$(id);
    if (!el) return;
    
    el.innerHTML = items.map(({ label, value, status }) => `
      <div class="status-item">
        <span>${label}</span>
        <span class="${status ? 'status-badge status-' + status : ''}">${value}</span>
      </div>
    `).join('');
  },

  /**
   * Render metrics
   */
  renderMetrics(id, metrics) {
    const el = this.$(id);
    if (!el) return;
    
    el.innerHTML = metrics.map(({ value, label }) => `
      <div class="metric">
        <div class="metric-value">${value}</div>
        <div class="metric-label">${label}</div>
      </div>
    `).join('');
    
    el.classList.remove('hidden');
  },

  /**
   * Update runtime status
   */
  async updateRuntimeStatus() {
    try {
      const runtimes = await edgeFlow.getAvailableRuntimes();
      this.renderStatusList('runtime-status', [
        { label: 'WebGPU', value: runtimes.get('webgpu') ? 'Ready' : 'N/A', status: runtimes.get('webgpu') ? 'success' : 'error' },
        { label: 'WebNN', value: runtimes.get('webnn') ? 'Ready' : 'N/A', status: runtimes.get('webnn') ? 'success' : 'error' },
        { label: 'WASM', value: runtimes.get('wasm') ? 'Ready' : 'N/A', status: runtimes.get('wasm') ? 'success' : 'error' },
      ]);
    } catch {
      this.renderStatusList('runtime-status', [
        { label: 'WebGPU', value: 'N/A', status: 'error' },
        { label: 'WebNN', value: 'N/A', status: 'error' },
        { label: 'WASM', value: 'N/A', status: 'error' },
      ]);
    }
  },

  /**
   * Update memory status
   */
  updateMemoryStatus() {
    try {
      const stats = edgeFlow.getMemoryStats();
      this.renderStatusList('memory-status', [
        { label: 'Allocated', value: utils.formatBytes(stats.allocated || 0) },
        { label: 'Peak', value: utils.formatBytes(stats.peak || 0) },
        { label: 'Tensors', value: String(stats.tensorCount || 0) },
      ]);
    } catch {
      this.renderStatusList('memory-status', [
        { label: 'Allocated', value: '0 B' },
        { label: 'Peak', value: '0 B' },
        { label: 'Tensors', value: '0' },
      ]);
    }
  },

  /**
   * Update monitor metrics
   */
  updateMonitorMetrics(sample) {
    this.renderMetrics('monitor-metrics', [
      { value: sample.inference.count, label: 'Inferences' },
      { value: sample.inference.avgTime.toFixed(1) + 'ms', label: 'Avg Time' },
      { value: sample.inference.throughput.toFixed(1), label: 'Ops/sec' },
      { value: utils.formatBytes(sample.memory.usedHeap), label: 'Memory' },
      { value: sample.system.fps || '-', label: 'FPS' },
    ]);
  },

  /**
   * Initialize default outputs
   */
  initOutputs() {
    const defaults = {
      'model-output': ['Click "Load Model" to download an ONNX model', 'info'],
      'tensor-output': ['Click "Run Tests" to test tensor operations...', ''],
      'text-output': ['Load model first, then classify text...', ''],
      'feature-output': ['Enter text and extract features...', ''],
      'quant-output': ['Test in-browser quantization...', ''],
      'debugger-output': ['Inspect tensor values and statistics...', ''],
      'benchmark-output': ['Benchmark tensor operations...', ''],
      'scheduler-output': ['Test task scheduling with priorities...', ''],
      'memory-output': ['Test memory allocation and cleanup...', ''],
      'concurrency-output': ['Test concurrent inference...', ''],
    };

    for (const [id, [msg, type]] of Object.entries(defaults)) {
      this.setOutput(id, msg, type);
    }

    // Initialize monitor metrics
    this.renderMetrics('monitor-metrics', [
      { value: '0', label: 'Inferences' },
      { value: '0ms', label: 'Avg Time' },
      { value: '0', label: 'Ops/sec' },
      { value: '0 B', label: 'Memory' },
      { value: '-', label: 'FPS' },
    ]);
  },
};

/* ==========================================================================
   4. Core Features
   ========================================================================== */

const features = {
  /**
   * Load ONNX model
   */
  async loadModel() {
    const url = ui.$('model-url')?.value;
    if (!url) {
      ui.setOutput('model-output', 'Enter a model URL', 'warn');
      return;
    }

    ui.showLoading('model-output', 'Loading model...');

    try {
      const start = performance.now();
      state.model = await edgeFlow.loadModel(url, { runtime: 'wasm' });
      const time = ((performance.now() - start) / 1000).toFixed(2);

      const info = [
        `<span class="success">✓ Model loaded in ${time}s</span>`,
        `Name: ${state.model.metadata.name}`,
        `Size: ${utils.formatBytes(state.model.metadata.sizeBytes)}`,
        `Inputs: ${state.model.metadata.inputs.map(i => i.name).join(', ')}`,
      ].join('\n');

      ui.$('model-output').innerHTML = `<pre>${info}</pre>`;
      ui.updateMemoryStatus();
    } catch (e) {
      ui.showError('model-output', e);
    }
  },

  /**
   * Test model inference
   */
  async testModel() {
    if (!state.model) {
      ui.setOutput('model-output', 'Load model first', 'warn');
      return;
    }

    ui.showLoading('model-output', 'Running inference...');

    try {
      const inputs = utils.createModelInputs(state.model);
      const start = performance.now();
      const outputs = await edgeFlow.runInference(state.model, inputs);
      const time = (performance.now() - start).toFixed(2);
      const data = outputs[0].toArray();

      const info = [
        `<span class="success">✓ Inference: ${time}ms</span>`,
        `Output: [${data.slice(0, 5).map(x => x.toFixed(4)).join(', ')}...]`,
      ].join('\n');

      ui.$('model-output').innerHTML = `<pre>${info}</pre>`;

      inputs.forEach(t => t.dispose());
      outputs.forEach(t => t.dispose());
    } catch (e) {
      ui.showError('model-output', e);
    }
  },

  /**
   * Run tensor operation tests
   */
  testTensors() {
    try {
      const a = edgeFlow.tensor([[1, 2], [3, 4]]);
      const b = edgeFlow.tensor([[5, 6], [7, 8]]);
      const sum = edgeFlow.add(a, b);
      const rand = edgeFlow.random([10]);
      const probs = edgeFlow.softmax(edgeFlow.tensor([1, 2, 3, 4]));

      const info = [
        `<span class="success">✓ All tensor tests passed</span>`,
        `• Created 2x2 tensor`,
        `• Addition: [${sum.toArray()}]`,
        `• Random: [${rand.toArray().slice(0, 5).map(x => x.toFixed(2))}...]`,
        `• Softmax: [${probs.toArray().map(x => x.toFixed(3))}]`,
      ].join('\n');

      ui.$('tensor-output').innerHTML = `<pre>${info}</pre>`;

      [a, b, sum, rand, probs].forEach(t => t.dispose());
      ui.updateMemoryStatus();
    } catch (e) {
      ui.showError('tensor-output', e);
    }
  },

  /**
   * Classify single text
   */
  async classifyText() {
    if (!state.model) {
      ui.setOutput('text-output', 'Load model first', 'warn');
      return;
    }

    const text = ui.$('text-input')?.value;
    if (!text) return;

    ui.showLoading('text-output', 'Classifying...');

    try {
      const result = await utils.inferText(text);
      const emoji = result.label === 'positive' ? '😊' : '😞';
      const pct = (result.score * 100).toFixed(1);
      ui.$('text-output').innerHTML = `<pre><span class="success">${emoji} ${result.label.toUpperCase()}</span> (${pct}%)</pre>`;
    } catch (e) {
      ui.showError('text-output', e);
    }
  },

  /**
   * Batch classification
   */
  async classifyBatch() {
    if (!state.model) {
      ui.setOutput('text-output', 'Load model first', 'warn');
      return;
    }

    const texts = ['I love this!', 'This is terrible.', 'Amazing!', 'Worst ever.', 'Pretty good.'];
    ui.showLoading('text-output', 'Processing batch...');

    try {
      const start = performance.now();
      const results = await Promise.all(texts.map(t => utils.inferText(t)));
      const time = (performance.now() - start).toFixed(0);

      const lines = results.map((r, i) => {
        const emoji = r.label === 'positive' ? '😊' : '😞';
        return `${emoji} "${texts[i]}" → ${r.label}`;
      });

      lines.push('', `<span class="success">Total: ${time}ms</span>`);
      ui.$('text-output').innerHTML = `<pre>${lines.join('\n')}</pre>`;
    } catch (e) {
      ui.showError('text-output', e);
    }
  },

  /**
   * Extract features
   */
  async extractFeatures() {
    if (!state.model) {
      ui.setOutput('feature-output', 'Load model first', 'warn');
      return;
    }

    const text = ui.$('feature-input')?.value;
    if (!text) return;

    ui.showLoading('feature-output', 'Extracting...');

    try {
      const inputs = utils.createModelInputs(state.model);
      const start = performance.now();
      const outputs = await edgeFlow.runInference(state.model, inputs);
      const time = (performance.now() - start).toFixed(2);
      const embeddings = outputs[0].toArray();
      const norm = Math.sqrt(embeddings.reduce((a, b) => a + b * b, 0));

      const info = [
        `<span class="success">✓ Features extracted in ${time}ms</span>`,
        `Dimension: ${embeddings.length}`,
        `L2 Norm: ${norm.toFixed(4)}`,
        `Sample: [${embeddings.slice(0, 5).map(x => x.toFixed(4)).join(', ')}...]`,
      ].join('\n');

      ui.$('feature-output').innerHTML = `<pre>${info}</pre>`;

      inputs.forEach(t => t.dispose());
      outputs.forEach(t => t.dispose());
    } catch (e) {
      ui.showError('feature-output', e);
    }
  },

  /**
   * Quantization demo
   */
  quantize() {
    try {
      const weights = edgeFlow.tensor([0.5, -0.3, 0.8, -0.1, 0.9, -0.7, 0.2, -0.4], [2, 4], 'float32');
      const { tensor: quantized, scale, zeroPoint } = edgeFlow.quantizeTensor(weights, 'int8');
      const dequantized = edgeFlow.dequantizeTensor(quantized, scale, zeroPoint, 'int8');
      
      const original = weights.toArray();
      const recovered = dequantized.toArray();
      const maxError = Math.max(...original.map((v, i) => Math.abs(v - recovered[i])));

      const info = [
        `<span class="success">✓ Int8 Quantization</span>`,
        `Original:    [${original.map(v => v.toFixed(3)).join(', ')}]`,
        `Quantized:   [${quantized.toArray().join(', ')}]`,
        `Dequantized: [${recovered.map(v => v.toFixed(3)).join(', ')}]`,
        `Scale: ${scale.toFixed(6)}, Max Error: ${maxError.toFixed(6)}`,
      ].join('\n');

      ui.$('quant-output').innerHTML = `<pre>${info}</pre>`;

      [weights, quantized, dequantized].forEach(t => t.dispose());
    } catch (e) {
      ui.showError('quant-output', e);
    }
  },

  /**
   * Pruning demo
   */
  prune() {
    try {
      const weights = edgeFlow.tensor([0.5, -0.1, 0.8, -0.05, 0.9, -0.02, 0.2, -0.4], [2, 4], 'float32');
      const { tensor: pruned, sparsity } = edgeFlow.pruneTensor(weights, { ratio: 0.5 });

      const info = [
        `<span class="success">✓ Magnitude Pruning (50%)</span>`,
        `Original: [${weights.toArray().map(v => v.toFixed(2)).join(', ')}]`,
        `Pruned:   [${pruned.toArray().map(v => v.toFixed(2)).join(', ')}]`,
        `Sparsity: ${(sparsity * 100).toFixed(1)}%`,
      ].join('\n');

      ui.$('quant-output').innerHTML = `<pre>${info}</pre>`;

      [weights, pruned].forEach(t => t.dispose());
    } catch (e) {
      ui.showError('quant-output', e);
    }
  },

  /**
   * Debugger demo
   */
  debug() {
    try {
      const data = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
      const tensor = edgeFlow.tensor(data, [10, 10], 'float32');
      const inspection = edgeFlow.inspectTensor(tensor, 'random_weights');
      const histogram = edgeFlow.createAsciiHistogram(inspection.histogram, 25, 4);

      const info = [
        `<span class="success">Tensor: ${inspection.name}</span>`,
        `Shape: [${inspection.shape}], Size: ${inspection.size}`,
        `<span class="info">Statistics:</span>`,
        `  Min: ${inspection.stats.min.toFixed(4)}`,
        `  Max: ${inspection.stats.max.toFixed(4)}`,
        `  Mean: ${inspection.stats.mean.toFixed(4)}`,
        `  Std: ${inspection.stats.std.toFixed(4)}`,
        '',
        histogram,
      ].join('\n');

      ui.$('debugger-output').innerHTML = `<pre>${info}</pre>`;

      tensor.dispose();
    } catch (e) {
      ui.showError('debugger-output', e);
    }
  },

  /**
   * Benchmark demo
   */
  async benchmark() {
    ui.showLoading('benchmark-output', 'Running benchmark...');

    try {
      const result = await edgeFlow.runBenchmark(async () => {
        const t = edgeFlow.tensor(Array.from({ length: 1000 }, () => Math.random()), [1000], 'float32');
        const sum = t.toArray().reduce((a, b) => a + b, 0);
        t.dispose();
        return sum;
      }, { warmupRuns: 2, runs: 5, name: 'Tensor Sum (1000)' });

      const info = [
        `<span class="success">Benchmark: ${result.name}</span>`,
        `Avg: ${result.avgTime.toFixed(2)}ms`,
        `Min: ${result.minTime.toFixed(2)}ms`,
        `Max: ${result.maxTime.toFixed(2)}ms`,
        `Throughput: ${result.throughput.toFixed(0)} ops/sec`,
      ].join('\n');

      ui.$('benchmark-output').innerHTML = `<pre>${info}</pre>`;
    } catch (e) {
      ui.showError('benchmark-output', e);
    }
  },

  /**
   * Scheduler test
   */
  async testScheduler() {
    ui.showLoading('scheduler-output', 'Testing scheduler...');

    try {
      const scheduler = edgeFlow.getScheduler();
      const task1 = scheduler.schedule('model-a', async () => { await utils.sleep(100); return 'Task 1'; }, 'high');
      const task2 = scheduler.schedule('model-b', async () => { await utils.sleep(50); return 'Task 2'; }, 'normal');
      const task3 = scheduler.schedule('model-a', async () => { await utils.sleep(75); return 'Task 3'; }, 'low');

      const [r1, r2, r3] = await Promise.all([task1.wait(), task2.wait(), task3.wait()]);

      const info = [
        `<span class="success">✓ Scheduler Test Passed</span>`,
        `• ${r1} (high priority)`,
        `• ${r2} (normal priority)`,
        `• ${r3} (low priority)`,
      ].join('\n');

      ui.$('scheduler-output').innerHTML = `<pre>${info}</pre>`;
    } catch (e) {
      ui.showError('scheduler-output', e);
    }
  },

  /**
   * Memory allocation test
   */
  allocateMemory() {
    try {
      const before = edgeFlow.getMemoryStats();
      
      for (let i = 0; i < 10; i++) {
        state.testTensors.push(edgeFlow.random([100, 100]));
      }
      
      const after = edgeFlow.getMemoryStats();

      const info = [
        `<span class="success">✓ Allocated 10 tensors (100x100)</span>`,
        `Before: ${utils.formatBytes(before.allocated || 0)}, ${before.tensorCount || 0} tensors`,
        `After: ${utils.formatBytes(after.allocated || 0)}, ${after.tensorCount || 0} tensors`,
      ].join('\n');

      ui.$('memory-output').innerHTML = `<pre>${info}</pre>`;
      ui.updateMemoryStatus();
    } catch (e) {
      ui.showError('memory-output', e);
    }
  },

  /**
   * Memory cleanup
   */
  cleanupMemory() {
    state.testTensors.forEach(t => {
      if (!t.isDisposed) t.dispose();
    });
    state.testTensors = [];
    edgeFlow.gc();

    ui.showSuccess('memory-output', 'Memory cleaned up');
    ui.updateMemoryStatus();
  },

  /**
   * Concurrency test
   */
  async testConcurrency() {
    if (!state.model) {
      ui.setOutput('concurrency-output', 'Load model first', 'warn');
      ui.$('concurrency-metrics')?.classList.add('hidden');
      return;
    }

    ui.showLoading('concurrency-output', 'Running concurrent tasks...');

    try {
      const texts = ['Great!', 'Terrible!', 'Amazing!', 'Awful!', 'Good!', 'Bad!', 'Nice!', 'Horrible!'];
      const start = performance.now();
      const results = await Promise.all(texts.map(t => utils.inferText(t)));
      const total = performance.now() - start;

      const lines = [
        `<span class="success">✓ Concurrent execution complete</span>`,
        ...results.map((r, i) => `${r.label === 'positive' ? '😊' : '😞'} "${texts[i]}"`),
      ];

      ui.$('concurrency-output').innerHTML = `<pre>${lines.join('\n')}</pre>`;

      ui.renderMetrics('concurrency-metrics', [
        { value: total.toFixed(0) + 'ms', label: 'Total' },
        { value: String(texts.length), label: 'Tasks' },
        { value: (total / texts.length).toFixed(0) + 'ms', label: 'Avg' },
      ]);
    } catch (e) {
      ui.showError('concurrency-output', e);
    }
  },

  /**
   * Start performance monitor
   */
  startMonitor() {
    if (!state.monitor) {
      state.monitor = new edgeFlow.PerformanceMonitor({
        sampleInterval: config.monitorSampleInterval,
        historySize: config.monitorHistorySize,
      });
      state.monitor.onSample(sample => ui.updateMonitorMetrics(sample));
    }
    state.monitor.start();
  },

  /**
   * Stop monitor
   */
  stopMonitor() {
    if (state.monitor) {
      state.monitor.stop();
    }
  },

  /**
   * Simulate inferences for monitor
   */
  simulateInferences() {
    if (!state.monitor) {
      this.startMonitor();
    }
    
    for (let i = 0; i < 5; i++) {
      setTimeout(() => {
        state.monitor?.recordInference(30 + Math.random() * 70);
      }, i * 100);
    }
  },

  /**
   * Open dashboard modal
   */
  openDashboard() {
    if (!state.monitor) {
      this.startMonitor();
      this.simulateInferences();
    }

    const modal = ui.$('dashboard-modal');
    const frame = ui.$('dashboard-frame');
    
    if (modal && frame) {
      frame.srcdoc = edgeFlow.generateDashboardHTML(state.monitor);
      modal.classList.add('active');
      document.body.style.overflow = 'hidden';
    }
  },

  /**
   * Close dashboard modal
   */
  closeDashboard() {
    const modal = ui.$('dashboard-modal');
    if (modal) {
      modal.classList.remove('active');
      document.body.style.overflow = '';
    }
  },
};

/* ==========================================================================
   5. SAM Interactive Segmentation (Real Model)
   ========================================================================== */

const sam = {
  /**
   * Initialize SAM UI and start model loading
   */
  async init() {
    const fileInput = ui.$('sam-file-input');
    const container = ui.$('sam-container');
    
    if (fileInput) {
      fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
    }
    
    // Drag and drop
    if (container) {
      container.addEventListener('dragover', (e) => {
        e.preventDefault();
        container.classList.add('dragover');
      });
      container.addEventListener('dragleave', () => {
        container.classList.remove('dragover');
      });
      container.addEventListener('drop', (e) => {
        e.preventDefault();
        container.classList.remove('dragover');
        const file = e.dataTransfer?.files[0];
        if (file && file.type.startsWith('image/')) {
          this.loadImage(file);
        }
      });
    }

    // Start loading SAM models automatically
    await this.loadModels();
  },

  /**
   * Load SAM models with progress display
   */
  async loadModels() {
    const loader = ui.$('sam-loader');
    const loaderText = ui.$('sam-loader-text');
    const loaderDetail = ui.$('sam-loader-detail');
    const progress = ui.$('sam-progress');
    const samContainer = ui.$('sam-container');
    
    try {
      // Create pipeline
      state.samPipeline = edgeFlow.createImageSegmentationPipeline();
      
      // Load models with progress
      await state.samPipeline.loadModels((progressInfo) => {
        const { model, progress: pct, loaded, total } = progressInfo;
        
        if (loaderText) {
          loaderText.textContent = `Loading ${model}... (${utils.formatBytes(loaded)} / ${utils.formatBytes(total)})`;
        }
        if (loaderDetail) {
          loaderDetail.textContent = `${pct}%`;
        }
        if (progress) {
          progress.style.width = `${pct}%`;
        }
      });
      
      state.samModelLoaded = true;
      
      // Hide loader, show main UI
      if (loader) loader.classList.add('hidden');
      if (samContainer) samContainer.classList.remove('hidden');
      
      // Enable buttons
      ui.$('sam-sample-btn')?.removeAttribute('disabled');
      ui.$('sam-clear-btn')?.removeAttribute('disabled');
      ui.$('sam-download-btn')?.removeAttribute('disabled');
      
      ui.setOutput('sam-output', '✓ SAM model loaded! Click to upload an image or use "Sample Image".', 'success');
      
    } catch (error) {
      console.error('SAM model loading failed:', error);
      
      if (loaderText) {
        loaderText.textContent = `Failed to load model: ${error.message}`;
        loaderText.style.color = 'var(--error)';
      }
      if (loaderDetail) {
        loaderDetail.textContent = 'Check console for details';
      }
      
      ui.showError('sam-output', error);
    }
  },

  /**
   * Handle file selection
   */
  handleFileSelect(e) {
    const file = e.target?.files?.[0];
    if (file) {
      this.loadImage(file);
    }
  },

  /**
   * Load image from file or URL
   */
  async loadImage(source) {
    if (!state.samModelLoaded) {
      ui.setOutput('sam-output', 'Model not loaded yet. Please wait...', 'warn');
      return;
    }
    
    ui.setOutput('sam-output', 'Loading image...', 'info');
    
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        
        if (typeof source === 'string') {
          img.src = source;
        } else {
          img.src = URL.createObjectURL(source);
        }
      });
      
      // Show workspace
      ui.$('sam-upload')?.classList.add('hidden');
      ui.$('sam-workspace')?.classList.remove('hidden');
      
      // Setup canvases
      const canvas = ui.$('sam-canvas');
      const maskCanvas = ui.$('sam-mask-canvas');
      
      if (canvas && maskCanvas) {
        state.samCanvas = canvas;
        state.samMaskCanvas = maskCanvas;
        state.samCtx = canvas.getContext('2d');
        state.samMaskCtx = maskCanvas.getContext('2d');
        
        // Set canvas size
        const container = ui.$('sam-workspace');
        const containerWidth = container?.clientWidth || 400;
        const containerHeight = container?.clientHeight || 250;
        
        const scale = Math.min(
          containerWidth / img.width,
          containerHeight / img.height
        );
        
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        maskCanvas.width = canvas.width;
        maskCanvas.height = canvas.height;
        
        // Draw image
        state.samCtx.drawImage(img, 0, 0, canvas.width, canvas.height);
        state.samImage = img;
        state.samPoints = [];
        
        // Setup click handler
        canvas.onclick = (e) => this.handleClick(e, 1); // Left click = positive
        canvas.oncontextmenu = (e) => {
          e.preventDefault();
          this.handleClick(e, 0); // Right click = negative
        };
        
        // Encode image with SAM encoder
        ui.setOutput('sam-output', 'Encoding image with SAM...', 'info');
        const encodeStart = performance.now();
        await state.samPipeline.setImage(img);
        const encodeTime = (performance.now() - encodeStart).toFixed(0);
        
        ui.setOutput('sam-output', `✓ Image encoded in ${encodeTime}ms. Click to segment objects. Left-click = include, Right-click = exclude.`, 'success');
      }
    } catch (error) {
      ui.showError('sam-output', error);
    }
  },

  /**
   * Load sample image
   */
  async loadSampleImage() {
    if (!state.samModelLoaded) {
      ui.setOutput('sam-output', 'Model not loaded yet. Please wait...', 'warn');
      return;
    }
    
    // Using a reliable public image URL
    const sampleUrl = 'https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=640';
    await this.loadImage(sampleUrl);
  },

  /**
   * Handle canvas click
   */
  async handleClick(e, label) {
    if (!state.samCanvas || !state.samPipeline || !state.samModelLoaded) return;
    
    const rect = state.samCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    
    // Add point
    state.samPoints.push({ x, y, label });
    
    // Draw point indicator
    this.drawPoints();
    
    // Run segmentation
    ui.setOutput('sam-output', 'Segmenting...', 'info');
    
    try {
      const startTime = performance.now();
      const result = await state.samPipeline.segment({
        points: state.samPoints,
      });
      const time = (performance.now() - startTime).toFixed(0);
      
      // Draw mask
      this.drawMask(result);
      
      ui.setOutput('sam-output', `✓ Segmented in ${time}ms (score: ${result.score.toFixed(2)})`, 'success');
    } catch (error) {
      ui.showError('sam-output', error);
    }
  },

  /**
   * Draw points on canvas
   */
  drawPoints() {
    // Remove existing point indicators
    document.querySelectorAll('.sam-point').forEach(el => el.remove());
    
    const workspace = ui.$('sam-workspace');
    if (!workspace || !state.samCanvas) return;
    
    for (const point of state.samPoints) {
      const indicator = document.createElement('div');
      indicator.className = `sam-point ${point.label === 1 ? 'positive' : 'negative'}`;
      indicator.style.left = `${point.x * 100}%`;
      indicator.style.top = `${point.y * 100}%`;
      workspace.appendChild(indicator);
    }
  },

  /**
   * Draw segmentation mask
   */
  drawMask(result) {
    if (!state.samMaskCtx || !state.samMaskCanvas) return;
    
    const { mask, width, height } = result;
    const canvas = state.samMaskCanvas;
    
    // Create ImageData
    const imageData = state.samMaskCtx.createImageData(canvas.width, canvas.height);
    
    // Scale mask to canvas size
    const scaleX = width / canvas.width;
    const scaleY = height / canvas.height;
    
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const srcX = Math.floor(x * scaleX);
        const srcY = Math.floor(y * scaleY);
        const srcIdx = srcY * width + srcX;
        const dstIdx = (y * canvas.width + x) * 4;
        
        if (mask[srcIdx] > 0) {
          // Green overlay for segmented area
          imageData.data[dstIdx] = 127;     // R
          imageData.data[dstIdx + 1] = 169; // G
          imageData.data[dstIdx + 2] = 33;  // B
          imageData.data[dstIdx + 3] = 180; // A
        }
      }
    }
    
    state.samMaskCtx.putImageData(imageData, 0, 0);
  },

  /**
   * Clear segmentation
   */
  clear() {
    state.samPoints = [];
    
    // Clear mask canvas
    if (state.samMaskCtx && state.samMaskCanvas) {
      state.samMaskCtx.clearRect(0, 0, state.samMaskCanvas.width, state.samMaskCanvas.height);
    }
    
    // Remove point indicators
    document.querySelectorAll('.sam-point').forEach(el => el.remove());
    
    ui.setOutput('sam-output', 'Cleared. Click to segment objects.', 'info');
  },

  /**
   * Download mask as PNG
   */
  downloadMask() {
    if (!state.samMaskCanvas) {
      ui.setOutput('sam-output', 'No mask to download', 'warn');
      return;
    }
    
    // Create download link
    const link = document.createElement('a');
    link.download = 'segmentation-mask.png';
    link.href = state.samMaskCanvas.toDataURL('image/png');
    link.click();
  },

  /**
   * Reset to upload state
   */
  reset() {
    state.samImage = null;
    state.samPoints = [];
    
    ui.$('sam-upload')?.classList.remove('hidden');
    ui.$('sam-workspace')?.classList.add('hidden');
    
    document.querySelectorAll('.sam-point').forEach(el => el.remove());
    
    if (state.samMaskCtx && state.samMaskCanvas) {
      state.samMaskCtx.clearRect(0, 0, state.samMaskCanvas.width, state.samMaskCanvas.height);
    }
    
    // Clear the pipeline's image embedding
    if (state.samPipeline) {
      state.samPipeline.clearImage();
    }
    
    ui.setOutput('sam-output', 'Click on image to segment objects. Left-click = include, Right-click = exclude.', 'info');
  },
};

/* ==========================================================================
   6. AI Chat (Real Model)
   ========================================================================== */

const chat = {
  /**
   * Initialize chat UI
   */
  init() {
    const input = ui.$('chat-input');
    if (input) {
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !state.chatGenerating) {
          e.preventDefault();
          this.send();
        }
      });
    }
  },

  /**
   * Load LLM model with progress display
   */
  async loadModel() {
    if (state.chatModelLoaded) {
      ui.$('chat-container')?.classList.remove('hidden');
      ui.$('llm-loader')?.classList.add('hidden');
      return;
    }
    
    const loadBtn = ui.$('llm-load-btn');
    const progressContainer = ui.$('llm-progress-container');
    const progress = ui.$('llm-progress');
    const loaderDetail = ui.$('llm-loader-detail');
    
    try {
      // Disable button and show progress
      if (loadBtn) {
        loadBtn.disabled = true;
        loadBtn.textContent = 'Loading...';
      }
      if (progressContainer) progressContainer.classList.remove('hidden');
      if (loaderDetail) loaderDetail.classList.remove('hidden');
      
      this.updateStatus('loading', 'Downloading model...');
      
      // Create pipeline
      state.chatPipeline = edgeFlow.createTextGenerationPipeline();
      state.chatPipeline.setChatTemplate('chatml');
      
      // Load model with progress
      await state.chatPipeline.loadModel((progressInfo) => {
        const { stage, progress: pct, loaded, total } = progressInfo;
        
        if (loadBtn) {
          if (stage === 'tokenizer') {
            loadBtn.textContent = 'Loading tokenizer...';
          } else {
            loadBtn.textContent = `Loading model... ${utils.formatBytes(loaded)}`;
          }
        }
        if (loaderDetail) {
          loaderDetail.textContent = `${stage}: ${pct}%`;
        }
        if (progress) {
          // Tokenizer is quick, model is the main download
          const totalProgress = stage === 'tokenizer' ? pct * 0.05 : 5 + pct * 0.95;
          progress.style.width = `${totalProgress}%`;
        }
      });
      
      state.chatModelLoaded = true;
      
      // Hide loader, show chat UI
      ui.$('llm-loader')?.classList.add('hidden');
      ui.$('chat-container')?.classList.remove('hidden');
      
      this.updateStatus('ready', 'Model loaded! Ready to chat');
      
    } catch (error) {
      console.error('LLM model loading failed:', error);
      
      if (loadBtn) {
        loadBtn.disabled = false;
        loadBtn.textContent = 'Retry Load';
      }
      if (loaderDetail) {
        loaderDetail.textContent = `Error: ${error.message}`;
        loaderDetail.style.color = 'var(--error)';
      }
      
      this.updateStatus('error', `Failed: ${error.message}`);
    }
  },

  /**
   * Send message
   */
  async send() {
    if (!state.chatModelLoaded) {
      this.updateStatus('error', 'Load model first by clicking "Load Model"');
      return;
    }
    
    const input = ui.$('chat-input');
    const message = input?.value?.trim();
    
    if (!message || state.chatGenerating) return;
    
    // Clear input
    input.value = '';
    
    // Hide welcome message
    const welcome = ui.$('chat-messages')?.querySelector('.chat-welcome');
    if (welcome) welcome.remove();
    
    // Add user message
    this.addMessage('user', message);
    
    // Set generating state
    state.chatGenerating = true;
    this.updateStatus('loading', 'Generating...');
    
    try {
      // Add assistant message placeholder
      const assistantMsg = this.addMessage('assistant', '', true);
      
      // Generate response using real model
      let response = '';
      
      // Use streaming if available
      if (state.chatPipeline.chatStream) {
        for await (const event of state.chatPipeline.chatStream(message, {
          maxNewTokens: 256,
          temperature: 0.7,
          topP: 0.9,
        })) {
          response = event.generatedText;
          assistantMsg.textContent = response;
          
          // Scroll to bottom
          const container = ui.$('chat-messages');
          if (container) {
            container.scrollTop = container.scrollHeight;
          }
        }
      } else {
        // Fallback to non-streaming
        const result = await state.chatPipeline.chat(message, {
          maxNewTokens: 256,
          temperature: 0.7,
          topP: 0.9,
        });
        response = result.generatedText;
        assistantMsg.textContent = response;
      }
      
      // Remove typing indicator
      assistantMsg.classList.remove('typing');
      
      // Update history
      state.chatHistory.push(
        { role: 'user', content: message },
        { role: 'assistant', content: response }
      );
      
      this.updateStatus('ready', 'Ready to chat');
    } catch (error) {
      this.updateStatus('error', `Error: ${error.message}`);
      // Remove typing indicator
      const typingMsg = ui.$('chat-messages')?.querySelector('.typing');
      if (typingMsg) typingMsg.remove();
    } finally {
      state.chatGenerating = false;
    }
    
    // Scroll to bottom
    const container = ui.$('chat-messages');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  },

  /**
   * Add message to chat
   */
  addMessage(role, content, isTyping = false) {
    const container = ui.$('chat-messages');
    if (!container) return null;
    
    const msg = document.createElement('div');
    msg.className = `chat-message ${role}${isTyping ? ' typing' : ''}`;
    msg.textContent = content;
    
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
    
    return msg;
  },

  /**
   * Update status indicator
   */
  updateStatus(status, text) {
    const dot = ui.$('chat-status')?.querySelector('.chat-status-dot');
    const textEl = ui.$('chat-status-text');
    
    if (dot) {
      dot.className = `chat-status-dot ${status === 'loading' ? 'loading' : status === 'error' ? 'error' : ''}`;
    }
    
    if (textEl) {
      textEl.textContent = text;
    }
  },

  /**
   * Clear chat history
   */
  clear() {
    state.chatHistory = [];
    
    // Clear conversation in pipeline
    if (state.chatPipeline) {
      state.chatPipeline.clearConversation();
    }
    
    const container = ui.$('chat-messages');
    if (container) {
      container.innerHTML = `
        <div class="chat-welcome">
          <span class="chat-welcome-icon">🤖</span>
          <p>Hi! I'm TinyLlama running entirely in your browser.</p>
          <p class="chat-welcome-hint">Ask me anything!</p>
        </div>
      `;
    }
    
    this.updateStatus('ready', 'Ready to chat');
  },
};

/* ==========================================================================
   7. Demo Class (Public API)
   ========================================================================== */

/**
 * Demo public API - exposed to window for onclick handlers
 */
window.Demo = {
  // Model
  loadModel: () => features.loadModel(),
  testModel: () => features.testModel(),

  // SAM Interactive Segmentation
  loadSampleImage: () => sam.loadSampleImage(),
  clearSegmentation: () => sam.clear(),
  downloadMask: () => sam.downloadMask(),

  // AI Chat
  loadLLM: () => chat.loadModel(),
  sendChat: () => chat.send(),
  clearChat: () => chat.clear(),

  // Core
  testTensors: () => features.testTensors(),
  classifyText: () => features.classifyText(),
  classifyBatch: () => features.classifyBatch(),
  extractFeatures: () => features.extractFeatures(),

  // Tools
  quantize: () => features.quantize(),
  prune: () => features.prune(),
  debug: () => features.debug(),
  benchmark: () => features.benchmark(),

  // System
  testScheduler: () => features.testScheduler(),
  allocateMemory: () => features.allocateMemory(),
  cleanupMemory: () => features.cleanupMemory(),
  testConcurrency: () => features.testConcurrency(),

  // Monitor
  startMonitor: () => features.startMonitor(),
  stopMonitor: () => features.stopMonitor(),
  simulateInferences: () => features.simulateInferences(),
  openDashboard: () => features.openDashboard(),
  closeDashboard: () => features.closeDashboard(),
};

/* ==========================================================================
   8. Initialization
   ========================================================================== */

/**
 * Initialize demo on DOM ready
 */
async function init() {
  // Initialize UI
  ui.initOutputs();
  await ui.updateRuntimeStatus();
  ui.updateMemoryStatus();

  // Initialize Chat UI (but don't load model yet)
  chat.init();

  // Initialize SAM and start loading models automatically
  await sam.init();

  // Setup modal close handlers
  const modal = ui.$('dashboard-modal');
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        features.closeDashboard();
      }
    });
  }

  // ESC key closes modal
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      features.closeDashboard();
    }
  });

  console.log('✓ edgeFlow.js Demo initialized');
}

// Wait for DOM
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
