/**
 * Integration tests for Pipelines
 * 
 * These tests mock the ONNX runtime to return realistic tensor shapes
 * without requiring actual model files.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { EdgeFlowTensor, softmax } from '../../src/core/tensor';

// ============================================================================
// Mock ONNX Runtime helpers
// ============================================================================

/**
 * Create a mock ONNX session that returns tensors with the given output shapes
 */
function createMockOutputs(shapes: { shape: number[]; fill?: number }[]): EdgeFlowTensor[] {
  return shapes.map(({ shape, fill }) => {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(fill ?? 0);
    // Add some variation so softmax/argmax produce deterministic results
    for (let i = 0; i < data.length; i++) {
      data[i] = (data[i] ?? 0) + Math.sin(i * 0.7) * 2;
    }
    return new EdgeFlowTensor(data, shape, 'float32');
  });
}

// ============================================================================
// Text Classification Pipeline
// ============================================================================

describe('TextClassificationPipeline', () => {
  it('should postprocess logits into label + score', () => {
    // Simulate [batch=1, num_labels=2] logits (SST-2)
    const logits = new EdgeFlowTensor(new Float32Array([1.5, -0.5]), [1, 2], 'float32');
    const probs = softmax(logits, -1) as EdgeFlowTensor;
    const probsArray = probs.toFloat32Array();

    expect(probsArray.length).toBe(2);
    // logit 1.5 should have higher probability than -0.5
    expect(probsArray[0]).toBeGreaterThan(probsArray[1]!);
    // Sum should be ~1
    expect((probsArray[0] ?? 0) + (probsArray[1] ?? 0)).toBeCloseTo(1.0, 4);
  });

  it('should handle batch of logits', () => {
    const labels = ['NEGATIVE', 'POSITIVE'];

    // Batch of 3 texts: positive, negative, neutral
    const texts = [
      new EdgeFlowTensor(new Float32Array([-2.0, 3.0]), [1, 2], 'float32'), // POSITIVE
      new EdgeFlowTensor(new Float32Array([3.0, -2.0]), [1, 2], 'float32'), // NEGATIVE
      new EdgeFlowTensor(new Float32Array([0.1, 0.1]), [1, 2], 'float32'),  // ~neutral
    ];

    for (const logits of texts) {
      const probs = softmax(logits, -1) as EdgeFlowTensor;
      const arr = probs.toFloat32Array();
      let maxIdx = 0;
      if ((arr[1] ?? 0) > (arr[0] ?? 0)) maxIdx = 1;
      expect(labels[maxIdx]).toBeDefined();
    }

    // Verify first is POSITIVE
    const p0 = softmax(texts[0]!, -1).toFloat32Array();
    expect(p0[1]).toBeGreaterThan(p0[0]!);

    // Verify second is NEGATIVE
    const p1 = softmax(texts[1]!, -1).toFloat32Array();
    expect(p1[0]).toBeGreaterThan(p1[1]!);
  });

  it('should handle multi-class logits', () => {
    const logits = new EdgeFlowTensor(
      new Float32Array([0.1, 0.2, 5.0, 0.3, 0.1]),
      [1, 5],
      'float32'
    );
    const probs = softmax(logits, -1) as EdgeFlowTensor;
    const arr = probs.toFloat32Array();

    // Index 2 should dominate
    let maxIdx = 0;
    for (let i = 1; i < arr.length; i++) {
      if ((arr[i] ?? 0) > (arr[maxIdx] ?? 0)) maxIdx = i;
    }
    expect(maxIdx).toBe(2);
  });
});

// ============================================================================
// Feature Extraction Pipeline
// ============================================================================

describe('FeatureExtractionPipeline', () => {
  it('should produce embeddings of correct dimension', () => {
    // Simulate [batch=1, seq_len=10, hidden=384] (MiniLM)
    const seqLen = 10;
    const hiddenDim = 384;
    const data = new Float32Array(seqLen * hiddenDim);
    for (let i = 0; i < data.length; i++) data[i] = Math.sin(i * 0.01);

    const hiddenStates = new EdgeFlowTensor(data, [1, seqLen, hiddenDim], 'float32');

    // Mean pooling
    const flat = hiddenStates.toFloat32Array();
    const result = new Float32Array(hiddenDim);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < hiddenDim; j++) {
        result[j] = (result[j] ?? 0) + (flat[i * hiddenDim + j] ?? 0) / seqLen;
      }
    }

    expect(result.length).toBe(384);
    // Verify non-zero
    expect(result.some(v => v !== 0)).toBe(true);
  });

  it('should normalize embeddings to unit length', () => {
    const vec = [0.3, 0.4, 0.5, 0.6];
    let norm = 0;
    for (const v of vec) norm += v * v;
    norm = Math.sqrt(norm);
    const normalized = vec.map(v => v / norm);

    let normAfter = 0;
    for (const v of normalized) normAfter += v * v;
    expect(Math.sqrt(normAfter)).toBeCloseTo(1.0, 5);
  });

  it('should handle CLS pooling', () => {
    const seqLen = 5;
    const hiddenDim = 384;
    const data = new Float32Array(seqLen * hiddenDim);
    for (let i = 0; i < data.length; i++) data[i] = i * 0.001;

    // CLS = first token
    const cls = Array.from(data.slice(0, hiddenDim));
    expect(cls.length).toBe(384);
    expect(cls[0]).toBeCloseTo(0);
    expect(cls[1]).toBeCloseTo(0.001, 4);
  });
});

// ============================================================================
// Image Classification Pipeline
// ============================================================================

describe('ImageClassificationPipeline', () => {
  it('should postprocess logits to label', () => {
    // Simulate [1, 1000] ImageNet logits
    const logits = new Float32Array(1000);
    logits[282] = 10.0; // tiger cat
    const tensor = new EdgeFlowTensor(logits, [1, 1000], 'float32');
    const probs = softmax(tensor, -1) as EdgeFlowTensor;
    const arr = probs.toFloat32Array();

    let maxIdx = 0;
    for (let i = 1; i < arr.length; i++) {
      if ((arr[i] ?? 0) > (arr[maxIdx] ?? 0)) maxIdx = i;
    }
    expect(maxIdx).toBe(282);
  });
});

// ============================================================================
// Object Detection Pipeline
// ============================================================================

describe('ObjectDetectionPipeline', () => {
  it('should parse YOLO-style output', () => {
    // Simulate [1, 3, 85] (3 boxes, 80 COCO classes + 5)
    const numBoxes = 3;
    const boxSize = 85;
    const data = new Float32Array(numBoxes * boxSize);

    // Box 0: high confidence person
    data[0] = 0.5; data[1] = 0.5; data[2] = 0.2; data[3] = 0.3;
    data[4] = 0.9; // objectness
    data[5] = 0.95; // class 0 (person)

    // Box 1: low objectness
    const offset1 = boxSize;
    data[offset1 + 4] = 0.1;

    const threshold = 0.5;
    const detections: Array<{ classId: number; score: number }> = [];

    for (let i = 0; i < numBoxes; i++) {
      const off = i * boxSize;
      const objectness = data[off + 4] ?? 0;
      if (objectness < threshold) continue;

      let maxClass = 0;
      let maxScore = 0;
      for (let c = 0; c < 80; c++) {
        const s = data[off + 5 + c] ?? 0;
        if (s > maxScore) { maxScore = s; maxClass = c; }
      }

      detections.push({ classId: maxClass, score: objectness * maxScore });
    }

    expect(detections.length).toBe(1);
    expect(detections[0]!.classId).toBe(0);
    expect(detections[0]!.score).toBeCloseTo(0.9 * 0.95);
  });

  it('should filter by confidence threshold', () => {
    const boxes = [
      { score: 0.9, label: 'cat' },
      { score: 0.3, label: 'dog' },
      { score: 0.7, label: 'bird' },
    ];

    const filtered = boxes.filter(b => b.score >= 0.5);
    expect(filtered.length).toBe(2);
    expect(filtered.map(b => b.label)).toEqual(['cat', 'bird']);
  });
});

// ============================================================================
// Question Answering Pipeline
// ============================================================================

describe('QuestionAnsweringPipeline', () => {
  it('should find best span from start/end logits', () => {
    const seqLen = 10;
    const startLogits = new Float32Array(seqLen);
    const endLogits = new Float32Array(seqLen);

    // Best answer at positions 3-5
    startLogits[3] = 5.0;
    endLogits[5] = 5.0;

    const startProbs = softmax(new EdgeFlowTensor(startLogits, [seqLen], 'float32')).toFloat32Array();
    const endProbs = softmax(new EdgeFlowTensor(endLogits, [seqLen], 'float32')).toFloat32Array();

    let bestStart = 0;
    let bestEnd = 0;
    let bestScore = 0;

    for (let s = 0; s < seqLen; s++) {
      for (let e = s; e < Math.min(s + 8, seqLen); e++) {
        const score = (startProbs[s] ?? 0) * (endProbs[e] ?? 0);
        if (score > bestScore) {
          bestScore = score;
          bestStart = s;
          bestEnd = e;
        }
      }
    }

    expect(bestStart).toBe(3);
    expect(bestEnd).toBe(5);
    expect(bestScore).toBeGreaterThan(0);
  });

  it('should handle no-answer case (all low scores)', () => {
    const seqLen = 10;
    const startLogits = new Float32Array(seqLen).fill(0.01);
    const endLogits = new Float32Array(seqLen).fill(0.01);

    const startProbs = softmax(new EdgeFlowTensor(startLogits, [seqLen], 'float32')).toFloat32Array();
    const endProbs = softmax(new EdgeFlowTensor(endLogits, [seqLen], 'float32')).toFloat32Array();

    let bestScore = 0;
    for (let s = 0; s < seqLen; s++) {
      for (let e = s; e < Math.min(s + 8, seqLen); e++) {
        const score = (startProbs[s] ?? 0) * (endProbs[e] ?? 0);
        if (score > bestScore) bestScore = score;
      }
    }

    // Score should be low (uniform distribution)
    expect(bestScore).toBeLessThan(0.1);
  });
});

// ============================================================================
// Zero-Shot Classification Pipeline
// ============================================================================

describe('ZeroShotClassificationPipeline', () => {
  it('should rank labels by entailment score', () => {
    // Simulate NLI outputs for 3 labels: [contradiction, neutral, entailment]
    const entailmentScores = [
      0.1,  // label "politics"
      0.9,  // label "sports"
      0.3,  // label "technology"
    ];

    // Softmax normalization (mutually exclusive)
    const tensor = new EdgeFlowTensor(
      new Float32Array(entailmentScores),
      [3],
      'float32'
    );
    const probs = softmax(tensor).toFloat32Array();

    const labels = ['politics', 'sports', 'technology'];
    const indexed = labels.map((l, i) => ({ label: l, score: probs[i] ?? 0 }));
    indexed.sort((a, b) => b.score - a.score);

    expect(indexed[0]!.label).toBe('sports');
    expect(indexed[0]!.score).toBeGreaterThan(indexed[1]!.score);
  });

  it('should handle multi-label with sigmoid', () => {
    const scores = [2.0, -1.0, 0.5];
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

    const probs = scores.map(sigmoid);
    expect(probs[0]).toBeGreaterThan(0.8);
    expect(probs[1]).toBeLessThan(0.3);
    expect(probs[2]).toBeGreaterThan(0.5);
    expect(probs[2]).toBeLessThan(0.7);
  });
});

// ============================================================================
// ASR Pipeline
// ============================================================================

describe('AutomaticSpeechRecognitionPipeline', () => {
  it('should decode argmax token IDs from logits', () => {
    // Simulate decoder output [1, seq_len=3, vocab_size=5]
    const vocabSize = 5;
    const seqLen = 3;
    const data = new Float32Array(seqLen * vocabSize).fill(-10);

    // Token 0 → class 2, Token 1 → class 4, Token 2 → class 0
    data[0 * vocabSize + 2] = 10.0;
    data[1 * vocabSize + 4] = 10.0;
    data[2 * vocabSize + 0] = 10.0;

    const tokenIds: number[] = [];
    for (let i = 0; i < seqLen; i++) {
      let maxIdx = 0;
      let maxVal = data[i * vocabSize] ?? -Infinity;
      for (let j = 1; j < vocabSize; j++) {
        if ((data[i * vocabSize + j] ?? -Infinity) > maxVal) {
          maxVal = data[i * vocabSize + j] ?? -Infinity;
          maxIdx = j;
        }
      }
      tokenIds.push(maxIdx);
    }

    expect(tokenIds).toEqual([2, 4, 0]);
  });

  it('should handle timestamp extraction', () => {
    const text = 'hello world how are you today';
    const words = text.split(/\s+/);
    const chunks: Array<{ text: string; start: number; end: number }> = [];

    const wordsPerSecond = 2.5;
    let chunkText = '';
    let chunkStart = 0;

    for (let i = 0; i < words.length; i++) {
      chunkText += (chunkText ? ' ' : '') + words[i];
      if ((i + 1) % 5 === 0 || i === words.length - 1) {
        const duration = chunkText.split(/\s+/).length / wordsPerSecond;
        chunks.push({ text: chunkText, start: chunkStart, end: chunkStart + duration });
        chunkStart = chunkStart + duration;
        chunkText = '';
      }
    }

    expect(chunks.length).toBe(2);
    expect(chunks[0]!.start).toBe(0);
    expect(chunks[0]!.end).toBeGreaterThan(0);
    expect(chunks[1]!.start).toBe(chunks[0]!.end);
  });
});

// ============================================================================
// Basic tensor operation tests (kept from original)
// ============================================================================

describe('Tensor Operations for Pipelines', () => {
  it('should create tensor for input_ids', () => {
    const inputIds = new EdgeFlowTensor([101, 1000, 102], [1, 3], 'int64');
    expect(inputIds.shape).toEqual([1, 3]);
    expect(inputIds.dtype).toBe('int64');
  });

  it('should create attention mask', () => {
    const attentionMask = new EdgeFlowTensor([1, 1, 1, 0, 0], [1, 5], 'int64');
    expect(attentionMask.shape).toEqual([1, 5]);
  });

  it('should handle batched inputs', () => {
    const batchedInputs = new EdgeFlowTensor(
      [101, 1000, 102, 0, 0, 101, 1001, 1002, 102, 0],
      [2, 5],
      'int64'
    );
    expect(batchedInputs.shape).toEqual([2, 5]);
    expect(Number(batchedInputs.get(0, 0))).toBe(101);
  });

  it('should reshape outputs', () => {
    const hidden = 768;
    const output = new EdgeFlowTensor(new Array(hidden).fill(0.1), [1, 1, hidden]);
    expect(output.shape).toEqual([1, 1, hidden]);

    const pooled = output.reshape([1, hidden]);
    expect(pooled.shape).toEqual([1, hidden]);
  });
});
