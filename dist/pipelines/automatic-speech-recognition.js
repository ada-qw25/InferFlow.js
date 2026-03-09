/**
 * edgeFlow.js - Automatic Speech Recognition Pipeline
 *
 * Transcribe audio to text using Whisper ONNX models (encoder + decoder).
 */
import { BasePipeline, registerPipeline } from './base.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { AudioPreprocessor } from '../utils/preprocessor.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { loadModelData } from '../utils/model-loader.js';
import { loadModelFromBuffer, runInference, runInferenceNamed } from '../core/runtime.js';
// ============================================================================
// Default Model (Whisper-tiny, quantized encoder + decoder)
// ============================================================================
const DEFAULT_MODELS = {
    encoder: 'https://huggingface.co/Xenova/whisper-tiny/resolve/main/onnx/encoder_model_quantized.onnx',
    decoder: 'https://huggingface.co/Xenova/whisper-tiny/resolve/main/onnx/decoder_model_merged_quantized.onnx',
    tokenizer: 'https://huggingface.co/Xenova/whisper-tiny/resolve/main/tokenizer.json',
};
// Whisper special tokens
const SOT_TOKEN = 50258; // <|startoftranscript|>
const TRANSLATE_TOKEN = 50358; // <|translate|>
const TRANSCRIBE_TOKEN = 50359; // <|transcribe|>
const EOT_TOKEN = 50257; // <|endoftext|>
const NO_TIMESTAMPS_TOKEN = 50363; // <|notimestamps|>
const EN_TOKEN = 50259; // <|en|>
const MAX_DECODER_TOKENS = 448;
// ============================================================================
// ASR Pipeline
// ============================================================================
export class AutomaticSpeechRecognitionPipeline extends BasePipeline {
    audioPreprocessor;
    tokenizer = null;
    encoderModel = null;
    decoderModel = null;
    encoderUrl;
    decoderUrl;
    tokenizerUrl;
    constructor(config) {
        super(config ?? {
            task: 'automatic-speech-recognition',
            model: 'default',
        });
        this.encoderUrl = DEFAULT_MODELS.encoder;
        this.decoderUrl = DEFAULT_MODELS.decoder;
        this.tokenizerUrl = DEFAULT_MODELS.tokenizer;
        this.audioPreprocessor = new AudioPreprocessor({
            sampleRate: 16000,
            nMels: 80,
            nFft: 400,
            hopLength: 160,
            maxDuration: 30,
        });
    }
    async initialize() {
        await super.initialize();
        if (!this.tokenizer) {
            this.tokenizer = await Tokenizer.fromUrl(this.tokenizerUrl);
        }
        if (!this.encoderModel) {
            const data = await loadModelData(this.encoderUrl, { cache: this.config.cache ?? true });
            this.encoderModel = await loadModelFromBuffer(data);
        }
        if (!this.decoderModel) {
            const data = await loadModelData(this.decoderUrl, { cache: this.config.cache ?? true });
            this.decoderModel = await loadModelFromBuffer(data);
        }
    }
    setTokenizer(tokenizer) {
        this.tokenizer = tokenizer;
    }
    async run(input, options) {
        await this.initialize();
        const isBatch = Array.isArray(input);
        const inputs = isBatch ? input : [input];
        const opts = options ?? {};
        const results = [];
        for (const audio of inputs) {
            const result = await this.transcribeSingle(audio, opts);
            results.push(result);
        }
        return isBatch ? results : results[0];
    }
    async transcribeSingle(audio, options) {
        const startTime = performance.now();
        // 1. Preprocess audio → mel spectrogram
        const melTensor = await this.audioPreprocessor.process(audio);
        const melInput = new EdgeFlowTensor(melTensor.toFloat32Array(), [1, ...melTensor.shape], 'float32');
        // 2. Run encoder
        const encoderOutputs = await runInference(this.encoderModel, [melInput]);
        const encoderHidden = encoderOutputs[0];
        // 3. Autoregressive decoder loop
        const task = options.task ?? 'transcribe';
        const initialTokens = this.buildInitialTokens(task, options.language);
        const generatedTokens = await this.autoregressiveDecode(encoderHidden, initialTokens);
        // 4. Decode tokens to text
        const text = this.tokenizer.decode(generatedTokens, true);
        const result = {
            text: text.trim(),
            processingTime: performance.now() - startTime,
        };
        if (options.returnTimestamps) {
            result.chunks = this.extractTimestamps(generatedTokens, text);
        }
        return result;
    }
    buildInitialTokens(task, language) {
        const tokens = [SOT_TOKEN];
        tokens.push(language ? this.getLanguageToken(language) : EN_TOKEN);
        tokens.push(task === 'translate' ? TRANSLATE_TOKEN : TRANSCRIBE_TOKEN);
        tokens.push(NO_TIMESTAMPS_TOKEN);
        return tokens;
    }
    getLanguageToken(language) {
        // Whisper language tokens start at 50259 for English
        const langMap = {
            en: 50259, zh: 50260, de: 50261, es: 50262, ru: 50263,
            ko: 50264, fr: 50265, ja: 50266, pt: 50267, tr: 50268,
            pl: 50269, ca: 50270, nl: 50271, ar: 50272, sv: 50273,
            it: 50274, id: 50275, hi: 50276, fi: 50277, vi: 50278,
        };
        return langMap[language.toLowerCase()] ?? EN_TOKEN;
    }
    /**
     * Autoregressive decoder loop similar to text-generation.
     * Feeds encoder hidden states + growing token sequence to decoder.
     */
    async autoregressiveDecode(encoderHidden, initialTokens) {
        const tokens = [...initialTokens];
        for (let step = 0; step < MAX_DECODER_TOKENS; step++) {
            const decoderInputIds = new EdgeFlowTensor(BigInt64Array.from(tokens.map(t => BigInt(t))), [1, tokens.length], 'int64');
            const namedInputs = new Map();
            namedInputs.set('input_ids', decoderInputIds);
            namedInputs.set('encoder_hidden_states', encoderHidden);
            const decoderOutputs = await runInferenceNamed(this.decoderModel, namedInputs);
            const logits = decoderOutputs[0].toFloat32Array();
            // Get logits for the last token position
            const vocabSize = logits.length / tokens.length;
            const lastTokenLogits = logits.slice((tokens.length - 1) * vocabSize);
            // Greedy: argmax
            let bestId = 0;
            let bestVal = lastTokenLogits[0] ?? -Infinity;
            for (let i = 1; i < lastTokenLogits.length; i++) {
                if ((lastTokenLogits[i] ?? -Infinity) > bestVal) {
                    bestVal = lastTokenLogits[i] ?? -Infinity;
                    bestId = i;
                }
            }
            if (bestId === EOT_TOKEN)
                break;
            tokens.push(bestId);
        }
        // Strip initial tokens to return only generated tokens
        return tokens.slice(initialTokens.length);
    }
    extractTimestamps(_tokenIds, text) {
        // Simplified timestamp extraction: split by punctuation
        const words = text.split(/\s+/).filter(w => w.length > 0);
        const chunks = [];
        const wordsPerSecond = 2.5;
        let chunkText = '';
        let chunkStart = 0;
        for (let i = 0; i < words.length; i++) {
            chunkText += (chunkText ? ' ' : '') + words[i];
            if ((i + 1) % 5 === 0 || i === words.length - 1) {
                const duration = chunkText.split(/\s+/).length / wordsPerSecond;
                chunks.push({
                    text: chunkText,
                    start: chunkStart,
                    end: chunkStart + duration,
                });
                chunkStart = chunkStart + duration;
                chunkText = '';
            }
        }
        return chunks;
    }
    async processLongAudio(audio, options = {}) {
        const chunkDuration = options.chunkDuration ?? 30;
        const chunkOverlap = options.chunkOverlap ?? 5;
        const rawTensor = await this.audioPreprocessor.processRaw(audio);
        const audioData = rawTensor.toFloat32Array();
        const sampleRate = 16000;
        const chunkSamples = chunkDuration * sampleRate;
        const overlapSamples = chunkOverlap * sampleRate;
        const stepSamples = chunkSamples - overlapSamples;
        const chunks = [];
        for (let start = 0; start < audioData.length; start += stepSamples) {
            const end = Math.min(start + chunkSamples, audioData.length);
            const chunkAudio = audioData.slice(start, end);
            const chunkResult = await this.run(new Float32Array(chunkAudio), options);
            if (chunkResult.chunks) {
                const timeOffset = start / sampleRate;
                chunkResult.chunks = chunkResult.chunks.map(c => ({
                    ...c,
                    start: c.start + timeOffset,
                    end: c.end + timeOffset,
                }));
            }
            chunks.push(chunkResult);
        }
        const mergedText = chunks.map(c => c.text).join(' ');
        const mergedChunks = chunks.flatMap(c => c.chunks ?? []);
        return {
            text: mergedText,
            chunks: mergedChunks,
        };
    }
    async preprocess(input) {
        const inputs = Array.isArray(input) ? input : [input];
        const tensors = await Promise.all(inputs.map(audio => this.audioPreprocessor.process(audio)));
        if (tensors.length === 1) {
            const t = tensors[0];
            return [new EdgeFlowTensor(t.toFloat32Array(), [1, ...t.shape], 'float32')];
        }
        return tensors;
    }
    async postprocess(outputs, options) {
        const opts = options ?? {};
        const returnTimestamps = opts.returnTimestamps ?? false;
        if (!outputs[0]) {
            return { text: '' };
        }
        const outputData = outputs[0].toFloat32Array();
        const shape = outputs[0].shape;
        const text = this.decodeOutput(outputData, shape);
        const result = { text };
        if (returnTimestamps) {
            result.chunks = this.extractTimestamps([], text);
        }
        return result;
    }
    decodeOutput(data, shape) {
        const seqLen = shape[1] ?? data.length;
        const vocabSize = shape[2] ?? 1;
        const tokenIds = [];
        if (vocabSize > 1) {
            for (let i = 0; i < seqLen; i++) {
                const offset = i * vocabSize;
                let maxIdx = 0;
                let maxVal = data[offset] ?? -Infinity;
                for (let j = 1; j < vocabSize; j++) {
                    if ((data[offset + j] ?? -Infinity) > maxVal) {
                        maxVal = data[offset + j] ?? -Infinity;
                        maxIdx = j;
                    }
                }
                tokenIds.push(maxIdx);
            }
        }
        else {
            for (let i = 0; i < data.length; i++) {
                tokenIds.push(Math.round(data[i] ?? 0));
            }
        }
        if (this.tokenizer) {
            return this.tokenizer.decode(tokenIds, true);
        }
        return tokenIds.join(' ');
    }
}
// ============================================================================
// Factory
// ============================================================================
export function createASRPipeline(config) {
    return new AutomaticSpeechRecognitionPipeline(config);
}
registerPipeline('automatic-speech-recognition', (config) => new AutomaticSpeechRecognitionPipeline(config));
//# sourceMappingURL=automatic-speech-recognition.js.map