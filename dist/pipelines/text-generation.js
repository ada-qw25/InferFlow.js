/**
 * edgeFlow.js - Text Generation Pipeline
 *
 * Autoregressive text generation with streaming support.
 * Supports GPT-2, LLaMA, Mistral, and other causal LM models.
 * Includes chat/conversation support with message history.
 */
import { BasePipeline } from './base.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { EdgeFlowTensor, softmax } from '../core/tensor.js';
import { runInferenceNamed, loadModelFromBuffer } from '../core/runtime.js';
// ============================================================================
// Default Model URLs (TinyLlama - quantized for browser)
// ============================================================================
const DEFAULT_LLM_MODELS = {
    model: 'https://huggingface.co/Xenova/TinyLlama-1.1B-Chat-v1.0/resolve/main/onnx/model_q4f16.onnx',
    tokenizer: 'https://huggingface.co/Xenova/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json',
};
// ============================================================================
// Text Generation Pipeline
// ============================================================================
/**
 * TextGenerationPipeline - Autoregressive text generation
 *
 * @example
 * ```typescript
 * const generator = await pipeline('text-generation', 'Xenova/gpt2');
 *
 * // Simple generation
 * const result = await generator.run('Once upon a time');
 * console.log(result.generatedText);
 *
 * // Streaming generation
 * for await (const event of generator.stream('Hello, ')) {
 *   process.stdout.write(event.token);
 * }
 * ```
 */
export class TextGenerationPipeline extends BasePipeline {
    tokenizer = null;
    eosTokenId = 50256; // GPT-2 default
    llmModel = null;
    modelsLoaded = false;
    // Custom model URLs
    modelUrl;
    tokenizerUrl;
    constructor(config) {
        super(config ?? {
            task: 'text-generation',
            model: 'default',
        });
        this.modelUrl = DEFAULT_LLM_MODELS.model;
        this.tokenizerUrl = DEFAULT_LLM_MODELS.tokenizer;
    }
    /**
     * Check if model is loaded
     */
    get isModelLoaded() {
        return this.modelsLoaded;
    }
    /**
     * Set custom model URLs
     */
    setModelUrls(model, tokenizer) {
        this.modelUrl = model;
        this.tokenizerUrl = tokenizer;
    }
    /**
     * Load model and tokenizer with progress callback
     */
    async loadModel(onProgress) {
        if (this.modelsLoaded)
            return;
        // Load tokenizer first (small, fast)
        onProgress?.({ stage: 'tokenizer', loaded: 0, total: 100, progress: 0 });
        try {
            const tokenizerResponse = await fetch(this.tokenizerUrl);
            if (!tokenizerResponse.ok) {
                throw new Error(`Failed to fetch tokenizer: ${tokenizerResponse.status}`);
            }
            const tokenizerJson = await tokenizerResponse.json();
            this.tokenizer = await Tokenizer.fromJSON(tokenizerJson);
            const specialIds = this.tokenizer.getSpecialTokenIds();
            this.eosTokenId = specialIds.eosTokenId ?? specialIds.sepTokenId ?? 2; // TinyLlama uses 2 as EOS
            onProgress?.({ stage: 'tokenizer', loaded: 100, total: 100, progress: 100 });
        }
        catch (error) {
            throw new Error(`Failed to load tokenizer: ${error}`);
        }
        // Load model with progress tracking
        onProgress?.({ stage: 'model', loaded: 0, total: 100, progress: 0 });
        const modelData = await this.fetchModelWithProgress(this.modelUrl, (loaded, total) => {
            onProgress?.({
                stage: 'model',
                loaded,
                total,
                progress: Math.round((loaded / total) * 100),
            });
        });
        this.llmModel = await loadModelFromBuffer(modelData, {
            runtime: 'wasm', // Uses ONNXRuntime which auto-detects WebGPU internally
        });
        this.model = this.llmModel;
        this.modelsLoaded = true;
    }
    /**
     * Fetch model with progress tracking
     */
    async fetchModelWithProgress(url, onProgress) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
        }
        const contentLength = response.headers.get('content-length');
        const total = contentLength ? parseInt(contentLength, 10) : 0;
        if (!response.body) {
            // Fallback if no streaming support
            const buffer = await response.arrayBuffer();
            onProgress(buffer.byteLength, buffer.byteLength);
            return buffer;
        }
        const reader = response.body.getReader();
        const chunks = [];
        let loaded = 0;
        while (true) {
            const { done, value } = await reader.read();
            if (done)
                break;
            chunks.push(value);
            loaded += value.length;
            onProgress(loaded, total || loaded);
        }
        // Combine chunks into ArrayBuffer
        const buffer = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) {
            buffer.set(chunk, offset);
            offset += chunk.length;
        }
        return buffer.buffer;
    }
    /**
     * Initialize pipeline (override to skip default model loading)
     */
    async initialize() {
        if (this.isReady)
            return;
        // Don't call super.initialize() - we handle model loading separately
        this.isReady = true;
    }
    /**
     * Set tokenizer
     */
    setTokenizer(tokenizer) {
        this.tokenizer = tokenizer;
        const specialIds = tokenizer.getSpecialTokenIds();
        this.eosTokenId = specialIds.eosTokenId ?? specialIds.sepTokenId ?? 50256;
    }
    /**
     * Preprocess - not used for text generation (handled in generateSingle)
     */
    async preprocess(input) {
        // For text generation, preprocessing is handled in generateNextToken
        const text = Array.isArray(input) ? input[0] ?? '' : input;
        if (!this.tokenizer) {
            // Return dummy tensor if no tokenizer
            return [new EdgeFlowTensor(new Float32Array([0]), [1], 'float32')];
        }
        const encoded = this.tokenizer.encode(text, {
            addSpecialTokens: false,
            padding: 'do_not_pad',
        });
        return [new EdgeFlowTensor(BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))), [1, encoded.inputIds.length], 'int64')];
    }
    /**
     * Postprocess - not used for text generation (handled in generateSingle)
     */
    async postprocess(_outputs, _options) {
        // For text generation, postprocessing is handled in generateSingle
        return {
            generatedText: '',
            tokenIds: [],
            numTokens: 0,
            processingTime: 0,
        };
    }
    /**
     * Generate text (non-streaming)
     */
    async run(prompt, options) {
        await this.initialize();
        const prompts = Array.isArray(prompt) ? prompt : [prompt];
        const results = await Promise.all(prompts.map(p => this.generateSingle(p, options ?? {})));
        return Array.isArray(prompt) ? results : results[0];
    }
    /**
     * Generate text with streaming (async generator)
     */
    async *stream(prompt, options = {}) {
        const startTime = performance.now();
        if (!this.tokenizer) {
            throw new Error('Tokenizer not set. Call setTokenizer() first.');
        }
        const { maxNewTokens = 50, maxLength = 512, temperature = 1.0, topK = 0, topP = 1.0, repetitionPenalty = 1.0, stopSequences = [], doSample = true, } = options;
        // Encode prompt
        const encoded = this.tokenizer.encode(prompt, {
            addSpecialTokens: false,
            padding: 'do_not_pad',
            truncation: false,
        });
        let inputIds = [...encoded.inputIds];
        const generatedIds = [];
        let generatedText = '';
        // Generation loop
        for (let i = 0; i < maxNewTokens; i++) {
            // Check max length
            if (inputIds.length >= maxLength)
                break;
            // Run model forward pass
            const nextTokenId = await this.generateNextToken(inputIds, temperature, topK, topP, repetitionPenalty, doSample);
            // Check for EOS
            if (nextTokenId === this.eosTokenId) {
                yield {
                    token: '',
                    tokenId: nextTokenId,
                    generatedText,
                    done: true,
                };
                break;
            }
            // Decode token
            const token = this.tokenizer.decode([nextTokenId], true);
            generatedIds.push(nextTokenId);
            inputIds.push(nextTokenId);
            generatedText += token;
            // Call token callback
            if (options.onToken) {
                options.onToken(token, nextTokenId);
            }
            // Check stop sequences
            let shouldStop = false;
            for (const stopSeq of stopSequences) {
                if (generatedText.endsWith(stopSeq)) {
                    generatedText = generatedText.slice(0, -stopSeq.length);
                    shouldStop = true;
                    break;
                }
            }
            yield {
                token,
                tokenId: nextTokenId,
                generatedText,
                done: shouldStop,
            };
            if (shouldStop)
                break;
        }
        // Final event
        const endTime = performance.now();
        console.log(`Generation completed in ${(endTime - startTime).toFixed(2)}ms`);
    }
    /**
     * Generate a single sequence (non-streaming)
     */
    async generateSingle(prompt, options) {
        const startTime = performance.now();
        if (!this.tokenizer) {
            throw new Error('Tokenizer not set. Call setTokenizer() first.');
        }
        const { maxNewTokens = 50, maxLength = 512, temperature = 1.0, topK = 0, topP = 1.0, repetitionPenalty = 1.0, stopSequences = [], doSample = true, returnFullText = false, } = options;
        // Encode prompt
        const encoded = this.tokenizer.encode(prompt, {
            addSpecialTokens: false,
            padding: 'do_not_pad',
            truncation: false,
        });
        let inputIds = [...encoded.inputIds];
        const generatedIds = [];
        // Generation loop
        for (let i = 0; i < maxNewTokens; i++) {
            // Check max length
            if (inputIds.length >= maxLength)
                break;
            // Run model forward pass
            const nextTokenId = await this.generateNextToken(inputIds, temperature, topK, topP, repetitionPenalty, doSample);
            // Check for EOS
            if (nextTokenId === this.eosTokenId)
                break;
            // Add to sequence
            generatedIds.push(nextTokenId);
            inputIds.push(nextTokenId);
            // Call token callback
            if (options.onToken) {
                const token = this.tokenizer.decode([nextTokenId], true);
                options.onToken(token, nextTokenId);
            }
            // Check stop sequences
            const currentText = this.tokenizer.decode(generatedIds, true);
            let shouldStop = false;
            for (const stopSeq of stopSequences) {
                if (currentText.endsWith(stopSeq)) {
                    shouldStop = true;
                    break;
                }
            }
            if (shouldStop)
                break;
        }
        // Decode generated text
        const generatedText = this.tokenizer.decode(generatedIds, true);
        const endTime = performance.now();
        return {
            generatedText,
            fullText: returnFullText ? prompt + generatedText : undefined,
            tokenIds: generatedIds,
            numTokens: generatedIds.length,
            processingTime: endTime - startTime,
        };
    }
    /**
     * Generate next token using the model
     */
    async generateNextToken(inputIds, temperature, topK, topP, repetitionPenalty, doSample) {
        if (!this.model) {
            throw new Error('Model not loaded');
        }
        const seqLen = inputIds.length;
        // Prepare named inputs
        const inputs = new Map();
        // input_ids: [1, seq_len]
        inputs.set('input_ids', new EdgeFlowTensor(BigInt64Array.from(inputIds.map(id => BigInt(id))), [1, seqLen], 'int64'));
        // attention_mask: [1, seq_len]
        inputs.set('attention_mask', new EdgeFlowTensor(BigInt64Array.from(inputIds.map(() => BigInt(1))), [1, seqLen], 'int64'));
        // position_ids: [1, seq_len] - sequential positions from 0 to seq_len-1
        inputs.set('position_ids', new EdgeFlowTensor(BigInt64Array.from(Array.from({ length: seqLen }, (_, i) => BigInt(i))), [1, seqLen], 'int64'));
        // TinyLlama has 22 layers with GQA (4 KV heads, head_dim=64)
        // For first inference without cache, provide empty past_key_values
        const numLayers = 22;
        const numKVHeads = 4;
        const headDim = 64;
        for (let i = 0; i < numLayers; i++) {
            // past_key_values.{i}.key: [batch, num_kv_heads, 0, head_dim]
            inputs.set(`past_key_values.${i}.key`, new EdgeFlowTensor(new Float32Array(0), [1, numKVHeads, 0, headDim], 'float32'));
            // past_key_values.{i}.value: [batch, num_kv_heads, 0, head_dim]
            inputs.set(`past_key_values.${i}.value`, new EdgeFlowTensor(new Float32Array(0), [1, numKVHeads, 0, headDim], 'float32'));
        }
        // Run inference with named inputs
        const outputs = await runInferenceNamed(this.model, inputs);
        if (!outputs || outputs.length === 0) {
            throw new Error('Model returned no outputs');
        }
        // Get logits for last token
        const logits = outputs[0];
        const logitsData = logits.toFloat32Array();
        const vocabSize = logits.shape[logits.shape.length - 1] ?? 50257;
        // Get logits for the last position
        const lastPositionLogits = new Float32Array(vocabSize);
        const offset = (inputIds.length - 1) * vocabSize;
        for (let i = 0; i < vocabSize; i++) {
            lastPositionLogits[i] = logitsData[offset + i] ?? 0;
        }
        // Apply repetition penalty
        if (repetitionPenalty !== 1.0) {
            for (const prevId of inputIds) {
                if (prevId < vocabSize) {
                    const score = lastPositionLogits[prevId] ?? 0;
                    lastPositionLogits[prevId] = score > 0
                        ? score / repetitionPenalty
                        : score * repetitionPenalty;
                }
            }
        }
        // Apply temperature
        if (temperature !== 1.0) {
            for (let i = 0; i < vocabSize; i++) {
                lastPositionLogits[i] = (lastPositionLogits[i] ?? 0) / temperature;
            }
        }
        // Convert to probabilities
        const logitsTensor = new EdgeFlowTensor(lastPositionLogits, [vocabSize], 'float32');
        const probs = softmax(logitsTensor).toFloat32Array();
        // Sample or greedy
        if (doSample) {
            return this.sample(probs, topK, topP);
        }
        else {
            return this.greedy(probs);
        }
    }
    /**
     * Greedy decoding (argmax)
     */
    greedy(probs) {
        let maxIdx = 0;
        let maxProb = probs[0] ?? 0;
        for (let i = 1; i < probs.length; i++) {
            if ((probs[i] ?? 0) > maxProb) {
                maxProb = probs[i] ?? 0;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    /**
     * Sample from probability distribution with top-k/top-p filtering
     */
    sample(probs, topK, topP) {
        // Create sorted indices
        const indices = Array.from({ length: probs.length }, (_, i) => i);
        indices.sort((a, b) => (probs[b] ?? 0) - (probs[a] ?? 0));
        // Apply top-k filtering
        let candidateIndices = indices;
        if (topK > 0 && topK < probs.length) {
            candidateIndices = indices.slice(0, topK);
        }
        // Apply top-p (nucleus) filtering
        if (topP < 1.0) {
            let cumulativeProb = 0;
            const filtered = [];
            for (const idx of candidateIndices) {
                filtered.push(idx);
                cumulativeProb += probs[idx] ?? 0;
                if (cumulativeProb >= topP)
                    break;
            }
            candidateIndices = filtered;
        }
        // Renormalize probabilities
        let totalProb = 0;
        for (const idx of candidateIndices) {
            totalProb += probs[idx] ?? 0;
        }
        // Sample
        const r = Math.random() * totalProb;
        let cumulative = 0;
        for (const idx of candidateIndices) {
            cumulative += probs[idx] ?? 0;
            if (cumulative >= r) {
                return idx;
            }
        }
        // Fallback
        return candidateIndices[0] ?? 0;
    }
    // ==========================================================================
    // Chat / Conversation Support
    // ==========================================================================
    conversationHistory = [];
    chatTemplateType = 'chatml';
    /**
     * Set the chat template type
     */
    setChatTemplate(templateType) {
        this.chatTemplateType = templateType;
    }
    /**
     * Apply chat template to messages
     */
    applyChatTemplate(messages, options) {
        const templateType = options?.templateType ?? this.chatTemplateType;
        switch (templateType) {
            case 'chatml':
                return this.applyChatMLTemplate(messages);
            case 'llama2':
                return this.applyLlama2Template(messages);
            case 'llama3':
                return this.applyLlama3Template(messages);
            case 'mistral':
                return this.applyMistralTemplate(messages);
            case 'phi3':
                return this.applyPhi3Template(messages);
            case 'alpaca':
                return this.applyAlpacaTemplate(messages);
            case 'vicuna':
                return this.applyVicunaTemplate(messages);
            case 'custom':
                return this.applyCustomTemplate(messages, options?.customTemplate ?? {});
            default:
                return this.applyChatMLTemplate(messages);
        }
    }
    /**
     * ChatML template (used by many models including Qwen, Yi)
     */
    applyChatMLTemplate(messages) {
        let prompt = '';
        for (const msg of messages) {
            prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
        }
        prompt += '<|im_start|>assistant\n';
        return prompt;
    }
    /**
     * Llama 2 template
     */
    applyLlama2Template(messages) {
        let prompt = '';
        let systemMsg = '';
        for (const msg of messages) {
            if (msg.role === 'system') {
                systemMsg = msg.content;
            }
            else if (msg.role === 'user') {
                if (systemMsg) {
                    prompt += `<s>[INST] <<SYS>>\n${systemMsg}\n<</SYS>>\n\n${msg.content} [/INST]`;
                    systemMsg = '';
                }
                else {
                    prompt += `<s>[INST] ${msg.content} [/INST]`;
                }
            }
            else if (msg.role === 'assistant') {
                prompt += ` ${msg.content} </s>`;
            }
        }
        return prompt;
    }
    /**
     * Llama 3 template
     */
    applyLlama3Template(messages) {
        let prompt = '<|begin_of_text|>';
        for (const msg of messages) {
            prompt += `<|start_header_id|>${msg.role}<|end_header_id|>\n\n${msg.content}<|eot_id|>`;
        }
        prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n';
        return prompt;
    }
    /**
     * Mistral template
     */
    applyMistralTemplate(messages) {
        let prompt = '<s>';
        for (const msg of messages) {
            if (msg.role === 'user') {
                prompt += `[INST] ${msg.content} [/INST]`;
            }
            else if (msg.role === 'assistant') {
                prompt += ` ${msg.content}</s>`;
            }
            else if (msg.role === 'system') {
                prompt += `[INST] ${msg.content}\n`;
            }
        }
        return prompt;
    }
    /**
     * Phi-3 template
     */
    applyPhi3Template(messages) {
        let prompt = '';
        for (const msg of messages) {
            prompt += `<|${msg.role}|>\n${msg.content}<|end|>\n`;
        }
        prompt += '<|assistant|>\n';
        return prompt;
    }
    /**
     * Alpaca template
     */
    applyAlpacaTemplate(messages) {
        let prompt = '';
        let instruction = '';
        let input = '';
        for (const msg of messages) {
            if (msg.role === 'system') {
                instruction = msg.content;
            }
            else if (msg.role === 'user') {
                input = msg.content;
            }
        }
        if (instruction) {
            prompt = `### Instruction:\n${instruction}\n\n`;
        }
        if (input) {
            prompt += `### Input:\n${input}\n\n`;
        }
        prompt += '### Response:\n';
        return prompt;
    }
    /**
     * Vicuna template
     */
    applyVicunaTemplate(messages) {
        let prompt = '';
        for (const msg of messages) {
            if (msg.role === 'system') {
                prompt += `${msg.content}\n\n`;
            }
            else if (msg.role === 'user') {
                prompt += `USER: ${msg.content}\n`;
            }
            else if (msg.role === 'assistant') {
                prompt += `ASSISTANT: ${msg.content}\n`;
            }
        }
        prompt += 'ASSISTANT:';
        return prompt;
    }
    /**
     * Custom template
     */
    applyCustomTemplate(messages, template) {
        const { systemPrefix = '', systemSuffix = '\n', userPrefix = 'User: ', userSuffix = '\n', assistantPrefix = 'Assistant: ', assistantSuffix = '\n', separator = '', } = template;
        let prompt = '';
        for (let i = 0; i < messages.length; i++) {
            const msg = messages[i];
            if (i > 0)
                prompt += separator;
            switch (msg.role) {
                case 'system':
                    prompt += `${systemPrefix}${msg.content}${systemSuffix}`;
                    break;
                case 'user':
                    prompt += `${userPrefix}${msg.content}${userSuffix}`;
                    break;
                case 'assistant':
                    prompt += `${assistantPrefix}${msg.content}${assistantSuffix}`;
                    break;
            }
        }
        prompt += assistantPrefix;
        return prompt;
    }
    /**
     * Chat with the model
     *
     * @example
     * ```typescript
     * const generator = await pipeline('text-generation', 'model');
     *
     * // Single turn
     * const response = await generator.chat('Hello, how are you?');
     *
     * // Multi-turn with history
     * const response1 = await generator.chat('What is AI?');
     * const response2 = await generator.chat('Can you give an example?');
     *
     * // With system prompt
     * const response = await generator.chat('Hello', {
     *   systemPrompt: 'You are a helpful assistant.',
     * });
     * ```
     */
    async chat(userMessage, options) {
        // Add system message if provided and not already present
        if (options?.systemPrompt &&
            (this.conversationHistory.length === 0 || this.conversationHistory[0]?.role !== 'system')) {
            this.conversationHistory.unshift({
                role: 'system',
                content: options.systemPrompt,
            });
        }
        // Add user message
        this.conversationHistory.push({
            role: 'user',
            content: userMessage,
        });
        // Apply chat template
        const prompt = this.applyChatTemplate(this.conversationHistory, options);
        // Generate response
        const result = await this.run(prompt, {
            ...options,
            stopSequences: [
                ...(options?.stopSequences ?? []),
                '<|im_end|>',
                '<|end|>',
                '<|eot_id|>',
                '</s>',
                '\n\nUser:',
                '\n\nHuman:',
            ],
        });
        // Add assistant response to history
        const response = Array.isArray(result) ? result[0] : result;
        this.conversationHistory.push({
            role: 'assistant',
            content: response.generatedText.trim(),
        });
        return response;
    }
    /**
     * Stream chat response
     */
    async *chatStream(userMessage, options) {
        // Add system message if provided
        if (options?.systemPrompt &&
            (this.conversationHistory.length === 0 || this.conversationHistory[0]?.role !== 'system')) {
            this.conversationHistory.unshift({
                role: 'system',
                content: options.systemPrompt,
            });
        }
        // Add user message
        this.conversationHistory.push({
            role: 'user',
            content: userMessage,
        });
        // Apply chat template
        const prompt = this.applyChatTemplate(this.conversationHistory, options);
        // Stream response
        let fullResponse = '';
        for await (const event of this.stream(prompt, {
            ...options,
            stopSequences: [
                ...(options?.stopSequences ?? []),
                '<|im_end|>',
                '<|end|>',
                '<|eot_id|>',
                '</s>',
            ],
        })) {
            fullResponse = event.generatedText;
            yield event;
        }
        // Add assistant response to history
        this.conversationHistory.push({
            role: 'assistant',
            content: fullResponse.trim(),
        });
    }
    /**
     * Get conversation history
     */
    getConversationHistory() {
        return [...this.conversationHistory];
    }
    /**
     * Set conversation history
     */
    setConversationHistory(messages) {
        this.conversationHistory = [...messages];
    }
    /**
     * Clear conversation history
     */
    clearConversation() {
        this.conversationHistory = [];
    }
    /**
     * Remove last exchange (user message + assistant response)
     */
    undoLastExchange() {
        // Remove assistant message
        if (this.conversationHistory.length > 0 &&
            this.conversationHistory[this.conversationHistory.length - 1]?.role === 'assistant') {
            this.conversationHistory.pop();
        }
        // Remove user message
        if (this.conversationHistory.length > 0 &&
            this.conversationHistory[this.conversationHistory.length - 1]?.role === 'user') {
            this.conversationHistory.pop();
        }
    }
}
// ============================================================================
// Factory Functions
// ============================================================================
/**
 * Create text generation pipeline
 */
export function createTextGenerationPipeline(config) {
    return new TextGenerationPipeline(config);
}
//# sourceMappingURL=text-generation.js.map