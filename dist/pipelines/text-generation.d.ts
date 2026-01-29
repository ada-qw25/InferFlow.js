/**
 * edgeFlow.js - Text Generation Pipeline
 *
 * Autoregressive text generation with streaming support.
 * Supports GPT-2, LLaMA, Mistral, and other causal LM models.
 * Includes chat/conversation support with message history.
 */
import { BasePipeline, PipelineResult } from './base.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { EdgeFlowTensor } from '../core/tensor.js';
import { PipelineConfig, PipelineOptions } from '../core/types.js';
/**
 * LLM model loading progress callback
 */
export interface LLMLoadProgress {
    /** Stage: 'tokenizer' or 'model' */
    stage: 'tokenizer' | 'model';
    /** Bytes loaded */
    loaded: number;
    /** Total bytes */
    total: number;
    /** Progress percentage (0-100) */
    progress: number;
}
/**
 * Chat message
 */
export interface ChatMessage {
    /** Role: 'system', 'user', or 'assistant' */
    role: 'system' | 'user' | 'assistant';
    /** Message content */
    content: string;
}
/**
 * Chat template type
 */
export type ChatTemplateType = 'chatml' | 'llama2' | 'llama3' | 'mistral' | 'phi3' | 'alpaca' | 'vicuna' | 'custom';
/**
 * Text generation options
 */
export interface TextGenerationOptions {
    /** Maximum number of new tokens to generate */
    maxNewTokens?: number;
    /** Maximum total length (prompt + generated) */
    maxLength?: number;
    /** Minimum number of new tokens to generate */
    minNewTokens?: number;
    /** Sampling temperature (higher = more random) */
    temperature?: number;
    /** Top-k sampling (0 = disabled) */
    topK?: number;
    /** Top-p (nucleus) sampling (1.0 = disabled) */
    topP?: number;
    /** Repetition penalty (1.0 = disabled) */
    repetitionPenalty?: number;
    /** Stop sequences */
    stopSequences?: string[];
    /** Whether to do sampling (false = greedy) */
    doSample?: boolean;
    /** Number of sequences to return */
    numReturnSequences?: number;
    /** Return full text (including prompt) */
    returnFullText?: boolean;
    /** Callback for each generated token */
    onToken?: (token: string, tokenId: number) => void;
}
/**
 * Chat generation options
 */
export interface ChatOptions extends TextGenerationOptions {
    /** System prompt */
    systemPrompt?: string;
    /** Chat template type */
    templateType?: ChatTemplateType;
    /** Custom template (if templateType is 'custom') */
    customTemplate?: {
        systemPrefix?: string;
        systemSuffix?: string;
        userPrefix?: string;
        userSuffix?: string;
        assistantPrefix?: string;
        assistantSuffix?: string;
        separator?: string;
    };
}
/**
 * Text generation result
 */
export interface TextGenerationResult extends PipelineResult {
    /** Generated text */
    generatedText: string;
    /** Full text (prompt + generated) if returnFullText is true */
    fullText?: string;
    /** Generated token IDs */
    tokenIds: number[];
    /** Number of tokens generated */
    numTokens: number;
}
/**
 * Streaming generation event
 */
export interface GenerationStreamEvent {
    /** Current token */
    token: string;
    /** Token ID */
    tokenId: number;
    /** Generated text so far */
    generatedText: string;
    /** Whether generation is complete */
    done: boolean;
}
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
export declare class TextGenerationPipeline extends BasePipeline<string | string[], TextGenerationResult | TextGenerationResult[]> {
    private tokenizer;
    private eosTokenId;
    private llmModel;
    private modelsLoaded;
    private modelUrl;
    private tokenizerUrl;
    constructor(config?: PipelineConfig);
    /**
     * Check if model is loaded
     */
    get isModelLoaded(): boolean;
    /**
     * Set custom model URLs
     */
    setModelUrls(model: string, tokenizer: string): void;
    /**
     * Load model and tokenizer with progress callback
     */
    loadModel(onProgress?: (progress: LLMLoadProgress) => void): Promise<void>;
    /**
     * Fetch model with progress tracking
     */
    private fetchModelWithProgress;
    /**
     * Initialize pipeline (override to skip default model loading)
     */
    initialize(): Promise<void>;
    /**
     * Set tokenizer
     */
    setTokenizer(tokenizer: Tokenizer): void;
    /**
     * Preprocess - not used for text generation (handled in generateSingle)
     */
    protected preprocess(input: string | string[]): Promise<EdgeFlowTensor[]>;
    /**
     * Postprocess - not used for text generation (handled in generateSingle)
     */
    protected postprocess(_outputs: EdgeFlowTensor[], _options?: PipelineOptions): Promise<TextGenerationResult | TextGenerationResult[]>;
    /**
     * Generate text (non-streaming)
     */
    run(prompt: string | string[], options?: PipelineOptions & TextGenerationOptions): Promise<TextGenerationResult | TextGenerationResult[]>;
    /**
     * Generate text with streaming (async generator)
     */
    stream(prompt: string, options?: TextGenerationOptions): AsyncGenerator<GenerationStreamEvent>;
    /**
     * Generate a single sequence (non-streaming)
     */
    private generateSingle;
    /**
     * Generate next token using the model
     */
    private generateNextToken;
    /**
     * Greedy decoding (argmax)
     */
    private greedy;
    /**
     * Sample from probability distribution with top-k/top-p filtering
     */
    private sample;
    private conversationHistory;
    private chatTemplateType;
    /**
     * Set the chat template type
     */
    setChatTemplate(templateType: ChatTemplateType): void;
    /**
     * Apply chat template to messages
     */
    applyChatTemplate(messages: ChatMessage[], options?: ChatOptions): string;
    /**
     * ChatML template (used by many models including Qwen, Yi)
     */
    private applyChatMLTemplate;
    /**
     * Llama 2 template
     */
    private applyLlama2Template;
    /**
     * Llama 3 template
     */
    private applyLlama3Template;
    /**
     * Mistral template
     */
    private applyMistralTemplate;
    /**
     * Phi-3 template
     */
    private applyPhi3Template;
    /**
     * Alpaca template
     */
    private applyAlpacaTemplate;
    /**
     * Vicuna template
     */
    private applyVicunaTemplate;
    /**
     * Custom template
     */
    private applyCustomTemplate;
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
    chat(userMessage: string, options?: ChatOptions): Promise<TextGenerationResult>;
    /**
     * Stream chat response
     */
    chatStream(userMessage: string, options?: ChatOptions): AsyncGenerator<GenerationStreamEvent>;
    /**
     * Get conversation history
     */
    getConversationHistory(): ChatMessage[];
    /**
     * Set conversation history
     */
    setConversationHistory(messages: ChatMessage[]): void;
    /**
     * Clear conversation history
     */
    clearConversation(): void;
    /**
     * Remove last exchange (user message + assistant response)
     */
    undoLastExchange(): void;
}
/**
 * Create text generation pipeline
 */
export declare function createTextGenerationPipeline(config?: PipelineConfig): TextGenerationPipeline;
//# sourceMappingURL=text-generation.d.ts.map