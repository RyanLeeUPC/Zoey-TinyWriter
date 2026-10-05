// Messages between the page and the model's Web Worker.

export interface ModelInfo {
  name: string;
  params: number;
  config: { vocab_size: number; block_size: number; n_layer: number; n_head: number; d_model: number };
}

export interface GenerateOptions {
  prompt: string;
  maxTokens: number;
  temperature: number;
  seed: number;
}

/**
 * Everything about one position in the text: the token there, and the
 * prediction TinyWriter made after reading it (i.e. about the *next* token).
 */
export interface TokenTrace {
  pos: number;
  id: number;
  text: string;
  /** "start" is the hidden start-of-story marker every story begins with. */
  source: "start" | "prompt" | "zoey";
  /** The token that actually came next, and the probability TinyWriter gave it. */
  next?: number;
  nextProb?: number;
  /** TinyWriter's top guesses for the next token (temperature 1). */
  top: { ids: number[]; probs: number[] };
  /** Attention weights from this position: [layer][head][earlier position], flattened. */
  attention: Float32Array;
}

/**
 * The "logit lens" for one position: the top guesses, and the probability of
 * what really came next, decoded after the embedding and after each layer.
 * Computed only when someone asks (it's ~40% of the work per token).
 */
export interface Lens {
  ids: number[][];
  probs: number[][];
  next: number[];
}

/** Everything the Dictionary page shows about one token. */
export interface TokenInfo {
  id: number;
  /** The token's embedding: its learned numbers. */
  vector: Float32Array;
  /** The most similar tokens (cosine similarity of embeddings), best first. */
  neighbors: { ids: number[]; sims: number[] };
  /** A typical "large" embedding value, so every fingerprint uses the same color scale. */
  scale: number;
}

export type WorkerRequest =
  | { type: "load"; url: string }
  | ({ type: "generate"; run: number } & GenerateOptions)
  | { type: "lens"; run: number; pos: number }
  | { type: "lookup"; id: number }
  | { type: "stop" };

export type WorkerEvent =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; info: ModelInfo; tokens: string[] }
  | { type: "token"; run: number; trace: TokenTrace }
  | { type: "lens"; run: number; pos: number; lens: Lens }
  | { type: "lookup"; info: TokenInfo }
  | { type: "done"; run: number; reason: "length" | "end" | "stopped" | "full" }
  | { type: "error"; message: string };
