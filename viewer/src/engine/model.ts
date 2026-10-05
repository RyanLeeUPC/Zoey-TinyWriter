/**
 * TinyWriter's transformer, running in your browser.
 *
 * This is the same maths as zoey/models/gpt.py, written out by hand with
 * plain arrays: no GPU, no ML library. It processes one token at a time and
 * keeps every earlier token's keys and values (the "KV cache"), so each new
 * token only costs one pass through the network.
 *
 * Because we wrote every step ourselves, we can hand every intermediate
 * number to the viewer: attention weights, and the residual stream after
 * each layer.
 */

import type { TokenizerMeta } from "./tokenizer";

export interface ModelConfig {
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_head: number;
  d_model: number;
}

export interface ModelMeta {
  name: string;
  config: ModelConfig;
  params: number;
  tokenizer: TokenizerMeta;
  tensors: { name: string; shape: number[]; dtype: "int8" | "float32"; offset: number; scales?: number }[];
}

interface Layer {
  ln1w: Float32Array;
  ln1b: Float32Array;
  qkv: Float32Array; // (3d, d)
  proj: Float32Array; // (d, d)
  ln2w: Float32Array;
  ln2b: Float32Array;
  fc: Float32Array; // (4d, d)
  fcProj: Float32Array; // (d, 4d)
}

/** What the model produced after reading one token. */
export interface StepOutput {
  /** Scores for every possible next token. */
  logits: Float32Array;
  /** Attention weights: [layer][head][earlier position], flattened. */
  attention: Float32Array;
  /** The residual stream after the embedding and after each layer: [layer+1][d]. */
  residuals: Float32Array;
}

export class ZoeyModel {
  readonly config: ModelConfig;
  private readonly wte: Float32Array; // (vocab, d) - also the unembedding (weight tying)
  private readonly wpe: Float32Array; // (block, d)
  private readonly layers: Layer[];
  private readonly lnfW: Float32Array;
  private readonly lnfB: Float32Array;

  constructor(meta: ModelMeta, buffer: ArrayBuffer) {
    this.config = meta.config;
    const t = loadTensors(meta, buffer);
    this.wte = t["wte.weight"];
    this.wpe = t["wpe.weight"];
    this.layers = Array.from({ length: meta.config.n_layer }, (_, i) => {
      const p = `blocks.${i}.`;
      return {
        ln1w: t[p + "ln1.weight"],
        ln1b: t[p + "ln1.bias"],
        qkv: t[p + "attn.qkv.weight"],
        proj: t[p + "attn.proj.weight"],
        ln2w: t[p + "ln2.weight"],
        ln2b: t[p + "ln2.bias"],
        fc: t[p + "mlp.fc.weight"],
        fcProj: t[p + "mlp.proj.weight"],
      };
    });
    this.lnfW = t["ln_f.weight"];
    this.lnfB = t["ln_f.bias"];
  }

  newSession(): Session {
    return new Session(this);
  }

  /** Decode a residual-stream vector into next-token logits, as if it were the last layer. */
  unembed(residual: Float32Array, out = new Float32Array(this.config.vocab_size)): Float32Array {
    const d = this.config.d_model;
    const h = new Float32Array(d);
    layerNorm(residual, this.lnfW, this.lnfB, h);
    matvec(this.wte, h, out, this.config.vocab_size, d);
    return out;
  }

  /** A token's embedding: the d_model numbers TinyWriter learned for it. */
  embedding(id: number): Float32Array {
    const d = this.config.d_model;
    return this.wte.slice(id * d, (id + 1) * d);
  }

  /** @internal used by Session */
  get weights() {
    return { wte: this.wte, wpe: this.wpe, layers: this.layers };
  }
}

export class Session {
  /** How many tokens have been read so far. */
  pos = 0;
  private readonly keys: Float32Array[]; // per layer: (block, d)
  private readonly values: Float32Array[];

  constructor(private readonly model: ZoeyModel) {
    const { n_layer, block_size, d_model } = model.config;
    this.keys = Array.from({ length: n_layer }, () => new Float32Array(block_size * d_model));
    this.values = Array.from({ length: n_layer }, () => new Float32Array(block_size * d_model));
  }

  get full(): boolean {
    return this.pos >= this.model.config.block_size;
  }

  /** Read one more token and return everything the model computed. */
  step(token: number): StepOutput {
    const { n_layer, n_head, d_model: d } = this.model.config;
    const { wte, wpe, layers } = this.model.weights;
    const pos = this.pos;
    if (pos >= this.model.config.block_size) throw new Error("context is full");
    const hd = d / n_head;
    const T = pos + 1; // tokens visible to attention, including this one

    // 1. Embedding: what the token is + where it is.
    const x = new Float32Array(d);
    for (let i = 0; i < d; i++) x[i] = wte[token * d + i] + wpe[pos * d + i];

    const residuals = new Float32Array((n_layer + 1) * d);
    residuals.set(x, 0);
    const attention = new Float32Array(n_layer * n_head * T);

    const h = new Float32Array(d);
    const qkv = new Float32Array(3 * d);
    const y = new Float32Array(d);
    const tmp = new Float32Array(d);
    const hidden = new Float32Array(4 * d);
    const scores = new Float32Array(T);
    const scale = 1 / Math.sqrt(hd);

    layers.forEach((L, l) => {
      // 2. Attention.
      layerNorm(x, L.ln1w, L.ln1b, h);
      matvec(L.qkv, h, qkv, 3 * d, d);
      // Remember this token's key and value for every future token.
      this.keys[l].set(qkv.subarray(d, 2 * d), pos * d);
      this.values[l].set(qkv.subarray(2 * d, 3 * d), pos * d);
      const K = this.keys[l];
      const V = this.values[l];

      y.fill(0);
      for (let head = 0; head < n_head; head++) {
        const off = head * hd;
        // How well does my query match each earlier token's key?
        let max = -Infinity;
        for (let j = 0; j < T; j++) {
          let s = 0;
          for (let i = 0; i < hd; i++) s += qkv[off + i] * K[j * d + off + i];
          s *= scale;
          scores[j] = s;
          if (s > max) max = s;
        }
        // Softmax: turn scores into weights that sum to 1.
        let sum = 0;
        for (let j = 0; j < T; j++) {
          scores[j] = Math.exp(scores[j] - max);
          sum += scores[j];
        }
        const base = (l * n_head + head) * T;
        for (let j = 0; j < T; j++) {
          const w = scores[j] / sum;
          attention[base + j] = w;
          // Take that much of each earlier token's value.
          for (let i = 0; i < hd; i++) y[off + i] += w * V[j * d + off + i];
        }
      }
      matvec(L.proj, y, tmp, d, d);
      for (let i = 0; i < d; i++) x[i] += tmp[i]; // residual connection

      // 3. MLP.
      layerNorm(x, L.ln2w, L.ln2b, h);
      matvec(L.fc, h, hidden, 4 * d, d);
      for (let i = 0; i < hidden.length; i++) hidden[i] = gelu(hidden[i]);
      matvec(L.fcProj, hidden, tmp, d, 4 * d);
      for (let i = 0; i < d; i++) x[i] += tmp[i]; // residual connection

      residuals.set(x, (l + 1) * d);
    });

    this.pos++;
    // 4. Final LayerNorm + unembedding -> a score for every token.
    return { logits: this.model.unembed(x), attention, residuals };
  }
}

// ---------------------------------------------------------------- the maths

/** out = W @ x, where W is (rows, cols) stored row by row. */
function matvec(W: Float32Array, x: Float32Array, out: Float32Array, rows: number, cols: number) {
  for (let r = 0; r < rows; r++) {
    const o = r * cols;
    let s0 = 0,
      s1 = 0,
      s2 = 0,
      s3 = 0;
    let c = 0;
    for (; c + 3 < cols; c += 4) {
      s0 += W[o + c] * x[c];
      s1 += W[o + c + 1] * x[c + 1];
      s2 += W[o + c + 2] * x[c + 2];
      s3 += W[o + c + 3] * x[c + 3];
    }
    for (; c < cols; c++) s0 += W[o + c] * x[c];
    out[r] = s0 + s1 + s2 + s3;
  }
}

/** Normalize a vector to mean 0 / variance 1, then scale and shift. */
function layerNorm(x: Float32Array, w: Float32Array, b: Float32Array, out: Float32Array) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
  variance /= n;
  const inv = 1 / Math.sqrt(variance + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

/** GELU (tanh approximation), the MLP's non-linearity. */
function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

export function softmax(logits: Float32Array, temperature = 1): Float32Array {
  const out = new Float32Array(logits.length);
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp((logits[i] - max) / temperature);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/** Indices and values of the k largest entries. */
export function topK(probs: Float32Array, k: number): { ids: number[]; probs: number[] } {
  const ids: number[] = [];
  for (let i = 0; i < probs.length; i++) {
    if (ids.length < k) {
      ids.push(i);
      ids.sort((a, b) => probs[b] - probs[a]);
    } else if (probs[i] > probs[ids[k - 1]]) {
      ids[k - 1] = i;
      ids.sort((a, b) => probs[b] - probs[a]);
    }
  }
  return { ids, probs: ids.map((i) => probs[i]) };
}

// ---------------------------------------------------------------- loading

function loadTensors(meta: ModelMeta, buffer: ArrayBuffer): Record<string, Float32Array> {
  const out: Record<string, Float32Array> = {};
  for (const t of meta.tensors) {
    const size = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === "float32") {
      out[t.name] = new Float32Array(buffer, t.offset, size).slice();
    } else {
      // 8-bit: each row stored as integers in [-127, 127] plus one scale.
      const [rows, cols] = t.shape;
      const q = new Int8Array(buffer, t.offset, size);
      const scales = new Float32Array(buffer, t.scales!, rows);
      const w = new Float32Array(size);
      for (let r = 0; r < rows; r++) {
        const s = scales[r];
        for (let c = 0; c < cols; c++) w[r * cols + c] = q[r * cols + c] * s;
      }
      out[t.name] = w;
    }
  }
  return out;
}
