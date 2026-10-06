/// <reference lib="webworker" />
/**
 * Runs TinyWriter in a background thread so the page never freezes while it thinks.
 */
import { BPETokenizer } from "./tokenizer";
import { ZoeyModel, softmax, topK, type ModelMeta } from "./model";
import type { GenerateOptions, Lens, TokenInfo, TokenTrace, WorkerEvent, WorkerRequest } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

let model: ZoeyModel | null = null;
let tokenizer: BPETokenizer | null = null;
let currentRun = 0;

// What we keep about the latest story, so the lens can be computed on request.
let story: { run: number; residuals: Float32Array[]; next: (number | undefined)[] } = { run: 0, residuals: [], next: [] };

const post = (e: WorkerEvent, transfer: Transferable[] = []) => self.postMessage(e, transfer);

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (msg.type === "load") load(msg.url).catch((err) => post({ type: "error", message: String(err) }));
  else if (msg.type === "generate") {
    currentRun = msg.run;
    generate(msg.run, msg).catch((err) => post({ type: "error", message: String(err) }));
  } else if (msg.type === "lens") {
    if (msg.run === story.run && story.residuals[msg.pos]) post({ type: "lens", run: msg.run, pos: msg.pos, lens: lens(msg.pos) });
  } else if (msg.type === "lookup") {
    const info = tokenInfo(msg.id);
    post({ type: "lookup", info }, [info.vector.buffer]);
  } else if (msg.type === "stop") currentRun = -1;
};

async function load(url: string) {
  const meta: ModelMeta = await (await fetch(`${url}/model.json`)).json();
  const total = Math.max(...meta.tensors.map((t) => (t.scales ?? t.offset) + 4 * t.shape[0]));
  const res = await fetch(`${url}/weights.bin`);
  if (!res.ok || !res.body) throw new Error(`Couldn't download the model (${res.status})`);

  // Stream the download so the page can show a progress bar.
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    post({ type: "progress", loaded, total: Math.max(total, loaded) });
  }
  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) {
    bytes.set(p, o);
    o += p.length;
  }

  model = new ZoeyModel(meta, bytes.buffer);
  tokenizer = new BPETokenizer(meta.tokenizer);
  const tokens = Array.from({ length: meta.config.vocab_size }, (_, i) => tokenizer!.tokenText(i));
  post({ type: "ready", info: { name: meta.name, params: meta.params, config: meta.config }, tokens });
}

async function generate(run: number, opts: GenerateOptions) {
  if (!model || !tokenizer) throw new Error("model not loaded");
  const { block_size } = model.config;
  const random = mulberry32(opts.seed);

  // Every story in the training data starts right after an end-of-text
  // marker, so we start with one too: it tells TinyWriter "a new story begins".
  // A prompt too long to fit (with room left to write) keeps its *end*: that's
  // what the story continues from.
  const encoded = tokenizer.encode(opts.prompt);
  const room = block_size - 1 - opts.maxTokens;
  const prompt = encoded.length > room ? encoded.slice(encoded.length - room) : encoded;
  if (prompt.length < encoded.length) post({ type: "trimmed", run, dropped: encoded.length - prompt.length });
  const tokens = [tokenizer.eotId, ...prompt];
  const session = model.newSession();
  story = { run, residuals: [], next: [] };
  let written = 0;
  let reason: "length" | "end" | "stopped" | "full" = "length";

  for (let pos = 0; pos < tokens.length; pos++) {
    if (run !== currentRun) {
      // Stopped: the token picked last time around was never written, so forget it.
      story.next[pos - 1] = undefined;
      return post({ type: "done", run, reason: "stopped" });
    }

    const out = session.step(tokens[pos]);
    const probs = softmax(out.logits);

    // What comes next: the rest of the prompt, or TinyWriter's own pick.
    let next: number | undefined = tokens[pos + 1];
    if (next === undefined) {
      if (written >= opts.maxTokens) reason = "length";
      else if (session.full) reason = "full";
      else {
        next = sample(out.logits, opts.temperature, random);
        written++;
        if (next === tokenizer.eotId) reason = "end";
        else tokens.push(next);
      }
    }

    story.residuals[pos] = out.residuals;
    story.next[pos] = next;

    const trace: TokenTrace = {
      pos,
      id: tokens[pos],
      text: tokenizer.tokenText(tokens[pos]),
      source: pos === 0 ? "start" : pos <= prompt.length ? "prompt" : "zoey",
      next,
      nextProb: next === undefined ? undefined : probs[next],
      top: topK(probs, 8),
      attention: out.attention,
    };
    post({ type: "token", run, trace }, [out.attention.buffer]);

    if (next === tokenizer.eotId) break;
    // Let the page breathe (and hear a "stop") between tokens.
    await new Promise((r) => setTimeout(r, 0));
  }
  post({ type: "done", run, reason });
}

// ------------------------------------------------------------- dictionary

// Every embedding scaled to length 1, so a dot product is the cosine similarity.
let unitEmbeddings: Float32Array | null = null;
let embeddingScale = 0;

function prepareDictionary() {
  const { vocab_size: V, d_model: d } = model!.config;
  unitEmbeddings = new Float32Array(V * d);
  const all: number[] = [];
  for (let t = 0; t < V; t++) {
    const e = model!.embedding(t);
    let norm = 0;
    for (let i = 0; i < d; i++) norm += e[i] * e[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < d; i++) {
      unitEmbeddings[t * d + i] = e[i] / norm;
      if (i % 7 === 0) all.push(Math.abs(e[i]));
    }
  }
  // 98th percentile of |value|: big enough that most cells aren't maxed out.
  all.sort((a, b) => a - b);
  embeddingScale = all[Math.floor(all.length * 0.98)];
}

function tokenInfo(id: number): TokenInfo {
  if (!unitEmbeddings) prepareDictionary();
  const { vocab_size: V, d_model: d } = model!.config;
  const sims = new Float32Array(V);
  for (let t = 0; t < V; t++) {
    let s = 0;
    for (let i = 0; i < d; i++) s += unitEmbeddings![id * d + i] * unitEmbeddings![t * d + i];
    sims[t] = t === id ? -Infinity : s;
  }
  const top = topK(sims, 40);
  return { id, vector: model!.embedding(id), neighbors: { ids: top.ids, sims: top.probs }, scale: embeddingScale };
}

/** Logit lens: what would TinyWriter have said if it stopped after each layer? */
function lens(pos: number): Lens {
  const { n_layer, d_model: d } = model!.config;
  const residuals = story.residuals[pos];
  const next = story.next[pos];
  const out: Lens = { ids: [], probs: [], next: [] };
  for (let l = 0; l <= n_layer; l++) {
    const p = softmax(model!.unembed(residuals.subarray(l * d, (l + 1) * d)));
    const t = topK(p, 5);
    out.ids.push(t.ids);
    out.probs.push(t.probs);
    out.next.push(next === undefined ? 0 : p[next]);
  }
  return out;
}

/**
 * Pick the next token. Temperature reshapes the probabilities (low = safer,
 * high = wilder), and we only ever pick from the 50 most likely tokens so a
 * single unlucky roll can't derail the story.
 */
function sample(logits: Float32Array, temperature: number, random: () => number): number {
  const probs = softmax(logits, Math.max(temperature, 0.05));
  const { ids, probs: p } = topK(probs, 50);
  const total = p.reduce((a, b) => a + b, 0);
  let r = random() * total;
  for (let i = 0; i < ids.length; i++) {
    r -= p[i];
    if (r <= 0) return ids[i];
  }
  return ids[ids.length - 1];
}

/** A tiny seeded random number generator, so the same seed gives the same story. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
