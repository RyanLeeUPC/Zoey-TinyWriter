/**
 * The page's handle on TinyWriter. One shared worker for the whole app, so the
 * model downloads once and survives page changes.
 */
import { useSyncExternalStore } from "react";
import { MODEL_ID } from "../config";
import type { GenerateOptions, Lens, ModelInfo, TokenInfo, TokenTrace, WorkerEvent, WorkerRequest } from "./protocol";

export type Status = "idle" | "loading" | "ready" | "generating" | "error";

export interface ZoeyState {
  status: Status;
  progress: number; // 0..1 while downloading
  error?: string;
  info?: ModelInfo;
  /** Display text for every token id. */
  vocab: string[];
  /** The current story, one entry per token (including the hidden start marker). */
  tokens: TokenTrace[];
  /** Layer-by-layer views computed so far, by position. */
  lens: Map<number, Lens>;
  /** Dictionary lookups so far, by token id. */
  tokenInfo: Map<number, TokenInfo>;
  /** Increments with every new story, so views can reset their selection. */
  run: number;
}

const MODEL_URL = `./models/${MODEL_ID}`;

let state: ZoeyState = { status: "idle", progress: 0, vocab: [], tokens: [], lens: new Map(), tokenInfo: new Map(), run: 0 };
const lensRequested = new Set<number>();
const tokenRequested = new Set<number>();
const listeners = new Set<() => void>();
let worker: Worker | null = null;
let pending: TokenTrace[] = [];
let flushScheduled = false;

function set(patch: Partial<ZoeyState>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

function send(msg: WorkerRequest) {
  worker!.postMessage(msg);
}

/** Batch incoming tokens into one update per animation frame. */
function flush() {
  flushScheduled = false;
  if (pending.length) {
    set({ tokens: [...state.tokens, ...pending] });
    pending = [];
  }
}

function onEvent(e: MessageEvent<WorkerEvent>) {
  const ev = e.data;
  switch (ev.type) {
    case "progress":
      set({ progress: ev.loaded / ev.total });
      break;
    case "ready":
      set({ status: "ready", progress: 1, info: ev.info, vocab: ev.tokens });
      break;
    case "token":
      if (ev.run !== state.run) return;
      pending.push(ev.trace);
      if (!flushScheduled) {
        flushScheduled = true;
        requestAnimationFrame(flush);
      }
      break;
    case "lens":
      if (ev.run === state.run) set({ lens: new Map(state.lens).set(ev.pos, ev.lens) });
      break;
    case "lookup":
      set({ tokenInfo: new Map(state.tokenInfo).set(ev.info.id, ev.info) });
      break;
    case "done":
      if (ev.run !== state.run) return;
      flush();
      set({ status: "ready" });
      break;
    case "error":
      set({ status: "error", error: ev.message });
      break;
  }
}

export const zoey = {
  load() {
    if (worker) return;
    worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = onEvent;
    worker.onerror = (e) => set({ status: "error", error: e.message || "The model crashed." });
    set({ status: "loading", progress: 0 });
    send({ type: "load", url: new URL(MODEL_URL, document.baseURI).href });
  },
  generate(opts: GenerateOptions) {
    if (state.status !== "ready" && state.status !== "generating") return;
    const run = state.run + 1;
    pending = [];
    lensRequested.clear();
    set({ status: "generating", tokens: [], lens: new Map(), run });
    send({ type: "generate", run, ...opts });
  },
  /** Ask for the layer-by-layer view of one position (cached once computed). */
  requestLens(pos: number) {
    if (lensRequested.has(pos) || !state.tokens[pos]) return;
    lensRequested.add(pos);
    send({ type: "lens", run: state.run, pos });
  },
  /** Look a token up in the dictionary (cached once computed). */
  requestToken(id: number) {
    if (!worker || tokenRequested.has(id)) return;
    tokenRequested.add(id);
    send({ type: "lookup", id });
  },
  stop() {
    send({ type: "stop" });
    set({ status: "ready" });
  },
};

export function useZoey(): ZoeyState {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => state,
  );
}
