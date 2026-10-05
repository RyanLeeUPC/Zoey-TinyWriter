// The trace format written by zoey/trace.py: training snapshots for the replay page.
// These types are the reference for its shape; change both sides together.

export interface LossPoint {
  step: number;
  train: number;
}

export interface CheckpointRef {
  step: number;
  file: string;
  train: number;
  val: number;
}

export interface Manifest {
  schema: number;
  id: string;
  title: string;
  description?: string;
  model: { name: string; config: Record<string, unknown>; params: number };
  train: {
    batch_size: number;
    max_steps: number;
    learning_rate: number;
    tokens_per_step: number;
    seconds?: number;
    device?: string;
  };
  tokenizer: { name: string; tokens: string[] };
  probe: { text: string; ids: number[] };
  prompts: string[];
  embedding_words?: { word: string; id: number; group: string }[];
  loss: LossPoint[];
  checkpoints: CheckpointRef[];
}

export interface Sample {
  prompt: string;
  ids: number[];
  probs: number[];
}

export interface Probe {
  top_ids: number[][];
  top_probs: number[][];
  target_probs: number[];
  loss: number;
}

export interface Snapshot {
  step: number;
  samples: Sample[];
  probe: Probe;
  views: {
    attention?: number[][][][];
    lens_top_ids?: number[][][];
    lens_top_probs?: number[][][];
    lens_target?: number[][];
    induction?: number[][];
    embedding_2d?: number[][];
  };
}
