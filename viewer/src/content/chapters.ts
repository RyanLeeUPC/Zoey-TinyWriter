// The course roadmap. Chapters with a `runId` have a trained run to explore.

import { DOCS_URL as DOCS } from "../config";

export interface Chapter {
  number: number;
  title: string;
  tagline: string;
  runId?: string;
  doc: string;
}

export const CHAPTERS: Chapter[] = [
  {
    number: 1,
    title: "The bigram model",
    tagline: "Predict the next letter from only the current one. A whole model in one table.",
    runId: "01-bigram",
    doc: `${DOCS}/01-bigram.md`,
  },
  {
    number: 2,
    title: "Tokenization",
    tagline: "Teach TinyWriter to read in chunks like \"the\" and \"ing\" instead of single letters.",
    doc: `${DOCS}/02-tokenization.md`,
  },
  {
    number: 3,
    title: "Embeddings",
    tagline: "Turn each token into a list of numbers, and watch similar words drift together.",
    doc: `${DOCS}/03-embeddings.md`,
  },
  {
    number: 4,
    title: "Attention",
    tagline: "Let every token look back at earlier ones and decide what matters.",
    doc: `${DOCS}/04-attention.md`,
  },
  {
    number: 5,
    title: "The transformer",
    tagline: "Stack attention and MLP blocks into a real GPT, and watch induction heads appear.",
    doc: `${DOCS}/05-transformer.md`,
  },
  {
    number: 6,
    title: "Sampling",
    tagline: "Temperature, top-k, and top-p: how a model picks its words.",
    doc: `${DOCS}/06-sampling.md`,
  },
];

export const chapterForRun = (runId: string) => CHAPTERS.find((c) => c.runId === runId);
