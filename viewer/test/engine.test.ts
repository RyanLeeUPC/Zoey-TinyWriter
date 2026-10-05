// The browser engine must compute exactly what PyTorch computes.
// Reference values come from `uv run python -m zoey.export <model>`.
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { BPETokenizer } from "../src/engine/tokenizer";
import { ZoeyModel, softmax, topK, type ModelMeta } from "../src/engine/model";

const MODELS = ["zoey"].filter((m) => existsSync(new URL(`./${m}.reference.json`, import.meta.url)));

for (const name of MODELS) {
  const dir = new URL(`../public/models/${name}/`, import.meta.url);
  const meta: ModelMeta = JSON.parse(readFileSync(new URL("model.json", dir), "utf-8"));
  const bin = readFileSync(new URL("weights.bin", dir));
  const weights = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const ref = JSON.parse(readFileSync(new URL(`./${name}.reference.json`, import.meta.url), "utf-8"));

  describe(`${name}: tokenizer`, () => {
    const tok = new BPETokenizer(meta.tokenizer);
    it("encodes exactly like the Python tokenizer", () => {
      for (const [text, ids] of Object.entries(ref.tokenizer_cases)) expect(tok.encode(text)).toEqual(ids);
    });
    it("round-trips text", () => {
      for (const text of Object.keys(ref.tokenizer_cases)) expect(tok.decode(tok.encode(text))).toBe(text);
    });
  });

  describe(`${name}: model`, () => {
    const model = new ZoeyModel(meta, weights);
    const session = model.newSession();
    let last!: ReturnType<typeof session.step>;
    for (const id of ref.ids as number[]) last = session.step(id);

    it("produces the same next-token scores as PyTorch", () => {
      let maxDiff = 0;
      ref.last_logits.forEach((v: number, i: number) => (maxDiff = Math.max(maxDiff, Math.abs(v - last.logits[i]))));
      expect(maxDiff).toBeLessThan(1e-3);
    });

    it("produces the same attention weights", () => {
      const { n_layer, n_head } = meta.config;
      const T = ref.ids.length;
      for (let l = 0; l < n_layer; l++)
        for (let h = 0; h < n_head; h++)
          for (let j = 0; j < T; j++)
            expect(last.attention[(l * n_head + h) * T + j]).toBeCloseTo(ref.last_attention[l][h][j], 4);
    });

    it("produces the same layer-by-layer (logit lens) guesses", () => {
      const d = meta.config.d_model;
      ref.last_lens_top_ids.forEach((ids: number[], layer: number) => {
        const probs = softmax(model.unembed(last.residuals.subarray(layer * d, (layer + 1) * d)));
        expect(topK(probs, 3).ids).toEqual(ids.slice(0, 3));
      });
    });
  });
}
