# Trace format

Training writes snapshots that the viewer replays. This is the contract between [`zoey/trace.py`](../zoey/trace.py) (writer) and [`viewer/src/lib/types.ts`](../viewer/src/lib/types.ts) (reader). If you change one, change the other.

Everything lives under `viewer/public/runs/`:

```
runs/
  01-bigram/
    manifest.json         everything about the run except the snapshots
    step_0000000.json     one file per snapshot
    step_0000001.json
    ...
```

## `manifest.json`

| Field | Meaning |
|---|---|
| `schema` | Format version (currently `1`) |
| `id`, `title`, `chapter`, `seed` | From the config's `[run]` table |
| `model` | `{ name, config, params }` |
| `train` | The config's `[train]` table, plus `tokens_per_step`, `seconds`, `device` |
| `tokenizer` | `{ name, tokens }`, where `tokens[i]` is the display string for token id `i` |
| `probe` | `{ text, ids }`: the fixed test sentence |
| `prompts` | Prompts used for the samples |
| `embedding_words` | `[{ word, id, group }]`: words plotted on the embedding map (GPT runs) |
| `loss` | `[{ step, train }]`: training loss of individual batches |
| `checkpoints` | `[{ step, file, train, val }]`: one per snapshot, with averaged losses |

## Snapshot (`step_*.json`)

```jsonc
{
  "step": 120,
  "samples": [
    // one per prompt; ids = generated tokens, probs = probability of each pick
    { "prompt": "Once upon a time", "ids": [ ... ], "probs": [ ... ] }
  ],
  "probe": {
    "top_ids":      [[ ... ]],  // [position][k]: top-k next-token guesses
    "top_probs":    [[ ... ]],  // [position][k]: their probabilities
    "target_probs": [ ... ],    // [position]: probability given to the real next token
    "loss": 2.41                // average surprise on the probe sentence
  },
  "views": {
    // model-specific internals, from model.inspect()

    // bigram runs
    "bigram": [[ ... ]],        // [current][next] probability table

    // GPT runs (L = layers, H = heads, T = probe length, k = 5)
    "attention":      [ ... ],  // [L][H][T][T]: how much each position attended to each earlier one
    "lens_top_ids":   [ ... ],  // [L+1][T][k]: logit lens top guesses after embedding + each layer
    "lens_top_probs": [ ... ],  // [L+1][T][k]
    "lens_target":    [ ... ],  // [L+1][T-1]: probability of the real next token, per layer
    "induction":      [ ... ],  // [L][H]: induction score per head (0 to 1)
    "embedding_2d":   [ ... ]   // [words][2]: PCA of the embedding_words, aligned frame to frame
  }
}
```

## Adding a new view

1. Return it from your model's `inspect(idx)` method as a tensor. It gets rounded and written automatically.
2. Add its type to `Snapshot.views` in `viewer/src/lib/types.ts`.
3. Write a panel in `viewer/src/views/` and render it in `pages/RunPage.tsx` when the view is present.
