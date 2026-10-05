/*
  The two color systems on the Explore page. Each has its own job, and they
  never share a meaning:

  - Confidence: how likely TinyWriter thought a token was. Four named bands, steps
    of one blue scale, light (unsure) to dark (sure).
  - Attention targets: which earlier word an attention head is looking at.
    One color per *word*, so every head looking at "Lily" matches "Lily"
    highlighted in the story. Six colors, always paired with a text label.

  Both sets were checked for colorblind separation and contrast with the
  dataviz palette validator, separately for light and dark mode.
*/
import type { Theme } from "./theme";

// ------------------------------------------------------------- confidence

export const BANDS = [
  { label: "long shot", min: 0, range: "under 5%" },
  { label: "possible", min: 0.05, range: "5–25%" },
  { label: "likely", min: 0.25, range: "25–60%" },
  { label: "confident", min: 0.6, range: "over 60%" },
] as const;

const BAND_COLORS: Record<Theme, string[]> = {
  light: ["#86b6ef", "#5598e7", "#2a78d6", "#104281"],
  dark: ["#184f95", "#2a78d6", "#6da7ec", "#cde2fb"],
};

export function bandIndex(p: number): number {
  for (let i = BANDS.length - 1; i >= 0; i--) if (p >= BANDS[i].min) return i;
  return 0;
}
export const bandColor = (theme: Theme, p: number) => BAND_COLORS[theme][bandIndex(p)];
export const bandSwatch = (theme: Theme, i: number) => BAND_COLORS[theme][i];

/** The confidence underline: a thick bar under the word, text left plain. */
export const underline = (color: string) => ({
  textDecorationLine: "underline",
  textDecorationColor: color,
  textDecorationThickness: "3px",
  textUnderlineOffset: "6px",
  textDecorationSkipInk: "none" as const,
});

// ------------------------------------------------------- attention targets

const TARGET_COLORS: Record<Theme, string[]> = {
  light: ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"],
  dark: ["#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
};
const NEUTRAL: Record<Theme, string> = { light: "#a3a29c", dark: "#6f6e69" };

export const TARGET_SLOTS = TARGET_COLORS.light.length;

/** Color for an attention-target slot; `null` = neutral (START, or a minor target). */
export const targetColor = (theme: Theme, slot: number | null) => (slot === null ? NEUTRAL[theme] : TARGET_COLORS[theme][slot]);

export function withAlpha(hex: string, alpha: number): string {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${Math.max(0, Math.min(1, alpha)).toFixed(3)})`;
}
