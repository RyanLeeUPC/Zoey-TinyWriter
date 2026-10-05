import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type Theme = "light" | "dark";

const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({
  theme: "light",
  toggle: () => {},
});

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem("zoey-theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // storage unavailable (private mode etc.) - fall through to the OS setting
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Apply a theme to the page right away, so canvases that redraw on the change read the new colors. */
function apply(theme: Theme): Theme {
  document.documentElement.dataset.theme = theme;
  return theme;
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => apply(initialTheme()));

  useEffect(() => {
    try {
      localStorage.setItem("zoey-theme", theme);
    } catch {
      // ignore
    }
  }, [theme]);

  const toggle = () => setTheme(apply(theme === "light" ? "dark" : "light"));

  return (
    <ThemeContext.Provider value={{ theme, toggle }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);

/*
  Sequential blue ramp for probabilities: near-zero recedes into the surface,
  certainty is the strongest step. Each theme has its own stops.
*/
const RAMPS: Record<Theme, string[]> = {
  light: ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
  dark: ["#1a1a19", "#104281", "#1c5cab", "#2a78d6", "#6da7ec", "#cde2fb"],
};

const hexToRgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

export function rampColor(theme: Theme, t: number): string {
  const stops = RAMPS[theme];
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = hexToRgb(stops[i]);
  const b = hexToRgb(stops[i + 1]);
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/**
 * Probabilities are very lopsided (most are near 0), so we stretch the low
 * end with a square root - otherwise everything but the top guess is invisible.
 */
export const probToT = (p: number) => Math.sqrt(p);

/** Ink that stays readable on top of a ramp color. */
export function inkOn(theme: Theme, t: number): string {
  const strong = theme === "light" ? t > 0.55 : t > 0.75;
  if (theme === "light") return strong ? "#ffffff" : "#0b0b0b";
  return strong ? "#0b0b0b" : "#ffffff";
}
