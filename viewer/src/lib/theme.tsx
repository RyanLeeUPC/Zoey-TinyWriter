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
