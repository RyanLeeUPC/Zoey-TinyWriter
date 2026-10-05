import type { ReactNode } from "react";
import { Link, NavLink } from "react-router";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "../lib/theme";
import { MODEL_ID, REPO_URL } from "../config";

export function Layout({ children }: { children: ReactNode }) {
  const { theme, toggle } = useTheme();
  return (
    <div className="min-h-screen">
      <header className="border-b border-line">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
          <Link to="/" className="flex items-center gap-2.5 font-semibold tracking-tight">
            <Logo />
            <span className="hidden text-lg whitespace-nowrap sm:inline">Zoey-TinyWriter</span>
          </Link>
          <nav className="flex items-center gap-0.5 text-sm whitespace-nowrap sm:gap-1">
            <NavLink to="/" end className={navClass}>
              Explore
            </NavLink>
            <NavLink to="/dictionary" className={navClass}>
              Dictionary
            </NavLink>
            <NavLink to={`/run/${MODEL_ID}`} className={navClass}>
              <span className="hidden sm:inline">How TinyWriter learned</span>
              <span className="sm:hidden">Training</span>
            </NavLink>
            <a href={REPO_URL} className="hidden rounded-lg px-3 py-1.5 text-ink-2 hover:bg-surface-2 hover:text-ink sm:inline">
              Code
            </a>
            <button
              onClick={toggle}
              aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
              className="grid size-9 place-items-center rounded-lg text-ink-2 hover:bg-surface-2 hover:text-ink"
            >
              {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
            </button>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 sm:px-6">{children}</main>
      <footer className="mx-auto mt-16 max-w-7xl border-t border-line px-4 py-8 text-sm text-muted sm:px-6">
        Zoey-TinyWriter is a teaching project. It stands on the shoulders of Andrej Karpathy's nanoGPT, Sebastian Raschka's{" "}
        <i>Build a Large Language Model (From Scratch)</i>, the Transformer Explainer, and the TinyStories dataset by
        Ronen Eldan &amp; Yuanzhi Li.
      </footer>
    </div>
  );
}

export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" fill="var(--accent)" />
      <path d="M10 10h12L10 22h12" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const navClass = ({ isActive }: { isActive: boolean }) =>
  `rounded-lg px-2.5 py-1.5 sm:px-3 ${isActive ? "bg-surface-2 font-medium text-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink"}`;
