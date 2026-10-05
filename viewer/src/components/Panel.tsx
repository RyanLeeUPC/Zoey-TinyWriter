import { useState, type ReactNode } from "react";
import { ChevronDown, CircleHelp } from "lucide-react";

/**
 * A card on the lab page. Every panel carries a plain-language
 * "What am I looking at?" explainer, so nobody is ever left guessing.
 */
export function Panel({
  title,
  subtitle,
  help,
  actions,
  children,
  className = "",
}: {
  title: string;
  subtitle?: ReactNode;
  help?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <section className={`rounded-2xl border border-line bg-surface p-5 shadow-sm ${className}`}>
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
          {subtitle && <p className="mt-0.5 text-sm text-ink-2">{subtitle}</p>}
        </div>
        <div className="flex items-center gap-2">{actions}</div>
      </header>

      {help && (
        <div className="mb-4">
          <button
            onClick={() => setOpen((o) => !o)}
            className="inline-flex items-center gap-1.5 rounded-full bg-accent-wash px-3 py-1 text-sm font-medium text-accent-ink hover:brightness-95"
            aria-expanded={open}
          >
            <CircleHelp size={15} />
            What am I looking at?
            <ChevronDown size={15} className={`transition-transform ${open ? "rotate-180" : ""}`} />
          </button>
          {open && (
            <div className="mt-3 space-y-2 rounded-xl bg-surface-2 p-4 text-[15px] leading-relaxed text-ink-2 [&_b]:text-ink [&_b]:font-semibold">
              {help}
            </div>
          )}
        </div>
      )}

      {children}
    </section>
  );
}

/** Small segmented toggle used for panel options. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg bg-surface-2 p-0.5 text-sm" role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-2.5 py-1 font-medium transition-colors ${
            value === o.value ? "bg-surface text-ink shadow-sm" : "text-muted hover:text-ink-2"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
