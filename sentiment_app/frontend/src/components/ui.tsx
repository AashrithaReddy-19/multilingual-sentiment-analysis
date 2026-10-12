import type { ReactNode } from "react";
import type { ClassName, ModelInfo } from "../api";
import { CLASSES } from "../api";

export function PageHeader({ title, subtitle, children }: { title: string; subtitle?: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-slate-600">{subtitle}</p>}
      </div>
      {children}
    </header>
  );
}

export function Card({ title, children, className = "", actions }: { title?: ReactNode; children: ReactNode; className?: string; actions?: ReactNode }) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      {(title || actions) && (
        <div className="mb-4 flex items-center justify-between gap-3">
          {title && <h2 className="text-base font-semibold text-slate-900">{title}</h2>}
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-3 py-8 text-sm text-slate-600">
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600" aria-hidden />
      {label}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
      <p className="font-medium">Something went wrong</p>
      <p className="mt-1 break-words">{message}</p>
      <p className="mt-1 text-rose-700">Is the backend running on port 8000?</p>
      {onRetry && (
        <button onClick={onRetry} className="mt-3 rounded-md bg-rose-600 px-3 py-1.5 font-medium text-white hover:bg-rose-700">
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-6 text-center text-sm text-slate-600">
      <p className="font-medium text-slate-800">{title}</p>
      {children && <div className="mt-1">{children}</div>}
    </div>
  );
}

const STATUS_STYLE = {
  trained: "bg-emerald-50 text-emerald-800 ring-emerald-200",
  pending: "bg-amber-50 text-amber-800 ring-amber-200",
  skipped: "bg-slate-100 text-slate-700 ring-slate-300",
} as const;

export function StatusBadge({ status }: { status: keyof typeof STATUS_STYLE }) {
  const text = { trained: "Trained", pending: "Pending GPU run", skipped: "Skipped" }[status];
  return <span className={`inline-flex shrink-0 items-center whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${STATUS_STYLE[status]}`}>{text}</span>;
}

export const SENTIMENT_STYLE: Record<ClassName, { bar: string; text: string; chip: string; emoji: string }> = {
  Negative: { bar: "bg-rose-500", text: "text-rose-700", chip: "bg-rose-50 text-rose-800 ring-rose-200", emoji: "☹️" },
  Neutral: { bar: "bg-slate-500", text: "text-slate-700", chip: "bg-slate-100 text-slate-800 ring-slate-300", emoji: "😐" },
  Positive: { bar: "bg-emerald-500", text: "text-emerald-700", chip: "bg-emerald-50 text-emerald-800 ring-emerald-200", emoji: "🙂" },
};

export function SentimentChip({ label }: { label: ClassName }) {
  return <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${SENTIMENT_STYLE[label].chip}`}>{label}</span>;
}

export function ProbabilityBars({ probabilities }: { probabilities: { label: ClassName; probability: number }[] }) {
  return (
    <ul className="space-y-3" aria-label="Class probabilities">
      {probabilities.map((p) => (
        <li key={p.label}>
          <div className="mb-1 flex justify-between text-sm">
            <span className="font-medium text-slate-800">{p.label}</span>
            <span className="tabular-nums text-slate-600">{(p.probability * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100" role="progressbar" aria-label={`${p.label} probability`}
            aria-valuenow={Math.round(p.probability * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div className={`h-full rounded-full ${SENTIMENT_STYLE[p.label].bar} transition-all duration-500`} style={{ width: `${p.probability * 100}%` }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

export function ConfusionMatrix({ matrix, title }: { matrix: number[][]; title?: string }) {
  const max = Math.max(...matrix.flat(), 1);
  return (
    <figure>
      {title && <figcaption className="mb-2 text-sm font-medium text-slate-700">{title}</figcaption>}
      <div className="overflow-x-auto">
        <table className="text-sm">
          <thead>
            <tr>
              <th className="p-1 text-left text-xs font-normal text-slate-500">Actual ↓ / Predicted →</th>
              {CLASSES.map((c) => <th key={c} scope="col" className="p-1 text-xs font-medium text-slate-600">{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={CLASSES[i]}>
                <th scope="row" className="p-1 pr-2 text-left text-xs font-medium text-slate-600">{CLASSES[i]}</th>
                {row.map((v, j) => {
                  const a = v / max;
                  return (
                    <td key={j} className="p-0.5">
                      <div className={`flex h-12 w-16 items-center justify-center rounded-md tabular-nums sm:w-20 ${i === j ? "font-semibold" : ""}`}
                        style={{ backgroundColor: `rgba(79,70,229,${0.08 + a * 0.82})`, color: a > 0.55 ? "white" : "#1e293b" }}>
                        {v.toLocaleString()}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}

export function ModelScopeBanner({ model }: { model: ModelInfo }) {
  const telugu = model.scope === "telugu_only";
  return (
    <div className={`rounded-lg border px-4 py-3 text-sm ${telugu ? "border-amber-300 bg-amber-50 text-amber-900" : "border-indigo-200 bg-indigo-50 text-indigo-900"}`} role="note">
      <p className="font-semibold">{telugu ? "Telugu-only model currently loaded." : "Multilingual model loaded."}</p>
      <p className="mt-0.5">
        {model.name}
        {telugu && " — trained on Telugu (TeSent) only. Its predictions are not multilingual model output; English and code-mixed text are outside its training data."}
      </p>
      <p className="mt-0.5 text-xs opacity-80">Uses the current message only — not context-aware (Model C was not trained: no real conversation data).</p>
    </div>
  );
}

export function Field({ label, htmlFor, hint, children }: { label: string; htmlFor: string; hint?: string; children: ReactNode }) {
  return (
    <div>
      <label htmlFor={htmlFor} className="mb-1 block text-sm font-medium text-slate-800">{label}</label>
      {children}
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}
