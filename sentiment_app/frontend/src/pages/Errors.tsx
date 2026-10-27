import { useEffect, useState } from "react";
import type { ErrorsPage } from "../api";
import { CLASSES, LANG_LABEL, pct } from "../api";
import { useApi } from "../hooks";
import { Card, EmptyState, ErrorState, Field, PageHeader, SentimentChip, Spinner } from "../components/ui";

const REASONS = ["low confidence prediction", "negation handling", "emoji ambiguity", "mixed-language ambiguity",
  "insufficient conversational context", "annotation ambiguity", "sarcasm/implicit sentiment", "class overlap"];
const PAGE = 25;

export default function Errors() {
  const allSources = useApi<{ run: string; tag: string; label: string; display_name: string; dashboard_visible: boolean; is_production: boolean }[]>("/api/errors/sources");
  const sources = { ...allSources, data: allSources.data?.filter((s) => s.dashboard_visible) ?? null };
  const [source, setSource] = useState("");
  const [q, setQ] = useState("");
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState("");
  const [actual, setActual] = useState("");
  const [predicted, setPredicted] = useState("");
  const [reason, setReason] = useState("");
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    if (source || !sources.data?.length) return;
    const first = sources.data.find((s) => s.is_production) ?? sources.data[0];      // default: the served model
    setSource(`${first.run}|${first.tag}`);
  }, [sources.data, source]);
  useEffect(() => { const t = setTimeout(() => setQuery(q), 300); return () => clearTimeout(t); }, [q]);
  useEffect(() => setOffset(0), [source, query, language, actual, predicted, reason]);

  const [run, tag] = source.split("|");
  const params = new URLSearchParams({ run: run ?? "", tag: tag ?? "", offset: String(offset), limit: String(PAGE) });
  if (query) params.set("q", query);
  if (language) params.set("language", language);
  if (actual) params.set("actual", actual);
  if (predicted) params.set("predicted", predicted);
  if (reason) params.set("reason", reason);
  const page = useApi<ErrorsPage>(source ? `/api/errors?${params}` : null);

  const select = (id: string, label: string, value: string, set: (v: string) => void, options: [string, string][]) => (
    <Field label={label} htmlFor={id}>
      <select id={id} value={value} onChange={(e) => set(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm">
        <option value="">All</option>
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </Field>
  );

  return (
    <>
      <PageHeader title="Error analysis" subtitle="Misclassified test examples, most confident first." />
      <div className="mb-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900" role="note">
        Likely error reasons are heuristic indicators for analysis and are not ground-truth explanations.
      </div>
      {sources.error && <ErrorState message={sources.error} onRetry={sources.reload} />}
      <Card className="mb-6">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          <div className="sm:col-span-2">
            <Field label="Model" htmlFor="src">
              <select id="src" value={source} onChange={(e) => setSource(e.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm">
                {sources.data?.map((s) => <option key={s.run + s.tag} value={`${s.run}|${s.tag}`}>{s.display_name}{s.is_production ? " (production)" : ""}</option>)}
              </select>
            </Field>
          </div>
          <div className="sm:col-span-2 lg:col-span-4">
            <Field label="Search text" htmlFor="q">
              <input id="q" type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. bagaledu, సినిమా, not"
                className="te w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
            </Field>
          </div>
          {select("lang", "Language", language, setLanguage, Object.entries(LANG_LABEL))}
          {select("act", "Actual", actual, setActual, CLASSES.map((c) => [c, c]))}
          {select("pred", "Predicted", predicted, setPredicted, CLASSES.map((c) => [c, c]))}
          <div className="sm:col-span-2 lg:col-span-3">{select("reason", "Heuristic reason", reason, setReason, REASONS.map((r) => [r, r]))}</div>
        </div>
      </Card>
      {page.loading && <Spinner />}
      {page.error && <ErrorState message={page.error} onRetry={page.reload} />}
      {page.data && page.data.rows.length === 0 && <EmptyState title="No misclassified examples match these filters" />}
      {page.data && page.data.rows.length > 0 && (
        <>
          <p className="mb-3 text-sm text-slate-600">{page.data.total.toLocaleString()} matching of {page.data.total_mistakes?.toLocaleString()} mistakes · showing {offset + 1}–{Math.min(offset + PAGE, page.data.total)}</p>
          <ul className="space-y-3">
            {page.data.rows.map((r, i) => (
              <li key={offset + i} className="rounded-lg border border-slate-200 bg-white p-4">
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="text-slate-500">Actual</span><SentimentChip label={r.actual_label} />
                  <span className="text-slate-500">→ predicted</span><SentimentChip label={r.predicted_label} />
                  <span className="text-slate-500">confidence {pct(r.confidence)}</span>
                  <span className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-700">{LANG_LABEL[r.language_type] ?? r.language_type}</span>
                  {r.source_dataset && <span className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-700">{r.source_dataset}</span>}
                </div>
                <p className="te mt-2 break-words text-slate-900">{r.cleaned_text}</p>
                <p className="mt-2 text-xs text-amber-800"><span className="font-medium">Likely reason (heuristic):</span> {r.likely_error_reason}</p>
              </li>
            ))}
          </ul>
          <div className="mt-4 flex justify-between">
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm disabled:opacity-40">← Previous</button>
            <button disabled={offset + PAGE >= page.data.total} onClick={() => setOffset(offset + PAGE)}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm disabled:opacity-40">Next →</button>
          </div>
        </>
      )}
    </>
  );
}
