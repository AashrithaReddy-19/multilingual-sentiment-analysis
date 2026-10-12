import { useState } from "react";
import type { RunResult } from "../api";
import { CLASSES, LANG_LABEL, MODEL_C_SKIP, num, pct } from "../api";
import { useApi } from "../hooks";
import { Card, ConfusionMatrix, ErrorState, PageHeader, Spinner, StatusBadge } from "../components/ui";

export default function Comparison() {
  const { data, error, loading, reload } = useApi<RunResult[]>("/api/comparison");
  const [runId, setRunId] = useState<string>("multilingual");
  const run = data?.find((r) => r.run === runId);
  const trained = run?.models.filter((m) => m.status === "trained") ?? [];
  const base = run?.models.find((m) => m.id === "A" && m.status === "trained");
  const plus = run?.models.find((m) => m.id === "A+" && m.status === "trained");

  return (
    <>
      <PageHeader title="Model comparison" subtitle="Real test-set results read from the saved metric files. The two runs use different data and splits, so compare models within a run.">
        <div role="tablist" aria-label="Run" className="inline-flex rounded-lg border border-slate-300 bg-white p-1">
          {data?.map((r) => (
            <button key={r.run} role="tab" aria-selected={r.run === runId} onClick={() => setRunId(r.run)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium ${r.run === runId ? "bg-indigo-600 text-white" : "text-slate-700 hover:bg-slate-100"}`}>
              {r.run === "multilingual" ? "Multilingual" : "Telugu-only"}
            </button>
          ))}
        </div>
      </PageHeader>
      {loading && <Spinner />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {run && (
        <div className="space-y-6">
          <Card title={`${run.title}: overall`}>
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="border-b border-slate-200 text-slate-600">
                  <tr>
                    {["Model", "Input", "Status", "Accuracy", "Macro P", "Macro R", "Macro F1", "Weighted F1", "Best val. macro F1", "Test rows"].map((h) =>
                      <th key={h} scope="col" className="px-3 py-2 font-medium">{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {run.models.map((m) => (
                    <tr key={m.id} className="border-b border-slate-100 align-top">
                      <td className="px-3 py-3 font-medium text-slate-900">{m.id}: {m.name}</td>
                      <td className="px-3 py-3 font-mono text-xs text-slate-600">{m.input}</td>
                      <td className="px-3 py-3"><StatusBadge status={m.status} /></td>
                      {m.status === "trained" ? (
                        <>
                          <td className="px-3 py-3 tabular-nums">{num(m.accuracy)}</td>
                          <td className="px-3 py-3 tabular-nums">{num(m.macro_precision)}</td>
                          <td className="px-3 py-3 tabular-nums">{num(m.macro_recall)}</td>
                          <td className="px-3 py-3 font-semibold tabular-nums">{num(m.macro_f1)}</td>
                          <td className="px-3 py-3 tabular-nums">{num(m.weighted_f1)}</td>
                          <td className="px-3 py-3 tabular-nums">{num(m.best_validation_macro_f1)}</td>
                          <td className="px-3 py-3 tabular-nums">{m.n_test?.toLocaleString()}</td>
                        </>
                      ) : (
                        <td colSpan={7} className="px-3 py-3 text-slate-600">
                          {m.status === "skipped" ? <><span className="font-medium">{MODEL_C_SKIP}</span><br /><span className="text-xs">{m.explanation}</span></> : m.note}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {base && plus && (
            <Card title="Original baseline (Model A) vs Model A+ — same test split">
              <p className="mb-3 text-sm text-slate-600">{plus.note} Model A+ adds character n-grams (char_wb 3–5) to the word TF-IDF features and was chosen on validation macro F1 only; the test split was used once.</p>
              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm tabular-nums">
                  <thead className="border-b border-slate-200 text-slate-600">
                    <tr>{["Metric", "Model A (original)", "Model A+", "Difference"].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {([["Accuracy", base.accuracy, plus.accuracy], ["Macro F1", base.macro_f1, plus.macro_f1], ["Weighted F1", base.weighted_f1, plus.weighted_f1],
                      ...["telugu", "english", "code_mixed"].map((l) => [`${LANG_LABEL[l]} macro F1`, base.per_language?.[l]?.macro_f1, plus.per_language?.[l]?.macro_f1])] as [string, number | undefined, number | undefined][])
                      .map(([k, a, b]) => (
                        <tr key={k} className="border-b border-slate-100">
                          <td className="px-3 py-2">{k}</td><td className="px-3 py-2">{num(a)}</td><td className="px-3 py-2 font-semibold">{num(b)}</td>
                          <td className={`px-3 py-2 font-medium ${a != null && b != null && b >= a ? "text-emerald-700" : "text-rose-700"}`}>
                            {a != null && b != null ? `${b - a >= 0 ? "+" : ""}${(b - a).toFixed(4)}` : "—"}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {plus && trained.some((m) => !["A", "A+"].includes(m.id)) && (
            <Card title="Absolute change vs Model A+ (same multilingual test split)">
              <p className="mb-3 text-xs text-slate-500">See experiments/statistical_comparison.json for McNemar tests and bootstrap 95% CIs before calling a difference meaningful.</p>
              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm tabular-nums">
                  <thead className="border-b border-slate-200 text-slate-600">
                    <tr><th className="px-3 py-2 font-medium">Model</th>{["Accuracy", "Macro F1", ...CLASSES.map((c) => `${c} F1`), ...["english", "telugu", "code_mixed"].map((l) => `${LANG_LABEL[l]} macro F1`)].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {trained.filter((m) => !["A", "A+"].includes(m.id)).map((m) => {
                      const vals: [number | undefined, number | undefined][] = [[m.accuracy, plus.accuracy], [m.macro_f1, plus.macro_f1],
                        ...CLASSES.map((c) => [m.per_class?.[c]?.f1, plus.per_class?.[c]?.f1] as [number | undefined, number | undefined]),
                        ...["english", "telugu", "code_mixed"].map((l) => [m.per_language?.[l]?.macro_f1 ?? undefined, plus.per_language?.[l]?.macro_f1 ?? undefined] as [number | undefined, number | undefined])];
                      return (
                        <tr key={m.id} className="border-b border-slate-100">
                          <td className="px-3 py-2 font-medium">{m.id}: {m.name}</td>
                          {vals.map(([a, b], i) => <td key={i} className={`px-3 py-2 ${a != null && b != null && a - b >= 0 ? "text-emerald-700" : "text-rose-700"}`}>
                            {a != null && b != null ? `${a - b >= 0 ? "+" : ""}${(a - b).toFixed(4)}` : "—"}</td>)}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {trained.length > 0 && (
            <Card title="Macro F1 by language (test set)">
              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm">
                  <thead className="border-b border-slate-200 text-slate-600">
                    <tr><th className="px-3 py-2 font-medium">Model</th>{["telugu", "english", "code_mixed"].map((l) =>
                      <th key={l} className="px-3 py-2 font-medium">{LANG_LABEL[l]}</th>)}</tr>
                  </thead>
                  <tbody>
                    {trained.map((m) => (
                      <tr key={m.id} className="border-b border-slate-100">
                        <td className="px-3 py-2 font-medium">{m.id}: {m.name}</td>
                        {["telugu", "english", "code_mixed"].map((l) => {
                          const s = m.per_language?.[l];
                          return <td key={l} className="px-3 py-2 tabular-nums">
                            {s && s.macro_f1 !== null ? <>{pct(s.macro_f1)} <span className="text-xs text-slate-500">(n={s.n.toLocaleString()})</span></> : <span className="text-slate-400">not in this run</span>}
                          </td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          <div className="grid gap-6 lg:grid-cols-2">
            {trained.map((m) => (
              <Card key={m.id} title={`Model ${m.id}: per class and confusion matrix`}>
                <table className="mb-5 min-w-full text-left text-sm">
                  <thead className="text-slate-600"><tr>{["Class", "Precision", "Recall", "F1", "Support"].map((h) => <th key={h} className="py-1 pr-3 font-medium">{h}</th>)}</tr></thead>
                  <tbody>
                    {CLASSES.map((c) => {
                      const pc = m.per_class?.[c];
                      return <tr key={c} className="border-t border-slate-100 tabular-nums">
                        <td className="py-1.5 pr-3 font-medium">{c}</td><td className="pr-3">{num(pc?.precision)}</td>
                        <td className="pr-3">{num(pc?.recall)}</td><td className="pr-3">{num(pc?.f1)}</td><td>{pc?.support.toLocaleString()}</td>
                      </tr>;
                    })}
                  </tbody>
                </table>
                {m.confusion_matrix && <ConfusionMatrix matrix={m.confusion_matrix} title="Confusion matrix (test)" />}
                {m.confidence && (
                  <p className="mt-3 text-xs text-slate-500">Mean confidence {pct(m.confidence.mean)} · when correct {pct(m.confidence.mean_when_correct)} · when wrong {pct(m.confidence.mean_when_wrong)}</p>
                )}
              </Card>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
