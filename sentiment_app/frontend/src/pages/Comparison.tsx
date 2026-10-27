import type { ModelResult, RunResult } from "../api";
import { CLASSES, LANG_LABEL, isDashboardVisible, modelName, num, pct } from "../api";
import { useApi } from "../hooks";
import { Card, ConfusionMatrix, ErrorState, PageHeader, Spinner } from "../components/ui";

const LANGS = ["telugu", "english", "code_mixed"];
const delta = (a?: number | null, b?: number | null) => (a != null && b != null ? a - b : null);

function DeltaCell({ d }: { d: number | null }) {
  return <td className={`px-3 py-2 tabular-nums ${d == null ? "" : d >= 0 ? "text-emerald-700" : "text-rose-700"}`}>
    {d == null ? "—" : `${d >= 0 ? "+" : ""}${d.toFixed(4)}`}</td>;
}

function MetricsTable({ models }: { models: ModelResult[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-left text-sm">
        <thead className="border-b border-slate-200 text-slate-600">
          <tr>{["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Weighted F1", "Best val. macro F1", "Test rows"].map((h) =>
            <th key={h} scope="col" className="whitespace-nowrap px-3 py-2 font-medium">{h}</th>)}</tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.id} className="border-b border-slate-100 align-top">
              <td className="px-3 py-3">
                <span className="font-medium text-slate-900">{modelName(m)}</span>
                {m.is_production && <span className="ml-2 rounded-full bg-indigo-600 px-2 py-0.5 text-[11px] font-semibold text-white">Production</span>}
                {m.description && <span className="block text-xs text-slate-500">{m.description}</span>}
              </td>
              <td className="px-3 py-3 tabular-nums">{num(m.accuracy)}</td>
              <td className="px-3 py-3 tabular-nums">{num(m.macro_precision)}</td>
              <td className="px-3 py-3 tabular-nums">{num(m.macro_recall)}</td>
              <td className="px-3 py-3 font-semibold tabular-nums">{num(m.macro_f1)}</td>
              <td className="px-3 py-3 tabular-nums">{num(m.weighted_f1)}</td>
              <td className="px-3 py-3 tabular-nums">{num(m.best_validation_macro_f1)}</td>
              <td className="px-3 py-3 tabular-nums">{m.n_test?.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Comparison() {
  const { data, error, loading, reload } = useApi<RunResult[]>("/api/comparison");
  const multi = data?.find((r) => r.run === "multilingual");
  const shown = (multi?.models.filter(isDashboardVisible) ?? []).sort((a, b) => Number(!!b.is_production) - Number(!!a.is_production));
  const telugu = data?.find((r) => r.run === "telugu_only")?.models.filter(isDashboardVisible) ?? [];
  const base = multi?.models.find((m) => m.id === "A" && m.status === "trained");
  const plus = multi?.models.find((m) => m.id === "A+" && m.status === "trained");

  return (
    <>
      <PageHeader title="Model comparison" subtitle="Test-set results of every trained model. Multilingual and Telugu-only models use different test sets, so compare within a section." />
      {loading && <Spinner />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {multi && (
        <div className="space-y-6">
          <h2 className="text-lg font-semibold text-slate-900">Multilingual Models</h2>
          <Card title="Overall performance (multilingual test set)">
            <MetricsTable models={shown} />
          </Card>

          <Card title="Macro F1 by language">
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="border-b border-slate-200 text-slate-600">
                  <tr><th className="px-3 py-2 font-medium">Model</th>{LANGS.map((l) => <th key={l} className="px-3 py-2 font-medium">{LANG_LABEL[l]}</th>)}</tr>
                </thead>
                <tbody>
                  {shown.map((m) => (
                    <tr key={m.id} className="border-b border-slate-100">
                      <td className="px-3 py-2 font-medium">{modelName(m)}</td>
                      {LANGS.map((l) => {
                        const s = m.per_language?.[l];
                        return <td key={l} className="px-3 py-2 tabular-nums">
                          {s && s.macro_f1 !== null ? <>{pct(s.macro_f1)} <span className="text-xs text-slate-500">(n={s.n.toLocaleString()})</span></> : "—"}
                        </td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {plus && (
            <Card title="Improvement over A+">
              <p className="mb-3 text-xs text-slate-500">Absolute change against the Enhanced TF-IDF model (A+) on the same test set.</p>
              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm">
                  <thead className="border-b border-slate-200 text-slate-600">
                    <tr><th className="px-3 py-2 font-medium">Model</th>{["Accuracy", "Macro F1", ...CLASSES.map((c) => `${c} F1`), ...LANGS.map((l) => `${LANG_LABEL[l]} macro F1`)].map((h) =>
                      <th key={h} className="whitespace-nowrap px-3 py-2 font-medium">{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {shown.map((m) => (
                      <tr key={m.id} className="border-b border-slate-100">
                        <td className="px-3 py-2 font-medium">{modelName(m)}</td>
                        <DeltaCell d={delta(m.accuracy, plus.accuracy)} />
                        <DeltaCell d={delta(m.macro_f1, plus.macro_f1)} />
                        {CLASSES.map((c) => <DeltaCell key={c} d={delta(m.per_class?.[c]?.f1, plus.per_class?.[c]?.f1)} />)}
                        {LANGS.map((l) => <DeltaCell key={l} d={delta(m.per_language?.[l]?.macro_f1, plus.per_language?.[l]?.macro_f1)} />)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          <div className="grid gap-6 lg:grid-cols-2">
            {shown.map((m) => (
              <Card key={m.id} title={`${modelName(m)}: per class`}>
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

          {base && plus && (
            <Card title="TF-IDF Baseline (A) vs Enhanced TF-IDF (A+)">
              <p className="mb-3 text-sm text-slate-600">A+ extends the baseline with character n-gram features and improves performance across all three language groups.</p>
              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm tabular-nums">
                  <thead className="border-b border-slate-200 text-slate-600">
                    <tr>{["Metric", "A", "A+", "Difference"].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {([["Accuracy", base.accuracy, plus.accuracy], ["Macro F1", base.macro_f1, plus.macro_f1], ["Weighted F1", base.weighted_f1, plus.weighted_f1],
                      ...LANGS.map((l) => [`${LANG_LABEL[l]} macro F1`, base.per_language?.[l]?.macro_f1, plus.per_language?.[l]?.macro_f1])] as [string, number | null | undefined, number | null | undefined][])
                      .map(([k, x, y]) => (
                        <tr key={k} className="border-b border-slate-100">
                          <td className="px-3 py-2">{k}</td><td className="px-3 py-2">{num(x)}</td><td className="px-3 py-2 font-semibold">{num(y)}</td>
                          <DeltaCell d={delta(y, x)} />
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {telugu.length > 0 && (
            <>
              <h2 className="pt-2 text-lg font-semibold text-slate-900">Telugu-Only Models</h2>
              <Card title="Overall performance (Telugu-only test set)">
                <MetricsTable models={telugu} />
              </Card>
              <div className="grid gap-6 lg:grid-cols-2">
                {telugu.map((m) => (
                  <Card key={m.id} title={`${modelName(m)}: per class`}>
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
                    {m.confusion_matrix && <ConfusionMatrix matrix={m.confusion_matrix} title="Confusion matrix (Telugu-only test)" />}
                  </Card>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}
