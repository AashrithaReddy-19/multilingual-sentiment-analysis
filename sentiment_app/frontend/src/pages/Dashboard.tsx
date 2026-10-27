import { Link } from "react-router";
import type { Dashboard as DashboardData, ModelResult } from "../api";
import { CLASSES, LANG_LABEL, isDashboardVisible, modelName, pct } from "../api";
import { useApi } from "../hooks";
import { Card, ErrorState, PageHeader, Spinner } from "../components/ui";

const LANGS = ["telugu", "english", "code_mixed"];
const ENSEMBLE_IDS = new Set(["P", "E", "E-lang"]);
const reached75 = (m: ModelResult) => (m.accuracy ?? 0) >= 0.75 && (m.macro_f1 ?? 0) >= 0.75;

function Badge75() {
  return <span className="rounded-full bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-800 ring-1 ring-inset ring-amber-200">75%+ achieved</span>;
}

function Stat({ label, value, big = false }: { label: string; value?: number | null; big?: boolean }) {
  return (
    <div className="rounded-lg bg-white/70 p-3 text-center ring-1 ring-inset ring-indigo-100">
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className={`${big ? "text-3xl" : "text-lg"} font-semibold tabular-nums text-slate-900`}>{pct(value)}</dd>
    </div>
  );
}

function ProductionCard({ m }: { m: ModelResult }) {
  return (
    <section aria-labelledby="prod-title" className="rounded-2xl border border-indigo-200 bg-gradient-to-br from-indigo-50 to-white p-5 shadow-sm sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-indigo-700">Model status · active</p>
          <h2 id="prod-title" className="mt-1 text-xl font-semibold text-slate-900">{modelName(m)}</h2>
          {m.description && <p className="text-sm text-slate-600">{m.description}</p>}
        </div>
        <div className="flex items-center gap-2">
          {reached75(m) && <Badge75 />}
          <span className="rounded-full bg-indigo-600 px-3 py-1 text-xs font-semibold text-white">Production</span>
        </div>
      </div>
      <dl className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Accuracy" value={m.accuracy} big />
        <Stat label="Macro F1" value={m.macro_f1} big />
        <Stat label="Weighted F1" value={m.weighted_f1} />
        <Stat label="Macro recall" value={m.macro_recall} />
      </dl>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">F1 by class</p>
          <dl className="grid grid-cols-3 gap-2">
            {CLASSES.map((c) => <Stat key={c} label={c} value={m.per_class?.[c]?.f1} />)}
          </dl>
        </div>
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Macro F1 by language</p>
          <dl className="grid grid-cols-3 gap-2">
            {LANGS.map((l) => <Stat key={l} label={LANG_LABEL[l]} value={m.per_language?.[l]?.macro_f1} />)}
          </dl>
        </div>
      </div>
      <p className="mt-4 text-xs text-slate-500">Performance is measured on the held-out test set ({m.n_test?.toLocaleString()} texts).</p>
    </section>
  );
}

function ModelCard({ m }: { m: ModelResult }) {
  const langs = LANGS.filter((l) => m.per_language?.[l]?.macro_f1 != null);
  return (
    <div className="flex h-full flex-col rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-slate-900">{modelName(m)}</p>
          {m.description && <p className="text-xs text-slate-500">{m.description}</p>}
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <span className="rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-800 ring-1 ring-inset ring-emerald-200">Trained</span>
          {m.is_production && <span className="rounded-full bg-indigo-600 px-2.5 py-0.5 text-xs font-semibold text-white">Production</span>}
          {reached75(m) && <Badge75 />}
        </div>
      </div>
      <dl className="mt-4 grid grid-cols-3 gap-2 text-center">
        {([["Accuracy", m.accuracy], ["Macro F1", m.macro_f1], ["Weighted F1", m.weighted_f1],
          ["Macro precision", m.macro_precision], ["Macro recall", m.macro_recall]] as [string, number | null | undefined][]).map(([k, v]) => (
          <div key={k} className="rounded-md bg-slate-50 p-2">
            <dt className="text-[11px] text-slate-500">{k}</dt>
            <dd className="text-base font-semibold tabular-nums text-slate-900">{pct(v)}</dd>
          </div>
        ))}
      </dl>
      {m.per_class && (
        <p className="mt-3 text-xs text-slate-600">F1 by class: {CLASSES.map((c) => `${c} ${pct(m.per_class?.[c]?.f1)}`).join(" · ")}</p>
      )}
      {langs.length > 0 && (
        <p className="mt-1 text-xs text-slate-600">Macro F1 by language: {langs.map((l) => `${LANG_LABEL[l]} ${pct(m.per_language?.[l]?.macro_f1)}`).join(" · ")}</p>
      )}
    </div>
  );
}

export default function Dashboard() {
  const { data, error, loading, reload } = useApi<DashboardData>("/api/dashboard");
  const multi = data?.runs.find((r) => r.run === "multilingual");
  const production = multi?.models.find((m) => m.is_production && m.status === "trained");
  const others = (multi?.models ?? []).filter((m) => isDashboardVisible(m) && !m.is_production);
  const singles = others.filter((m) => !ENSEMBLE_IDS.has(m.id));
  const ensembles = others.filter((m) => ENSEMBLE_IDS.has(m.id));
  const telugu = (data?.runs.find((r) => r.run === "telugu_only")?.models ?? []).filter(isDashboardVisible);
  return (
    <>
      <PageHeader title="Multilingual Sentiment Analysis"
        subtitle="Sentiment classification for Telugu, English and Telugu-English code-mixed text · Negative · Neutral · Positive" />
      {loading && <Spinner />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {data && (
        <div className="space-y-6">
          <Card title="Datasets">
            <ul className="grid gap-3 md:grid-cols-3">
              {data.datasets.map((d) => (
                <li key={d.name} className="rounded-lg border border-slate-200 p-4">
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-semibold text-slate-900">{d.name}</p>
                    {d.checksums_match && <span className="text-xs font-medium text-emerald-700">✓ verified</span>}
                  </div>
                  <p className="mt-1 text-2xl font-semibold tabular-nums">{d.final_rows?.toLocaleString() ?? "—"}</p>
                  <p className="text-xs text-slate-500">rows after cleaning</p>
                  {d.url ? <a href={d.url} target="_blank" rel="noreferrer" className="mt-2 block break-words text-xs font-medium text-indigo-700 hover:underline">{d.source} ↗</a>
                    : <p className="mt-2 break-words text-xs text-slate-600">{d.source}</p>}
                </li>
              ))}
            </ul>
          </Card>
          {production && <ProductionCard m={production} />}
          {singles.length > 0 && (
            <Card title="Multilingual Models" actions={<Link to="/comparison" className="text-sm font-medium text-indigo-700 hover:underline">Compare →</Link>}>
              <p className="mb-3 text-xs text-slate-500">Individual trained models, evaluated on the same multilingual test set ({production?.n_test?.toLocaleString() ?? "15,124"} texts). The production model above answers predictions.</p>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {singles.map((m) => <ModelCard key={m.id} m={m} />)}
              </div>
            </Card>
          )}
          {ensembles.length > 0 && (
            <Card title="Enhanced-Ensemble Models">
              <p className="mb-3 text-xs text-slate-500">Probability ensembles of the multilingual models, on the same test set.</p>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {ensembles.map((m) => <ModelCard key={m.id} m={m} />)}
              </div>
            </Card>
          )}
          {telugu.length > 0 && (
            <Card title="Telugu-Only Models">
              <p className="mb-3 text-xs text-slate-500">Trained and evaluated on a separate Telugu-only split (TeSent, {telugu[0]?.n_test?.toLocaleString()} test texts); not directly comparable with the multilingual results.</p>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {telugu.map((m) => <ModelCard key={m.id} m={m} />)}
              </div>
            </Card>
          )}
        </div>
      )}
    </>
  );
}
