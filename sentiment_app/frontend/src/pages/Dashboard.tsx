import { Link } from "react-router";
import type { Dashboard as DashboardData, ModelResult } from "../api";
import { CLASSES, LANG_LABEL, MODEL_C_SKIP, pct } from "../api";
import { useApi } from "../hooks";
import { Card, ErrorState, ModelScopeBanner, PageHeader, Spinner, StatusBadge } from "../components/ui";

function MetricCard({ m }: { m: ModelResult }) {
  return (
    <div className="rounded-lg border border-slate-200 p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Model {m.id}</p>
          <p className="text-sm font-semibold text-slate-900">{m.name}</p>
        </div>
        <StatusBadge status={m.status} />
      </div>
      {m.status === "trained" ? (
        <>
          <dl className="mt-4 grid grid-cols-3 gap-2 text-center">
            {[["Accuracy", m.accuracy], ["Macro F1", m.macro_f1], ["Weighted F1", m.weighted_f1],
              ["Macro precision", m.macro_precision], ["Macro recall", m.macro_recall]].map(([k, v]) => (
              <div key={k as string} className="rounded-md bg-slate-50 p-2">
                <dt className="text-[11px] text-slate-500">{k}</dt>
                <dd className="text-lg font-semibold tabular-nums text-slate-900">{pct(v as number)}</dd>
              </div>
            ))}
          </dl>
          {m.per_class && (
            <p className="mt-2 text-xs text-slate-500">
              F1 by class: {CLASSES.map((c) => `${c} ${pct(m.per_class?.[c]?.f1)}`).join(" · ")}
            </p>
          )}
        </>
      ) : (
        <p className="mt-3 text-sm text-slate-600">{m.status === "skipped" ? MODEL_C_SKIP : m.note}</p>
      )}
      {m.status === "trained" && m.note && (
        <p className="mt-2 text-xs text-indigo-700">{m.note}</p>
      )}
      {m.status === "trained" && m.per_language && (
        <p className="mt-3 text-xs text-slate-500">
          Macro F1 by language:{" "}
          {Object.entries(m.per_language).filter(([, v]) => v.macro_f1 !== null)
            .map(([k, v]) => `${LANG_LABEL[k]} ${pct(v.macro_f1)}`).join(" · ")}
        </p>
      )}
    </div>
  );
}

export default function Dashboard() {
  const { data, error, loading, reload } = useApi<DashboardData>("/api/dashboard");
  return (
    <>
      <PageHeader title="Sentiment Analysis: Telugu, English and Code-Mixed"
        subtitle="Telugu, English, and Telugu-English code-mixed conversational chat · Negative / Neutral / Positive. Every number on this page is read from saved, executed results; pending experiments show no numbers. Project goal: context-aware chat sentiment — the served models use the current message only, because no real conversation data was available (see Context status)." />
      {loading && <Spinner />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {data && (
        <div className="space-y-6">
          {data.serving_model && <ModelScopeBanner model={data.serving_model} />}
          {data.selection_note && <p className="text-sm text-slate-600">{data.selection_note}</p>}
          <Card title="Datasets (verified)">
            <ul className="grid gap-3 md:grid-cols-3">
              {data.datasets.map((d) => (
                <li key={d.name} className="rounded-lg border border-slate-200 p-4">
                  <div className="flex items-center justify-between">
                    <p className="font-semibold text-slate-900">{d.name}</p>
                    <span className={`text-xs font-medium ${d.checksums_match ? "text-emerald-700" : "text-rose-700"}`}>
                      {d.checksums_match ? "✓ checksums match" : "✗ checksum mismatch"}
                    </span>
                  </div>
                  <p className="mt-1 text-2xl font-semibold tabular-nums">{d.final_rows?.toLocaleString() ?? "—"}</p>
                  <p className="text-xs text-slate-500">rows after cleaning</p>
                  <p className="mt-2 break-all text-xs text-slate-600">{d.source}</p>
                </li>
              ))}
            </ul>
          </Card>
          {data.runs.map((run) => (
            <Card key={run.run} title={run.title} actions={<Link to="/comparison" className="text-sm font-medium text-indigo-700 hover:underline">Details →</Link>}>
              <div className={`grid gap-3 md:grid-cols-2 ${run.models.length > 3 ? "xl:grid-cols-4" : "xl:grid-cols-3"}`}>
                {run.models.map((m) => <MetricCard key={m.id} m={m} />)}
              </div>
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
