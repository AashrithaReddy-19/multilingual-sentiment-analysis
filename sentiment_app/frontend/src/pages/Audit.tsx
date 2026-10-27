import { useApi } from "../hooks";
import { Card, EmptyState, ErrorState, PageHeader, Spinner } from "../components/ui";
import { LANG_LABEL } from "../api";

interface AuditData {
  available: boolean;
  provenance: Record<string, { source: string; commit?: string; licence_note: string; all_match: boolean; files: Record<string, boolean>; citation?: string[] }> | null;
  label_mapping: { documented_mappings: Record<string, Record<string, string>>; sources: Record<string, string> } | null;
  cleaning: {
    per_dataset: Record<string, Record<string, number>>;
    total_removed_exact_duplicates: number;
    conflicting_labels: { texts_with_more_than_one_label: number; rows_in_those_groups: number; decision: string; by_dataset: Record<string, number> };
    split: { method: string; sizes: Record<string, number>; fractions: Record<string, number> };

    negation_words_preserved: Record<string, number>;
  } | null;
  class_distributions: { split: string; language_type: string; label_name: string; rows: number }[];
  leakage_checks: { cleaned_text: Record<string, number>; passed: boolean } | null;
}

const STAGES: [string, string][] = [
  ["initial_rows", "Rows in source"], ["removed_missing_text", "Removed: missing text"], ["removed_missing_label", "Removed: missing label"],
  ["removed_empty_after_cleaning", "Removed: empty after cleaning"], ["removed_exact_duplicate_text_label_pairs", "Removed: exact duplicate text-label pairs"],
  ["final_rows", "Final rows"],
];

export default function Audit() {
  const { data, error, loading, reload } = useApi<AuditData>("/api/audit");
  return (
    <>
      <PageHeader title="Data audit" subtitle="Sources, documented label mappings, cleaning statistics, class distributions, provenance and leakage checks for the multilingual run." />
      {loading && <Spinner />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {data && !data.available && <EmptyState title="No prepared data found">Run the data pipeline first (see README).</EmptyState>}
      {data?.available && data.cleaning && (
        <div className="space-y-6">
          <Card title="Sources and documented label mappings">
            <div className="grid gap-4 md:grid-cols-3">
              {Object.entries(data.provenance ?? {}).map(([name, p]) => (
                <div key={name} className="rounded-lg border border-slate-200 p-4 text-sm">
                  <p className="font-semibold text-slate-900">{name}</p>
                  <p className="mt-1 break-all text-xs text-slate-600">{p.source}{p.commit ? ` @ ${p.commit.slice(0, 7)}` : ""}</p>
                  <p className={`mt-2 text-xs font-medium ${p.all_match ? "text-emerald-700" : "text-rose-700"}`}>
                    {p.all_match ? `✓ ${Object.keys(p.files).length} files match audited SHA-256` : "✗ checksum mismatch"}
                  </p>
                  <p className="mt-2 text-xs text-slate-700"><span className="font-medium">Licence / use:</span> {p.licence_note}</p>
                  <div className="mt-3">
                    <p className="text-xs font-medium text-slate-800">Mapping</p>
                    <ul className="mt-1 flex flex-wrap gap-1">
                      {Object.entries(data.label_mapping?.documented_mappings[name] ?? {}).map(([k, v]) =>
                        <li key={k} className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[11px]">{k} → {v}</li>)}
                    </ul>
                    <p className="mt-1 text-[11px] text-slate-500">Source: {data.label_mapping?.sources[name]}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Cleaning statistics">
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="border-b border-slate-200 text-slate-600"><tr><th className="px-3 py-2 font-medium">Stage</th>
                  {Object.keys(data.cleaning.per_dataset).map((d) => <th key={d} className="px-3 py-2 font-medium">{d}</th>)}</tr></thead>
                <tbody>
                  {STAGES.map(([k, label]) => (
                    <tr key={k} className={`border-b border-slate-100 tabular-nums ${k === "final_rows" ? "font-semibold" : ""}`}>
                      <td className="px-3 py-2">{label}</td>
                      {Object.values(data.cleaning!.per_dataset).map((v, i) => <td key={i} className="px-3 py-2">{v[k]?.toLocaleString()}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-slate-600">Cleaning removes only URLs, HTML artefacts, control characters and extra whitespace. Telugu script, Romanized Telugu, emojis, ! and ? and negations are kept
              (occurrences kept: {Object.entries(data.cleaning.negation_words_preserved).map(([w, n]) => `${w} ${n.toLocaleString()}`).join(", ")}).</p>
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Conflicting labels and leakage">
              <p className="text-sm text-slate-700">{data.cleaning.conflicting_labels.texts_with_more_than_one_label} texts carry more than one label
                ({data.cleaning.conflicting_labels.rows_in_those_groups} rows; {Object.entries(data.cleaning.conflicting_labels.by_dataset).map(([k, v]) => `${k} ${v}`).join(", ")}).</p>
              <p className="mt-2 text-sm text-slate-700"><span className="font-medium">Decision:</span> {data.cleaning.conflicting_labels.decision}</p>
              <p className={`mt-3 text-sm font-medium ${data.leakage_checks?.passed ? "text-emerald-700" : "text-rose-700"}`}>
                {data.leakage_checks?.passed ? "✓ No cleaned text appears in more than one split." : "✗ Leakage detected"}
              </p>
              <ul className="mt-1 text-xs text-slate-600">
                {Object.entries(data.leakage_checks?.cleaned_text ?? {}).map(([k, v]) => <li key={k}>{k.replace("&", " ∩ ")}: {v}</li>)}
              </ul>
            </Card>
            <Card title="Split">
              <p className="text-sm text-slate-700">{data.cleaning.split.method}</p>
              <dl className="mt-3 grid grid-cols-3 gap-2 text-center">
                {["train", "validation", "test"].map((s) => (
                  <div key={s} className="rounded-md bg-slate-50 p-2">
                    <dt className="text-xs capitalize text-slate-500">{s}</dt>
                    <dd className="font-semibold tabular-nums">{data.cleaning!.split.sizes[s].toLocaleString()}</dd>
                    <dd className="text-xs text-slate-500">{(data.cleaning!.split.fractions[s] * 100).toFixed(1)}%</dd>
                  </div>
                ))}
              </dl>
            </Card>
          </div>

          <Card title="Class distribution by split and language">
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm tabular-nums">
                <thead className="border-b border-slate-200 text-slate-600">
                  <tr><th className="px-3 py-2 font-medium">Split</th><th className="px-3 py-2 font-medium">Language</th>
                    {["Negative", "Neutral", "Positive"].map((c) => <th key={c} className="px-3 py-2 font-medium">{c}</th>)}</tr>
                </thead>
                <tbody>
                  {["train", "validation", "test"].flatMap((s) => ["telugu", "english", "code_mixed"].map((l) => (
                    <tr key={s + l} className="border-b border-slate-100">
                      <td className="px-3 py-1.5 capitalize">{s}</td><td className="px-3 py-1.5">{LANG_LABEL[l]}</td>
                      {["Negative", "Neutral", "Positive"].map((c) => (
                        <td key={c} className="px-3 py-1.5">{data.class_distributions.find((d) => d.split === s && d.language_type === l && d.label_name === c)?.rows.toLocaleString() ?? 0}</td>
                      ))}
                    </tr>
                  )))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </>
  );
}
