import { useApi } from "../hooks";
import { Card, ErrorState, PageHeader, Spinner } from "../components/ui";

interface ContextStatus {
  model_c_status: string;
  explanation: string;
  required_file: string;
  required_columns: string[];
  column_rules: Record<string, string>;
  context_format: string;
  conversations_available: boolean;
}

export default function Context() {
  const { data, error, loading, reload } = useApi<ContextStatus>("/api/context-status");
  return (
    <>
      <PageHeader title="Context status" subtitle="What the context-aware model (Model C) needs, and why it is currently skipped." />
      {loading && <Spinner />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {data && (
        <div className="space-y-6">
          <div className="rounded-xl border border-slate-300 bg-slate-100 p-5" role="status">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Model C · context-aware XLM-RoBERTa</p>
            <p className="mt-1 text-lg font-semibold text-slate-900">{data.model_c_status}</p>
            <p className="mt-2 text-sm text-slate-700">{data.explanation}</p>
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card title={`Required file: ${data.required_file}`}>
              <pre className="overflow-x-auto rounded-lg bg-slate-900 p-3 text-sm text-slate-100"><code>{data.required_columns.join(",")}</code></pre>
              <dl className="mt-4 space-y-2 text-sm">
                {Object.entries(data.column_rules).map(([k, v]) => (
                  <div key={k} className="grid grid-cols-3 gap-2">
                    <dt className="font-mono text-xs text-indigo-700">{k}</dt>
                    <dd className="col-span-2 text-slate-700">{v}</dd>
                  </div>
                ))}
              </dl>
            </Card>
            <Card title="How context will be built">
              <p className="text-sm text-slate-700">Each message is combined with the two earlier messages of the same conversation, after sorting by message_order. Later messages are never used.</p>
              <pre className="mt-3 overflow-x-auto whitespace-pre-wrap rounded-lg bg-slate-50 p-3 text-xs text-slate-800">{data.context_format}</pre>
              <ul className="mt-3 space-y-1 text-sm text-slate-700">
                <li>• First message: <code className="text-xs">[CURRENT] …</code></li>
                <li>• Second message: <code className="text-xs">[PREVIOUS_1] … [CURRENT] …</code></li>
                <li>• Later messages: both earlier messages plus the current one</li>
              </ul>
              <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                No conversations are generated or simulated. Model C will be trained only after a real, annotated conversations.csv is provided.
              </p>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
