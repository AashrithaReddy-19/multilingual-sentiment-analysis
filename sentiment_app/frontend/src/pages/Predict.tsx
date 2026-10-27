import { useState } from "react";
import type { ModelInfo, PredictResponse } from "../api";
import { api, LANG_LABEL } from "../api";
import { useApi } from "../hooks";
import { Card, ErrorState, Field, ModelScopeBanner, PageHeader, ProbabilityBars, SENTIMENT_STYLE, Spinner } from "../components/ui";

const EXAMPLES = [
  { label: "Telugu", text: "ఈ సినిమా చాలా బాగుంది 👍" },
  { label: "Romanized Telugu", text: "movie asalu bagaledu, time waste" },
  { label: "Code-mixed", text: "Service super ga undi bro, thank you!" },
  { label: "English", text: "The delivery was late again. Not happy." },
  { label: "Neutral", text: "The meeting is scheduled for 3 pm tomorrow." },
  { label: "Negation", text: "This is not bad at all" },
];
const MAX = 2000;

export default function Predict() {
  const models = useApi<ModelInfo[]>("/api/models");
  const [text, setText] = useState("");
  const [modelId, setModelId] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const available = (models.data ?? []).filter((m) => m.available);       // loadable for inference
  const selected = available.find((m) => m.id === modelId) ?? available.find((m) => m.is_default);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return setError("Please enter some text.");
    setBusy(true);
    setError(null);
    try {
      setResult(await api.predict(text, modelId || undefined));
    } catch (err) {
      setError((err as Error).message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader title="Single text prediction"
        subtitle="Type Telugu, English, Romanized Telugu or code-mixed text; emojis are kept. The model that produced the answer is always shown." />
      {models.error && <ErrorState message={models.error} onRetry={models.reload} />}
      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <form onSubmit={submit} className="space-y-4" noValidate>
            <Field label="Model" htmlFor="model" hint="Default: the Production Ensemble.">
              <select id="model" value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={models.loading}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm">
                <option value="">Default{selected && !modelId ? ` — ${selected.display_name ?? selected.name}` : ""}</option>
                {available.map((m) => <option key={m.id} value={m.id}>{m.display_name ?? m.name}</option>)}
              </select>
            </Field>
            {selected && <ModelScopeBanner model={selected} />}
            <Field label="Text" htmlFor="text" hint={`${text.length}/${MAX} characters`}>
              <textarea id="text" value={text} maxLength={MAX} rows={5} onChange={(e) => setText(e.target.value)}
                placeholder="ఉదా: ఈ సినిమా చాలా బాగుంది 👍" lang="te"
                className="te w-full rounded-lg border border-slate-300 px-3 py-2 text-base" aria-describedby="examples" />
            </Field>
            <div id="examples" className="flex flex-wrap gap-2">
              <span className="text-xs text-slate-500">Try:</span>
              {EXAMPLES.map((ex) => (
                <button type="button" key={ex.label} onClick={() => setText(ex.text)}
                  className="rounded-full border border-slate-300 px-2.5 py-1 text-xs text-slate-700 hover:bg-slate-100">{ex.label}</button>
              ))}
            </div>
            <div className="flex gap-3">
              <button type="submit" disabled={busy || !text.trim()}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50">
                {busy ? "Predicting…" : "Predict sentiment"}
              </button>
              <button type="button" onClick={() => { setText(""); setResult(null); setError(null); }}
                className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50">Clear</button>
            </div>
          </form>
        </Card>
        <Card title="Result" className="lg:col-span-2">
          <div aria-live="polite">
            {busy && <Spinner label="Running the model… the first prediction can take a few seconds while it loads." />}
            {!busy && error && <ErrorState message={error} />}
            {!busy && !error && !result && <p className="text-sm text-slate-500">The prediction will appear here.</p>}
            {!busy && result && (
              <div className="space-y-5">
                <div className="flex items-center gap-3">
                  <span className="text-4xl" aria-hidden>{SENTIMENT_STYLE[result.label].emoji}</span>
                  <div>
                    <p className={`text-2xl font-semibold ${SENTIMENT_STYLE[result.label].text}`}>{result.label}</p>
                    <p className="text-sm text-slate-600">Confidence {(result.confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>
                {result.low_confidence && (
                  <div role="alert" className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                    <p className="font-semibold">Low-confidence prediction</p>
                    <p className="text-xs">Confidence is below {((result.low_confidence_threshold ?? 0.5) * 100).toFixed(0)}% — treat this label with caution.
                      {result.model.id === "tfidf_plus_multilingual" && " On the multilingual test set, Model A+ was right only 46% of the time in this confidence range."}</p>
                  </div>
                )}
                <ProbabilityBars probabilities={result.probabilities} />
                <div className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
                  <p><span className="font-medium text-slate-800">Generated by:</span> {result.model.display_name ?? result.model.name}</p>
                  <p className={result.model.scope === "telugu_only" ? "font-medium text-amber-800" : ""}>{result.model.scope_label}</p>
                  <p className="te mt-1 break-words"><span className="font-medium text-slate-800">Cleaned input:</span> {result.cleaned_text}</p>
                  {result.language && (
                    <p className="mt-1"><span className="font-medium text-slate-800">Detected language:</span> {LANG_LABEL[result.language.detected] ?? result.language.detected}
                      <span className="block text-[11px] text-slate-500">{result.language.method}</span></p>
                  )}
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>
    </>
  );
}
