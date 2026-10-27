export type Scope = "multilingual" | "telugu_only";
export type ClassName = "Negative" | "Neutral" | "Positive";
export const CLASSES: ClassName[] = ["Negative", "Neutral", "Positive"];

export type DisplayCategory = "production" | "evaluated" | "baseline" | "pending" | "skipped";

export interface ModelInfo {
  id: string;
  name: string;
  kind: "transformer" | "tfidf" | "ensemble";
  scope: Scope;
  scope_label: string;
  available: boolean;
  loaded: boolean;
  is_default: boolean;
  path: string;
  context_aware?: boolean;
  display_name?: string;
  test_accuracy?: number | null;
  test_macro_f1?: number | null;
  inference_available?: boolean;
}

export interface PredictResponse {
  label: ClassName;
  label_id: number;
  confidence: number;
  probabilities: { label: ClassName; probability: number }[];
  cleaned_text: string;
  model: ModelInfo;
  language?: { detected: "telugu" | "english" | "code_mixed" | "unknown"; method: string };
  low_confidence?: boolean;
  low_confidence_threshold?: number;
  context_aware?: boolean;
  note?: string;
}

export interface LangScore {
  status: string;
  n: number;
  accuracy: number | null;
  macro_f1: number | null;
  weighted_f1: number | null;
}

export interface ModelResult {
  id: string;
  name: string;
  input: string;
  status: "trained" | "pending" | "skipped";
  note: string | null;
  explanation?: string;
  accuracy?: number;
  macro_f1?: number;
  weighted_f1?: number;
  macro_precision?: number | null;
  macro_recall?: number | null;
  weighted_precision?: number | null;
  weighted_recall?: number | null;
  best_validation_macro_f1?: number | null;
  n_test?: number;
  per_class?: Record<ClassName, { precision: number; recall: number; f1: number; support: number }>;
  per_language?: Record<string, LangScore>;
  confusion_matrix?: number[][] | null;
  confusion_png?: string;
  confidence?: { mean: number; mean_when_correct: number | null; mean_when_wrong: number | null };
  display_name?: string;
  description?: string;
  category?: DisplayCategory;
  is_production?: boolean;
  dashboard_visible?: boolean;
}

/** Dashboard rule: every TRAINED model with real test metrics is shown, whatever its score;
 *  pending, skipped and metric-less entries are never shown. */
export const isDashboardVisible = (m: ModelResult) =>
  m.status === "trained" && m.accuracy != null && m.macro_f1 != null;
export const modelName = (m: { display_name?: string; name: string }) => m.display_name ?? m.name;

export interface RunResult {
  run: "multilingual" | "telugu_only";
  title: string;
  models: ModelResult[];
}

export interface Dashboard {
  title: string;
  datasets: { name: string; source: string; url?: string | null; licence_note: string; checksums_match: boolean; final_rows: number }[];
  runs: RunResult[];
  serving_model: ModelInfo | null;
}

export interface ErrorRow {
  actual_label: ClassName;
  predicted_label: ClassName;
  confidence: number;
  raw_text: string;
  cleaned_text: string;
  language_type: string;
  likely_error_reason: string;
  source_dataset?: string;
}

export interface ErrorsPage {
  available: boolean;
  total: number;
  total_mistakes?: number;
  rows: ErrorRow[];
  heuristic_note: string;
}

export interface BatchInspect {
  columns: string[];
  n_rows: number;
  preview: Record<string, string>[];
  suggested_column: string | null;
  max_rows: number;
}

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") msg = body.detail;
      else if (Array.isArray(body.detail)) msg = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new ApiError(msg, res.status);
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T,>(path: string) => fetch(path).then((r) => handle<T>(r)),
  predict: (text: string, model_id?: string) =>
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model_id: model_id || null }),
    }).then((r) => handle<PredictResponse>(r)),
  inspect: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/batch/inspect", { method: "POST", body: fd }).then((r) => handle<BatchInspect>(r));
  },
  batchPredict: async (file: File, textColumn: string, model_id?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("text_column", textColumn);
    if (model_id) fd.append("model_id", model_id);
    const res = await fetch("/api/batch/predict", { method: "POST", body: fd });
    if (!res.ok) await handle(res);
    return { blob: await res.blob(), modelId: res.headers.get("X-Model-Id"), scope: res.headers.get("X-Model-Scope") };
  },
};

export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`;
export const num = (v: number | null | undefined, digits = 4) =>
  v === null || v === undefined ? "—" : v.toFixed(digits);
export const LANG_LABEL: Record<string, string> = { telugu: "Telugu", english: "English", code_mixed: "Code-mixed", unknown: "Unknown (no letters)" };
