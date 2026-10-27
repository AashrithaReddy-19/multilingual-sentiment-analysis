"""Data-centric CPU study: TRAINING-only audit, quality scores, cleaned/weighted/augmented training variants screened
with the SAME A+ pipeline (word (1,2) 20k + char_wb (3,5) TF-IDF + balanced LR) and scored on VALIDATION only.
validation.csv / test.csv are never modified; test.csv is never read.

    python -m pipeline.data_centric audit oof variants curves report

Outputs (experiments/data_centric/): training_quality_audit_v2.csv (row-level flags, contains text -> local only),
training_quality_summary_v2.json, conflict_groups_v2.csv (local only), variants.csv, learning_curve.csv,
data_improvement_report.md, train_clean_v2.csv (local only, the best cleaning variant).
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from sentiment_core.experiment_io import full_metrics  # noqa: E402

OUT = paths.MULTI / "experiments" / "data_centric"
LANGS = ("english", "telugu", "code_mixed")
N = ("Negative", "Neutral", "Positive")
URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️‍]")
POS_W = re.compile(r"\b(good|great|love|loved|best|happy|awesome|amazing|excellent|super|nice|bagundi|baagundi|"
                   r"chala bagundi|keka|mass)\b", re.I)
NEG_W = re.compile(r"\b(bad|worst|hate|hated|sad|poor|waste|terrible|awful|horrible|disgusting|ledu|chetha|"
                   r"worst ga)\b", re.I)


def load(split):
    return pd.read_csv(paths.MULTI / "data" / f"{split}.csv", encoding="utf-8-sig", low_memory=False)


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t)).strip()


def loose(t: str) -> str:
    """near-duplicate key: also drop mentions, hashtags, URLs, digits and repeated letters"""
    t = URL.sub(" ", t)
    t = re.sub(r"[@#]\w+", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", norm(t))
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------------ PHASE 1-2: audit
def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load("train")
    t = tr["cleaned_text"].astype(str)
    raw = tr["raw_text"].astype(str)
    df = pd.DataFrame({"row_id": tr["row_id"], "label": tr["label"], "language_type": tr["language_type"],
                       "source_dataset": tr["source_dataset"], "cleaned_text": t})
    df["norm"] = t.map(norm)
    df["loose"] = t.map(loose)
    df["exact_dup"] = t.duplicated(keep=False)
    df["norm_dup"] = df["norm"].duplicated(keep=False)
    n_lab = df.groupby("norm")["label"].transform("nunique")
    df["conflict_group"] = n_lab > 1
    df["near_dup_conflict"] = (df.groupby("loose")["label"].transform("nunique") > 1) & ~df["conflict_group"] \
        & (df["loose"].str.len() > 0)
    df["cross_language_dup"] = df.groupby("norm")["language_type"].transform("nunique") > 1
    nw = t.str.split().str.len()
    stripped = raw.map(lambda s: re.sub(r"\s+", "", URL.sub("", re.sub(r"[@#]\w+", "", s))))
    df["empty_after_cleaning"] = t.str.strip().eq("")
    df["url_only"] = raw.map(lambda s: URL.sub("", s).strip() == "") & raw.str.contains(URL)
    df["mention_or_hashtag_only"] = stripped.eq("") & raw.str.contains(r"[@#]\w", regex=True)
    df["punct_only"] = t.map(lambda s: bool(s.strip()) and all(not c.isalnum() for c in s if not c.isspace())
                             and not EMOJI.search(s))
    df["emoji_only"] = t.map(lambda s: bool(EMOJI.search(s)) and not re.search(r"\w", EMOJI.sub("", s)))
    df["very_short_<=2_words"] = nw <= 2
    df["encoding_corruption"] = t.str.contains("Ã|â€|�|Â", regex=True)
    df["neutral_with_strong_pos"] = (df["label"] == 1) & t.str.contains(POS_W) & ~t.str.contains(NEG_W)
    df["neutral_with_strong_neg"] = (df["label"] == 1) & t.str.contains(NEG_W) & ~t.str.contains(POS_W)
    df["neutral_mixed_cues"] = (df["label"] == 1) & t.str.contains(POS_W) & t.str.contains(NEG_W)
    df["negative_with_pos_only_cues"] = (df["label"] == 0) & t.str.contains(POS_W) & ~t.str.contains(NEG_W)
    df["positive_with_neg_only_cues"] = (df["label"] == 2) & t.str.contains(NEG_W) & ~t.str.contains(POS_W)
    flags = [c for c in df.columns if df[c].dtype == bool]
    df.to_csv(OUT / "training_quality_audit_v2.csv", index=False, encoding="utf-8-sig")
    # conflict groups (deterministic description; no label is chosen)
    g = df[df["conflict_group"]].groupby("norm")
    cg = pd.DataFrame({"text": g["cleaned_text"].first(), "labels": g["label"].agg(lambda s: dict(s.value_counts())),
                       "rows": g.size(), "languages": g["language_type"].agg(lambda s: sorted(set(s))),
                       "sources": g["source_dataset"].agg(lambda s: sorted(set(s)))})
    cg["policy_class"] = np.where(cg["sources"].map(len) > 1, "cross-source disagreement (likely annotation-scheme)",
                                  np.where(cg["rows"] == 2, "single disagreeing pair (ambiguous)",
                                           "repeated text with split labels (likely noise or ambiguity)"))
    cg.to_csv(OUT / "conflict_groups_v2.csv", encoding="utf-8-sig")
    summary = {"train_rows": len(df), **{f: int(df[f].sum()) for f in flags},
               "conflict_groups": int(df.loc[df["conflict_group"], "norm"].nunique()),
               "conflict_policy_classes": cg["policy_class"].value_counts().to_dict(),
               "flagged_by_language": {f: df[df[f]]["language_type"].value_counts().to_dict() for f in
                                       ("conflict_group", "near_dup_conflict", "very_short_<=2_words",
                                        "neutral_with_strong_pos", "neutral_with_strong_neg")}}
    (OUT / "training_quality_summary_v2.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=1))
    print("conflict policy classes:", summary["conflict_policy_classes"])


# ------------------------------------------------------------------ PHASE 7: OOF quality score
def a_plus_pipe(C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from pipeline.cpu_round3 import a_plus_feats
    return Pipeline([("f", a_plus_feats()), ("c", LogisticRegression(C=C, class_weight="balanced", max_iter=1000,
                                                                       random_state=42))])


def oof():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC
    tr = load("train")
    x, y = tr["cleaned_text"].astype(str).to_numpy(), tr["label"].to_numpy(int)
    pa = np.zeros((len(y), 3)); pc = np.zeros(len(y), int)
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(x, y)):
        t0 = time.time()
        pa[b] = a_plus_pipe().fit(x[a], y[a]).predict_proba(x[b])
        cf = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=300000, sublinear_tf=True)
        Xa = cf.fit_transform(x[a])
        pc[b] = LinearSVC(C=0.1, class_weight="balanced").fit(Xa, y[a]).predict(cf.transform(x[b]))
        print(f"  oof fold {k + 1}/5 ({time.time() - t0:.0f}s)", flush=True)
    p_true = pa[np.arange(len(y)), y]
    correct = (pa.argmax(1) == y).astype(int) + (pc == y).astype(int)
    s = np.sort(pa, 1); margin = s[:, -1] - s[:, -2]
    audit_df = pd.read_csv(OUT / "training_quality_audit_v2.csv", encoding="utf-8-sig", low_memory=False)
    cat = np.where((correct == 2) & (p_true >= 0.6), "clean_consensus",
          np.where((correct == 0) & (pa.max(1) >= 0.7), "potentially_noisy",
          np.where(correct == 0, "hard_legitimate", "normal")))
    cat = np.where(audit_df["conflict_group"].to_numpy(), "potentially_noisy", cat)
    q = pd.DataFrame({"row_id": tr["row_id"], "label": y, "language_type": tr["language_type"],
                      "oof_models_correct": correct, "aplus_p_true": p_true, "aplus_margin": margin,
                      "conflict_group": audit_df["conflict_group"], "category": cat})
    q.to_csv(OUT / "train_quality_scores.csv", index=False)
    print(q["category"].value_counts().to_dict())
    noisy = q["category"] == "potentially_noisy"
    est = {"overall": float(noisy.mean()),
           **{f"lang_{l}": float(noisy[q["language_type"] == l].mean()) for l in LANGS},
           **{f"class_{N[c]}": float(noisy[q["label"] == c].mean()) for c in range(3)}}
    s = json.loads((OUT / "training_quality_summary_v2.json").read_text())
    s["oof_categories"] = q["category"].value_counts().to_dict()
    s["potential_label_noise_or_ambiguity_fraction"] = est
    (OUT / "training_quality_summary_v2.json").write_text(json.dumps(s, indent=2, default=int), encoding="utf-8")
    print("potential noise/ambiguity fractions:", {k: round(v, 4) for k, v in est.items()})


# ------------------------------------------------------------------ PHASES 3-4, 8-12, 14, 16: variants
def neutral_aug(texts):
    """semantics-preserving format variants: trailing punctuation removed/added, lower case for Latin text"""
    out = []
    for s in texts:
        v = s.rstrip(".!?") if s.rstrip(".!?") != s else s + "."
        out.append(v.lower() if all(ord(c) < 0x0C00 for c in v) else v)
    return out


def telugu_norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).replace("‌", "").replace("​", "")
    return re.sub(r"\s+([.,!?])", r"\1", s)


def variants():
    tr, va = load("train"), load("validation")
    q = pd.read_csv(OUT / "train_quality_scores.csv")
    au = pd.read_csv(OUT / "training_quality_audit_v2.csv", encoding="utf-8-sig", low_memory=False)
    x, y, lang = tr["cleaned_text"].astype(str).to_numpy(), tr["label"].to_numpy(int), tr["language_type"].to_numpy()
    xv, yv = va["cleaned_text"].astype(str).to_numpy(), va["label"].to_numpy(int)
    first_copy = ~pd.Series(x).duplicated(keep="first").to_numpy() | ~au["exact_dup"].to_numpy()
    same_label_dup_extra = au["exact_dup"].to_numpy() & ~first_copy & ~au["conflict_group"].to_numpy()
    keepB = ~same_label_dup_extra
    conflict = au["conflict_group"].to_numpy()
    cat = q["category"].to_numpy()
    ones = np.ones(len(y))
    langw = {l: 1 / np.sqrt((lang == l).mean()) for l in LANGS}
    lw = np.array([langw[l] for l in lang]); lw = lw / lw.mean()

    def fit_eval(name, mask=None, w=None, xs=None, ys=None, xv_=None, note=""):
        t0 = time.time()
        m = mask if mask is not None else np.ones(len(y), bool)
        xx = xs if xs is not None else x[m]
        yy = ys if ys is not None else y[m]
        ww = w if w is not None else None
        pipe = a_plus_pipe()
        pipe.fit(xx, yy, **({"c__sample_weight": ww} if ww is not None else {}))
        mm = full_metrics(yv, pipe.predict(xv_ if xv_ is not None else xv), va["language_type"])
        r = {"variant": name, "train_rows": len(yy), "val_accuracy": mm["accuracy"], "val_macro_f1": mm["macro_f1"],
             **{f"{c}_f1": mm["per_class"][c]["f1"] for c in N},
             **{f"{l}_macro_f1": mm["per_language"][l]["macro_f1"] for l in LANGS}, "note": note}
        print(f"{name:<34} rows {len(yy):6d} | VAL acc {r['val_accuracy']:.4f} F1 {r['val_macro_f1']:.4f} | N/U/P "
              f"{r['Negative_f1']:.3f}/{r['Neutral_f1']:.3f}/{r['Positive_f1']:.3f} | EN/TE/CM "
              f"{r['english_macro_f1']:.3f}/{r['telugu_macro_f1']:.3f}/{r['code_mixed_macro_f1']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        rows.append(r)
        pd.DataFrame(rows).to_csv(OUT / "variants.csv", index=False)
        return r

    rows = []
    fit_eval("A original")
    fit_eval("B same-label duplicates collapsed", mask=keepB)
    fit_eval("C B + conflict groups removed", mask=keepB & ~conflict)
    fit_eval("D B + conflict groups weight 0.5", mask=keepB, w=np.where(conflict, 0.5, 1.0)[keepB])
    for wn, wh in ((0.5, 1.1), (0.25, 1.0), (0.75, 1.2)):
        w = np.where(cat == "potentially_noisy", wn, np.where(cat == "hard_legitimate", wh, 1.0))
        fit_eval(f"E quality weights noisy {wn} / hard {wh}", w=w)
    fit_eval("F potentially-noisy rows removed", mask=cat != "potentially_noisy")
    for nw in (1.05, 1.1, 1.2):
        fit_eval(f"G Neutral weight x{nw}", w=np.where(y == 1, nw, 1.0))
    fit_eval("H sqrt language balancing", w=lw)
    for tw in (1.2, 1.3):
        fit_eval(f"H Telugu weight x{tw}", w=np.where(lang == "telugu", tw, 1.0))
    for en_neu, te_neu, te_pol in ((1.1, 1.1, 1.0), (1.0, 1.2, 1.1), (1.1, 1.2, 1.2)):
        w = ones.copy()
        w[(lang == "english") & (y == 1)] *= en_neu
        w[(lang == "telugu") & (y == 1)] *= te_neu
        w[(lang == "telugu") & (y != 1)] *= te_pol
        fit_eval(f"I matrix EN-neu {en_neu} TE-neu {te_neu} TE-pol {te_pol}", w=w)
    neu = y == 1
    aug_x = np.concatenate([x, np.array(neutral_aug(x[neu]))])
    aug_y = np.concatenate([y, y[neu]])
    fit_eval("J Neutral format augmentation", xs=aug_x, ys=aug_y, note=f"+{int(neu.sum())} variants")
    tn = lambda arr: np.array([telugu_norm(s) for s in arr])
    fit_eval("K Telugu Unicode normalisation", xs=tn(x), ys=y, xv_=tn(xv), note="same normalisation at inference")
    best = max(rows, key=lambda r: r["val_macro_f1"])
    print("BEST VARIANT:", best["variant"], round(best["val_macro_f1"], 4),
          "gain vs A", round(best["val_macro_f1"] - rows[0]["val_macro_f1"], 4))
    clean = tr[keepB & ~conflict] if best["variant"].startswith("C") else tr[keepB]
    clean.to_csv(OUT / "train_clean_v2.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ PHASES 21-22: learning curves
def curves():
    from sklearn.model_selection import train_test_split
    tr, va = load("train"), load("validation")
    x, y = tr["cleaned_text"].astype(str).to_numpy(), tr["label"].to_numpy(int)
    strata = tr["language_type"].astype(str) + "_" + tr["label"].astype(str)
    rows = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        idx = np.arange(len(y)) if frac == 1.0 else train_test_split(np.arange(len(y)), train_size=frac,
                                                                      stratify=strata, random_state=42)[0]
        mm = full_metrics(va["label"].to_numpy(int), a_plus_pipe().fit(x[idx], y[idx]).predict(
            va["cleaned_text"].astype(str)), va["language_type"])
        rows.append({"train_fraction": frac, "rows": len(idx), "val_macro_f1": mm["macro_f1"],
                     "val_accuracy": mm["accuracy"],
                     **{f"{l}_macro_f1": mm["per_language"][l]["macro_f1"] for l in LANGS}})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "learning_curve.csv", index=False)


# ------------------------------------------------------------------ PHASE 24: report
def report():
    s = json.loads((OUT / "training_quality_summary_v2.json").read_text())
    v = pd.read_csv(OUT / "variants.csv"); lc = pd.read_csv(OUT / "learning_curve.csv")
    base = v.iloc[0]
    v["gain_vs_original"] = v["val_macro_f1"] - base["val_macro_f1"]
    L = ["# Data-centric improvement study (training data only)", "",
         "validation.csv and test.csv were not modified; test.csv was not read. Every variant trains the SAME A+ "
         "pipeline (word (1,2) 20k + char_wb (3,5) TF-IDF, balanced logistic regression) and is scored on validation.",
         "", "## Training-data audit", ""]
    for k in ("train_rows", "exact_dup", "norm_dup", "conflict_groups", "conflict_group", "near_dup_conflict",
              "cross_language_dup", "empty_after_cleaning", "url_only", "mention_or_hashtag_only", "punct_only",
              "emoji_only", "very_short_<=2_words", "encoding_corruption", "neutral_with_strong_pos",
              "neutral_with_strong_neg", "neutral_mixed_cues", "negative_with_pos_only_cues",
              "positive_with_neg_only_cues"):
        L.append(f"- {k}: {s.get(k)}")
    L += ["", f"Conflict groups by deterministic policy class: {s.get('conflict_policy_classes')}", "",
          "## Train-CV quality categories (5-fold out-of-fold A+ probabilities + char LinearSVC)", "",
          f"{s.get('oof_categories')}", "",
          "Potential label noise / ambiguity (rows both OOF models get wrong with A+ confidence >= 0.7, or in a "
          "conflict group). Disagreement is NOT proof of a wrong label:", ""]
    L += [f"- {k}: {val:.2%}" for k, val in s.get("potential_label_noise_or_ambiguity_fraction", {}).items()]
    L += ["", "## Training variants (validation)", "",
          v[["variant", "train_rows", "val_accuracy", "val_macro_f1", "gain_vs_original", "Neutral_f1",
             "english_macro_f1", "telugu_macro_f1", "code_mixed_macro_f1"]].round(4).to_markdown(index=False), "",
          "## Learning curve (A+, stratified by language x label)", "", lc.round(4).to_markdown(index=False), ""]
    (OUT / "data_improvement_report.md").write_text("\n".join(L), encoding="utf-8")
    print("report written")


def main(argv=None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for step in (argv if argv is not None else sys.argv[1:]):
        globals()[step]()


if __name__ == "__main__":
    main()
