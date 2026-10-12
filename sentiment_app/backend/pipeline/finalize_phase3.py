"""Steps 7, 14, 15, 17, 18, 20: import Colab Transformer results, ensemble (weights tuned on VALIDATION), final test
evaluation of the validation-selected models, calibration, McNemar + bootstrap, final comparison table.

    python -m pipeline.finalize_phase3                  # uses whatever results are present (classical-only is fine)
    python -m pipeline.finalize_phase3 --import PATH    # first import phase3_results_*.zip (file or folder)

Rules enforced here:
- Transformer runs are imported only if they are real (not smoke) runs on exactly this project's split (SHA-256).
- Ensemble weights are chosen on validation only; the ensemble is scored on test once.
- A/A+ originals are read, never rewritten; derived files go to experiments/final/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import binomtest, chi2  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from pipeline.exp_common import (EXP, full_metrics, labels_from_names, leaderboard_row_from, read_split,  # noqa: E402
                                 save_eval_files, stage_banner, upsert_leaderboard)
from sentiment_core.evaluation import evaluate_and_save  # noqa: E402
from sentiment_core.labels import CLASS_NAMES, MODEL_C_SKIP  # noqa: E402

TRANS = EXP / "transformers"
FINAL = EXP / "final"
PROB_COLS = ["prob_Negative", "prob_Neutral", "prob_Positive"]
CONF_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
CONF_LABELS = ["0.00-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
FAMILY_LABEL = {"xlmr": "XLM-RoBERTa", "muril": "MuRIL", "indicbert": "IndicBERTv2", "mdeberta": "mDeBERTa-v3"}
FAMILY_OF_MODEL = {"xlm-roberta-base": "xlmr", "google/muril-base-cased": "muril",
                   "ai4bharat/IndicBERTv2-MLM-only": "indicbert", "microsoft/mdeberta-v3-base": "mdeberta"}
LANG_NAME = {"english": "English", "telugu": "Telugu", "code_mixed": "Code-mixed"}
MODEL_C_MESSAGE = ("True context-aware sentiment training could not be performed because the available datasets do "
                   "not contain reliable conversation structure.")


# ------------------------------------------------------------------ import
def split_hashes() -> dict:
    import hashlib
    return {s: hashlib.sha256((paths.DATA / f"{s}.csv").read_bytes()).hexdigest() for s in ("train", "validation", "test")}


def find_zip(arg: str | None) -> Path:
    if arg and Path(arg).is_file():
        return Path(arg)
    for folder in ([Path(arg)] if arg else []) + [paths.APP_ROOT / "artifacts" / "colab_results", Path.home() / "Downloads"]:
        if folder.is_dir():
            found = sorted(folder.glob("phase3_results_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
            if found:
                return found[0]
    raise SystemExit("No phase3_results_*.zip found (looked in the given path, artifacts/colab_results, Downloads).")


def import_results(arg: str | None) -> None:
    src = find_zip(arg)
    local = split_hashes()
    tmp = EXP / "_import_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(src) as z:
        if z.testzip() is not None:
            raise SystemExit(f"{src} is corrupted")
        z.extractall(tmp)
    TRANS.mkdir(parents=True, exist_ok=True)
    n = 0
    for cfg_file in sorted(tmp.glob("*/config.json")):
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        run = cfg_file.parent
        if cfg.get("smoke_test"):
            print(f"  skip {run.name}: smoke test")
            continue
        if cfg.get("split_sha256") != local:
            raise SystemExit(f"{run.name} was trained on a different split; nothing imported.")
        dest = TRANS / run.name
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(run, dest)
        n += 1
    sel = tmp / "_selected_for_test.json"
    if sel.exists() and n:
        shutil.copy(sel, TRANS / "_selected_for_test.json")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Imported {n} Transformer runs from {src.name} -> {TRANS}")


def transformer_runs() -> list[dict]:
    out = []
    for d in sorted(TRANS.glob("*/validation_metrics.json")) if TRANS.exists() else []:
        run = d.parent
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        out.append({"id": run.name, "cfg": cfg, "val": json.loads(d.read_text(encoding="utf-8")),
                    "test": json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
                    if (run / "test_metrics.json").exists() else None,
                    "family": FAMILY_OF_MODEL[cfg["model"]], "dir": run})
    return out


def register_transformers(runs: list[dict]) -> None:
    for r in runs:
        c = r["cfg"]
        hyper = {k: c[k] for k in ("learning_rate", "epochs_max", "best_epoch", "max_length", "loss", "focal_gamma",
                                   "batch_size", "grad_accum", "weight_decay", "warmup_ratio") if k in c}
        upsert_leaderboard(leaderboard_row_from(
            r["id"], c["model"], "fine-tuned Transformer" + (" + language tokens" if c["lang_tokens"] else ""),
            c["input"], hyper, r["val"], test=r["test"],
            notes=f"Colab {c.get('device', '')}, {c.get('train_seconds', 0) / 60:.0f} min"
                  + ("; SELECTED for test (best validation macro F1 in family)" if r["test"] else "")))


# ------------------------------------------------------------------ helpers
def probs_of(df: pd.DataFrame) -> np.ndarray:
    return df[PROB_COLS].to_numpy(dtype=float)


def check_rows(pred: pd.DataFrame, ref: pd.DataFrame, what: str) -> None:
    if "row_id" in pred:
        ok = np.array_equal(pred["row_id"].to_numpy(), ref["row_id"].to_numpy())
    else:
        ok = np.array_equal(pred["cleaned_text"].astype(str).to_numpy(), ref["cleaned_text"].astype(str).to_numpy())
    if not ok:
        raise SystemExit(f"{what}: prediction rows are not aligned with the split file.")


def fast_macro_f1(y, p) -> float:
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(3), where=(prec + rec) > 0)
    return float(f1.mean())


def calibration(y, probs) -> dict:
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    b = pd.cut(conf, CONF_BINS, labels=CONF_LABELS, right=False)
    t = pd.DataFrame({"bucket": b, "conf": conf, "correct": correct}).groupby("bucket", observed=False)
    buckets = [{"bucket": k, "n": int(len(g)), "accuracy": float(g["correct"].mean()) if len(g) else None,
                "mean_confidence": float(g["conf"].mean()) if len(g) else None} for k, g in t]
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, 9)
    ece = sum(abs(correct[idx == i].mean() - conf[idx == i].mean()) * (idx == i).mean() for i in range(10)
              if (idx == i).any())
    return {"buckets": buckets, "ece_10_bins": float(ece), "mean_confidence": float(conf.mean()),
            "accuracy": float(correct.mean())}


def mcnemar(y, pa, pb) -> dict:
    a_ok, b_ok = pa == y, pb == y
    b = int((a_ok & ~b_ok).sum())      # A right, B wrong
    c = int((~a_ok & b_ok).sum())      # A wrong, B right
    if b + c == 0:
        return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": 1.0, "test": "none (no disagreements)"}
    if b + c < 25:
        return {"a_right_b_wrong": b, "a_wrong_b_right": c, "test": "exact binomial",
                "p_value": float(binomtest(b, b + c, 0.5).pvalue)}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "test": "chi-square with continuity correction",
            "statistic": float(stat), "p_value": float(chi2.sf(stat, 1))}


def bootstrap(y, pa, pb, n=2000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    fa, fb, d = [], [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a, b = fast_macro_f1(y[i], pa[i]), fast_macro_f1(y[i], pb[i])
        fa.append(a); fb.append(b); d.append(b - a)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"resamples": n, "macro_f1_ci95_reference": q(fa), "macro_f1_ci95_model": q(fb),
            "macro_f1_diff_ci95": q(d), "macro_f1_diff_mean": float(np.mean(d)),
            "share_of_resamples_model_better": float(np.mean(np.array(d) > 0))}


def save_normalized_cm(y, pred, folder: Path, title: str) -> None:
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]).astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    pd.DataFrame(cmn, index=[f"true_{c}" for c in CLASS_NAMES], columns=[f"pred_{c}" for c in CLASS_NAMES]) \
        .to_csv(folder / "confusion_matrix_normalized.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(cmn, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{title}\nTest, row-normalized (recall)")
    fig.tight_layout(); fig.savefig(folder / "confusion_matrix_normalized.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="imp", nargs="?", const="", default=None)
    args = ap.parse_args(argv)
    if args.imp is not None:
        import_results(args.imp or None)

    val_df, test_df = read_split("validation"), read_split("test")
    yv, yt = val_df["label"].to_numpy(int), test_df["label"].to_numpy(int)
    print("Label counts (0=Negative, 1=Neutral, 2=Positive):")
    for name in ("train", "validation", "test"):
        print(f"  {name}: {read_split(name)['label'].value_counts().sort_index().to_dict()}")

    # ---- candidates: (name, input, val_probs, test_probs, best_val_f1, notes, experiment_id)
    cands = []
    a_test = pd.read_csv(paths.MULTI / "model_a" / "predictions_model_a.csv", encoding="utf-8-sig", low_memory=False)
    check_rows(a_test, test_df, "Model A test")
    a_val_f1 = json.loads((paths.MULTI / "model_a" / "metrics_model_a.json").read_text(encoding="utf-8"))["best_validation_macro_f1"]
    cands.append({"name": "Multilingual A", "input": "cleaned_text", "val": None, "test": probs_of(a_test),
                  "best_val": a_val_f1, "id": "model_a",
                  "notes": "Word TF-IDF (1,2) 20k + LogisticRegression balanced (original baseline)"})
    ap_test = pd.read_csv(paths.MULTI / "model_a_plus" / "predictions_model_a_plus.csv", encoding="utf-8-sig",
                          low_memory=False)
    check_rows(ap_test, test_df, "Model A+ test")
    ap_val = pd.read_csv(EXP / "s0_a_plus_reproduced" / "predictions.csv", encoding="utf-8-sig")
    check_rows(ap_val, val_df, "A+ validation")
    cands.append({"name": "Multilingual A+", "input": "cleaned_text", "val": probs_of(ap_val), "test": probs_of(ap_test),
                  "best_val": json.loads((paths.MULTI / "model_a_plus" / "metrics_model_a_plus.json")
                                         .read_text(encoding="utf-8"))["best_validation_macro_f1"],
                  "id": "model_a_plus", "notes": "Word (1,2) 20k + char_wb (3,5) TF-IDF + LogisticRegression balanced"})

    # best improved classical model = top classical validation run (S0-S6 search)
    lb = pd.read_csv(EXP / "experiment_leaderboard.csv", encoding="utf-8-sig")
    classical = lb[lb["experiment_id"].str.match(r"s[0-6]_")].sort_values(
        ["validation_macro_f1", "validation_accuracy"], ascending=False)
    best_cls = classical.iloc[0]
    classical_note = (f"Best of {len(classical)} validation-searched classical configs: {best_cls['experiment_id']} "
                      f"(val macro F1 {best_cls['validation_macro_f1']:.4f}).")
    same_as_a_plus = best_cls["experiment_id"] in ("s0_a_plus_reproduced", "s4_lr_bal_C1")
    print("\n" + classical_note + (" It IS the A+ configuration, so A+ remains the best classical model; no new "
                                   "classical test evaluation is needed." if same_as_a_plus else ""))
    if not same_as_a_plus:
        raise SystemExit("A new classical config won on validation: train it with pipeline.classical_final first.")

    runs = transformer_runs()
    if runs:
        register_transformers(runs)
    fam_best = {}
    for fam in FAMILY_LABEL:
        fam_runs = sorted([r for r in runs if r["family"] == fam], key=lambda r: rank_key(r["val"]), reverse=True)
        if not fam_runs:
            continue
        best = fam_runs[0]
        if best["test"] is None:
            raise SystemExit(f"{best['id']} is the best {fam} validation run but has no test predictions; re-run the "
                             "Colab 'Final test' cell.")
        vp = pd.read_csv(best["dir"] / "predictions.csv", encoding="utf-8-sig")
        tp = pd.read_csv(best["dir"] / "predictions_test.csv", encoding="utf-8-sig")
        check_rows(vp, val_df, f"{best['id']} validation"); check_rows(tp, test_df, f"{best['id']} test")
        c = best["cfg"]
        fam_best[fam] = {"input": c["input"], "val": probs_of(vp), "test": probs_of(tp), "best_val": best["val"]["macro_f1"],
                         "val_metrics": best["val"], "id": best["id"], "family": fam, "kind": "transformer",
                         "notes": transformer_note(c, len(fam_runs), fam)}
    if "xlmr" in fam_best:
        cands.append({**fam_best["xlmr"], "name": "Multilingual XLM-R Model B"})
    else:
        print("Multilingual XLM-R Model B: no imported runs yet (Not trained / pending GPU run).")
    alts = sorted([v for k, v in fam_best.items() if k != "xlmr"], key=lambda v: rank_key(v["val_metrics"]), reverse=True)
    for i, a in enumerate(alts):
        label = "Best alternative Transformer" if i == 0 else "Other alternative Transformer"
        cands.append({**a, "name": f"{label} ({FAMILY_LABEL[a['family']]})"})

    # ---- validation-only comparison of every family winner with A+ (Neutral boundary + English focus)
    vrows = []
    for c in [next(c for c in cands if c["id"] == "model_a_plus")] + [c for c in cands if c.get("kind") == "transformer"]:
        pv = c["val"].argmax(1)
        m = full_metrics(yv, pv, val_df["language_type"])
        en = m["per_language"]["english"]
        row = {"model": c["name"], "experiment_id": c["id"], "val_accuracy": m["accuracy"],
               "val_macro_precision": m["macro_precision"], "val_macro_recall": m["macro_recall"],
               "val_macro_f1": m["macro_f1"], "val_weighted_f1": m["weighted_f1"],
               **{f"val_{k}_{s}": m["per_class"][k][s] for k in CLASS_NAMES for s in ("precision", "recall", "f1")},
               "val_english_accuracy": en["accuracy"], "val_english_macro_precision": en["macro_precision"],
               "val_english_macro_recall": en["macro_recall"], "val_english_macro_f1": en["macro_f1"],
               "val_telugu_macro_f1": m["per_language"]["telugu"]["macro_f1"],
               "val_code_mixed_macro_f1": m["per_language"]["code_mixed"]["macro_f1"]}
        for a, b in (("Negative", "Neutral"), ("Neutral", "Negative"), ("Neutral", "Positive"), ("Positive", "Neutral")):
            row[f"val {a}->{b}"] = int(((yv == CLASS_NAMES.index(a)) & (pv == CLASS_NAMES.index(b))).sum())
        vrows.append(row)
    pd.DataFrame(vrows).to_csv(EXP / "validation_family_winners_vs_a_plus.csv", index=False, encoding="utf-8-sig")

    # ---- Step 20: ensemble (small rational grid, validation only; promoted only if it beats every member)
    ensemble_cfg = None
    xl = next((c for c in cands if c["id"] == fam_best.get("xlmr", {}).get("id")), None)
    alt = next((c for c in cands if alts and c["id"] == alts[0]["id"]), None)
    a_plus = next(c for c in cands if c["id"] == "model_a_plus")
    if xl is not None:
        members = [a_plus, xl]
        if alt is not None and alt["best_val"] >= xl["best_val"] - 0.02:      # "competitive" alternative
            members.append(alt)
        grid = ensemble_grid(len(members))
        scored = []
        for w in grid:
            pv = sum(wi * c["val"] for wi, c in zip(w, members))
            m = full_metrics(yv, pv.argmax(1))
            scored.append((rank_key(m), w, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_key, best_w, best_m = scored[0]
        best_single = max(fast_macro_f1(yv, c["val"].argmax(1)) for c in members)
        promoted = best_m["macro_f1"] > best_single
        names = [c["name"] for c in members]
        ens_dir = EXP / "ensemble_validation_tuned"
        ens_dir.mkdir(parents=True, exist_ok=True)
        pv = sum(wi * c["val"] for wi, c in zip(best_w, members))
        vm = save_eval_files(ens_dir, yv, pv, val_df, "validation", "Validation-tuned ensemble")
        ensemble_cfg = {"members": names, "member_ids": [c["id"] for c in members], "weights": list(map(float, best_w)),
                        "grid": [list(map(float, w)) for w in grid], "selected_on": "validation macro F1 (then recall, accuracy)",
                        "best_single_member_val_macro_f1": best_single, "val_macro_f1": best_m["macro_f1"],
                        "promoted": bool(promoted),
                        "top5_validation": [{"weights": list(map(float, w)), "val_macro_f1": m["macro_f1"]}
                                            for _, w, m in scored[:5]]}
        (ens_dir / "config.json").write_text(json.dumps(ensemble_cfg, indent=2), encoding="utf-8")
        print(f"\nEnsemble grid ({len(grid)} weightings, validation only): best {ensemble_cfg['weights']} for {names} "
              f"-> val macro F1 {best_m['macro_f1']:.4f} vs best single member {best_single:.4f}: "
              f"{'PROMOTED' if promoted else 'not promoted (no validation gain); not tested'}")
        if promoted:
            cands.append({"name": "Ensemble (validation-tuned weights)", "input": "cleaned_text (all members)",
                          "val": pv, "test": sum(wi * c["test"] for wi, c in zip(best_w, members)),
                          "best_val": vm["macro_f1"], "id": "ensemble_validation_tuned", "kind": "ensemble",
                          "notes": "weights " + ", ".join(f"{n} {w:.2f}" for n, w in zip(names, best_w))})
        else:
            (ens_dir / "config.json").unlink()        # nothing for the app to serve
            (ens_dir / "NOT_PROMOTED.json").write_text(json.dumps(ensemble_cfg, indent=2), encoding="utf-8")
    else:
        print("\nEnsemble: skipped (needs the imported XLM-R results).")

    # ---- Step 21: validation-learned class bias on the best validation model (optional, promoted only on a clear gain)
    pool = [c for c in cands if c["val"] is not None]
    top = max(pool, key=lambda c: rank_key(full_metrics(yv, c["val"].argmax(1))))
    bias, gain = fit_class_bias(yv, top["val"])
    bias_info = {"base_model": top["name"], "log_bias": bias, "val_macro_f1_gain": gain, "min_gain_to_promote": 0.003,
                 "promoted": gain >= 0.003}
    print(f"\nClass-bias calibration on {top['name']} (validation): offsets {bias}, gain {gain:+.4f} -> "
          f"{'PROMOTED' if bias_info['promoted'] else 'not promoted'}")
    if bias_info["promoted"]:
        cands.append({"name": f"{top['name']} + validation-tuned class bias", "input": top["input"],
                      "val": apply_bias(top["val"], bias), "test": apply_bias(top["test"], bias),
                      "best_val": fast_macro_f1(yv, apply_bias(top["val"], bias).argmax(1)),
                      "id": f"{top['id']}_classbias", "kind": "bias", "base_id": top["id"], "log_bias": bias,
                      "notes": f"per-class log-probability offsets {bias} learned on validation"})
    (EXP / "class_bias_calibration.json").write_text(json.dumps(bias_info, indent=2), encoding="utf-8")

    # ---- Step 23 + 22: final test evaluation of the validation-selected models (+ temperature fitted on validation)
    FINAL.mkdir(parents=True, exist_ok=True)
    rows, results = [], {}
    for c in cands:
        tag = c["id"]
        folder = FINAL / tag
        m_eval = evaluate_and_save(tag=tag, model_name=c["name"], input_col="cleaned_text", test_df=test_df,
                                   probs=c["test"], out_dir=folder, best_val_macro_f1=c["best_val"], n_errors=100)
        fm = full_metrics(yt, c["test"].argmax(1), test_df["language_type"])
        cal = calibration(yt, c["test"])
        save_normalized_cm(yt, c["test"].argmax(1), folder, c["name"])
        out = {**fm, "calibration_test": cal, "best_validation_macro_f1": c["best_val"], "notes": c["notes"],
               "confusion_matrix": m_eval["confusion_matrix"]}
        if c["val"] is not None:
            T = fit_temperature(yv, c["val"])
            out["calibration_validation"] = calibration(yv, c["val"])
            out["temperature_scaling"] = {"T_fitted_on_validation": T,
                                          "test_ece_before": cal["ece_10_bins"],
                                          "test_ece_after": calibration(yt, apply_temperature(c["test"], T))["ece_10_bins"]}
            c["temperature"] = T
        (folder / "test_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        results[tag] = {"cand": c, "m": fm, "cal": cal, "out": out}
        pl = fm["per_language"]
        rows.append({"Model": c["name"], "Input": c["input"], "Accuracy": fm["accuracy"],
                     "Macro Precision": fm["macro_precision"], "Macro Recall": fm["macro_recall"],
                     "Macro F1": fm["macro_f1"], "Weighted Precision": fm["weighted_precision"],
                     "Weighted Recall": fm["weighted_recall"], "Weighted F1": fm["weighted_f1"],
                     **{f"{k} {s.title() if s != 'f1' else 'F1'}": fm["per_class"][k][s]
                        for k in CLASS_NAMES for s in ("precision", "recall", "f1", "support")},
                     "English Accuracy": pl["english"]["accuracy"], "English Macro F1": pl["english"]["macro_f1"],
                     "Telugu Accuracy": pl["telugu"]["accuracy"], "Telugu Macro F1": pl["telugu"]["macro_f1"],
                     "Code-mixed Accuracy": pl["code_mixed"]["accuracy"],
                     "Code-mixed Macro F1": pl["code_mixed"]["macro_f1"], "ECE (test)": cal["ece_10_bins"],
                     "Best Validation Macro F1": c["best_val"], "Notes": c["notes"]})
        if tag == "ensemble_validation_tuned":
            upsert_leaderboard(leaderboard_row_from(tag, "weighted probability ensemble", c["notes"], "cleaned_text",
                                                    {"weights": c["notes"]}, full_metrics(yv, c["val"].argmax(1)),
                                                    test=fm, notes="weights chosen on validation; scored on test once"))
    rows.append({"Model": "Best improved classical model", "Input": "cleaned_text",
                 "Notes": classical_note + " No configuration beat A+ on validation: A+ remains the best classical model "
                                           "(see the A+ row; no separate test run)."})
    if "xlmr" not in fam_best:
        rows.append({"Model": "Multilingual XLM-R Model B", "Input": "cleaned_text", "Notes": "Not trained / pending GPU run."})
    if not alts:
        rows.append({"Model": "Best alternative Transformer", "Input": "cleaned_text", "Notes": "Not trained / pending GPU run."})
    if not any(c["id"] == "ensemble_validation_tuned" for c in cands):
        rows.append({"Model": "Best ensemble", "Input": "cleaned_text",
                     "Notes": ("Not promoted: no validation gain over its best member." if ensemble_cfg
                               else "Not trained / pending GPU run (needs the Transformer results).")})
    rows.append({"Model": "Model C (context-aware)", "Input": "context_text", "Notes": MODEL_C_SKIP + " " + MODEL_C_MESSAGE})
    comp = pd.DataFrame(rows)
    comp.to_csv(EXP / "final_model_comparison.csv", index=False, encoding="utf-8-sig")

    # ---- Step 24: absolute change vs A+
    ap = results["model_a_plus"]["m"]
    imp = []
    for tag, r in results.items():
        if tag in ("model_a_plus",):
            continue
        m = r["m"]
        imp.append({"Model": r["cand"]["name"], "Accuracy": m["accuracy"] - ap["accuracy"],
                    "Macro Precision": m["macro_precision"] - ap["macro_precision"],
                    "Macro Recall": m["macro_recall"] - ap["macro_recall"], "Macro F1": m["macro_f1"] - ap["macro_f1"],
                    "Weighted F1": m["weighted_f1"] - ap["weighted_f1"],
                    **{f"{k} F1": m["per_class"][k]["f1"] - ap["per_class"][k]["f1"] for k in CLASS_NAMES},
                    **{f"{LANG_NAME[l]} Macro F1": m["per_language"][l]["macro_f1"] - ap["per_language"][l]["macro_f1"]
                       for l in LANG_NAME}})
    pd.DataFrame(imp).to_csv(EXP / "improvements_over_a_plus.csv", index=False, encoding="utf-8-sig")

    # ---- Step 26: McNemar + bootstrap vs A+ (and ensemble vs its best single member)
    pa = results["model_a_plus"]["cand"]["test"].argmax(1)
    stats = {}
    for tag, r in results.items():
        if tag == "model_a_plus":
            continue
        pb = r["cand"]["test"].argmax(1)
        stats[f"{r['cand']['name']} vs Multilingual A+"] = {"mcnemar": mcnemar(yt, pa, pb),
                                                             "bootstrap": bootstrap(yt, pa, pb)}
    if "ensemble_validation_tuned" in results and xl is not None:
        best_t = max([results[c["id"]] for c in cands if c.get("kind") == "transformer"],
                     key=lambda r: rank_key(r["cand"]["val_metrics"]))
        pt = best_t["cand"]["test"].argmax(1)
        stats[f"Ensemble vs {best_t['cand']['name']}"] = {
            "mcnemar": mcnemar(yt, pt, results["ensemble_validation_tuned"]["cand"]["test"].argmax(1)),
            "bootstrap": bootstrap(yt, pt, results["ensemble_validation_tuned"]["cand"]["test"].argmax(1))}
    (EXP / "statistical_comparison.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---- Step 25: Transformer / ensemble error analysis vs A+
    trans_rows, ens_rows, transitions = [], [], []
    for tag, r in results.items():
        if r["cand"].get("kind") not in ("transformer", "ensemble", "bias"):
            continue
        df, trans = error_vs_a_plus(test_df, yt, pa, results["model_a_plus"]["cand"]["test"].max(1),
                                    r["cand"]["test"], r["cand"]["name"])
        (ens_rows if r["cand"].get("kind") == "ensemble" else trans_rows).append(df)
        transitions += trans
    if trans_rows:
        pd.concat(trans_rows).to_csv(EXP / "transformer_error_analysis.csv", index=False, encoding="utf-8-sig")
    if ens_rows:
        pd.concat(ens_rows).to_csv(EXP / "ensemble_error_analysis.csv", index=False, encoding="utf-8-sig")
    if transitions:
        pd.DataFrame(transitions).to_csv(EXP / "confusion_changes_vs_a_plus.csv", index=False, encoding="utf-8-sig")

    # ---- Step 7: possible label ambiguity (high-confidence mistakes on VALIDATION; never relabelled)
    amb = []
    for c in sorted({id(c): c for c in (top, a_plus)}.values(), key=lambda c: c["name"]):
        pv = c["val"]
        pred, conf = pv.argmax(1), pv.max(1)
        wrong = np.flatnonzero(pred != yv)
        sel = wrong[np.argsort(-conf[wrong])][:50]
        amb.append(pd.DataFrame({"text": val_df["cleaned_text"].to_numpy()[sel],
                                 "true_label": [CLASS_NAMES[i] for i in yv[sel]],
                                 "predicted_label": [CLASS_NAMES[i] for i in pred[sel]], "confidence": conf[sel],
                                 "language_type": val_df["language_type"].to_numpy()[sel],
                                 "source_dataset": val_df["source_dataset"].to_numpy()[sel],
                                 "annotations": val_df["annotations"].to_numpy()[sel], "model": c["name"],
                                 "split": "validation",
                                 "note": "High-confidence disagreement; POSSIBLE annotation ambiguity/noise. Not relabelled."}))
    pd.concat(amb).to_csv(EXP / "possible_label_ambiguity.csv", index=False, encoding="utf-8-sig")

    # ---- final model selection (written for the web app)
    selection = select_final(cands, results, stats, yv)
    (paths.MULTI / "final_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # ---- plots
    plot = comp.dropna(subset=["Macro F1"])
    if len(plot):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        m1 = plot.melt(id_vars="Model", value_vars=["Negative F1", "Neutral F1", "Positive F1", "Macro F1"])
        sns.barplot(data=m1, x="variable", y="value", hue="Model", ax=axes[0])
        axes[0].set_ylim(0.5, 0.9); axes[0].set_title("Test F1 per class (multilingual test, n=15,124)")
        axes[0].set_xlabel(""); axes[0].legend(fontsize=7)
        m2 = plot.melt(id_vars="Model", value_vars=["English Macro F1", "Telugu Macro F1", "Code-mixed Macro F1"])
        sns.barplot(data=m2, x="variable", y="value", hue="Model", ax=axes[1])
        axes[1].set_ylim(0.5, 0.9); axes[1].set_title("Test macro F1 per language"); axes[1].set_xlabel("")
        axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(EXP / "final_model_comparison.png", dpi=150); plt.close(fig)

    # ---- print
    pd.set_option("display.width", 250)
    show = ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "Negative F1",
            "Neutral F1", "Positive F1", "English Macro F1", "Telugu Macro F1", "Code-mixed Macro F1",
            "Best Validation Macro F1"]
    print("\nFINAL TEST COMPARISON (models selected on validation; multilingual test split, n=15,124)")
    print(comp[show].round(4).to_string(index=False))
    print("\nCalibration (test): " + "; ".join(
        f"{r['cand']['name']}: ECE {r['cal']['ece_10_bins']:.4f}"
        + (f" (T={r['out']['temperature_scaling']['T_fitted_on_validation']:.2f} -> "
           f"{r['out']['temperature_scaling']['test_ece_after']:.4f})" if "temperature_scaling" in r["out"] else "")
        for r in results.values()))
    for k, v in stats.items():
        mc, bs = v["mcnemar"], v["bootstrap"]
        print(f"{k}: McNemar p={mc['p_value']:.3g} (reference right/other wrong {mc['a_right_b_wrong']}, "
              f"reference wrong/other right {mc['a_wrong_b_right']}); macro F1 diff {bs['macro_f1_diff_mean']:+.4f} "
              f"95% CI [{bs['macro_f1_diff_ci95'][0]:+.4f}, {bs['macro_f1_diff_ci95'][1]:+.4f}]")
    print(f"\nFINAL SELECTED MODEL: {selection['selected_model_name']} (served id {selection['served_model_id']})")
    for line in selection["reasoning"]:
        print("  -", line)
    print(f"\nModel C: {MODEL_C_SKIP}\n{MODEL_C_MESSAGE}")
    stage_banner("final evaluation, comparison tables and final model selection",
                 "import Colab Transformer results with --import" if not runs else "install the selected model weights "
                 "if a Transformer was selected (python -m pipeline.install_phase3_model), then restart the backend")
    return selection


# ------------------------------------------------------------------ helpers for the Phase-3 additions
def rank_key(m: dict):
    return (m["macro_f1"], m["macro_recall"], m["accuracy"])


def transformer_note(c: dict, n_runs: int, fam: str) -> str:
    loss = c["loss"] + ("" if c["loss"] != "focal" else f" gamma {c['focal_gamma']:g}")
    return (f"{c['model']}, lr {c['learning_rate']:g}, loss {loss}, label smoothing {c.get('label_smoothing', 0):g}, "
            f"wd {c.get('weight_decay', 0.01):g}, warmup {c.get('warmup_ratio', 0.1):g}, batch "
            f"{c.get('effective_batch_size', c['batch_size'])}, max_len {c['max_length']}, "
            f"{'language tokens, ' if c['lang_tokens'] else ''}best epoch {c['best_epoch']:g}/{c['epochs_max']}; "
            f"best of {n_runs} {fam} runs by validation macro F1")


def ensemble_grid(k: int) -> list[tuple]:
    if k == 2:
        return [(a, round(1 - a, 2)) for a in (0.1, 0.2, 0.3, 0.4, 0.5)]
    grid = [(0.0, x, round(1 - x, 2)) for x in (0.3, 0.4, 0.5, 0.6, 0.7)]          # Transformers only
    for a in (0.1, 0.2, 0.3):
        for x in np.round(np.arange(0.1, 1 - a - 0.1 + 1e-9, 0.1), 2):
            grid.append((a, float(x), round(1 - a - float(x), 2)))
    return grid


def apply_bias(p: np.ndarray, bias) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1)) + np.asarray(bias)
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def fit_class_bias(y, p) -> tuple[list[float], float]:
    """Per-class log-probability offsets (Positive fixed at 0) maximising VALIDATION macro F1."""
    base = fast_macro_f1(y, p.argmax(1))
    best, best_b = base, [0.0, 0.0, 0.0]
    steps = np.round(np.arange(-0.4, 0.4001, 0.05), 2)
    logp = np.log(np.clip(p, 1e-12, 1))
    for bn in steps:
        for bu in steps:
            f = fast_macro_f1(y, (logp + np.array([bn, bu, 0.0])).argmax(1))
            if f > best + 1e-12:
                best, best_b = f, [float(bn), float(bu), 0.0]
    return best_b, float(best - base)


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    return apply_bias(np.exp(np.log(np.clip(p, 1e-12, 1)) / T), [0, 0, 0])


def fit_temperature(y, p) -> float:
    from scipy.optimize import minimize_scalar
    logp = np.log(np.clip(p, 1e-12, 1))

    def nll(t):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
    return float(minimize_scalar(nll, bounds=(0.25, 5.0), method="bounded").x)


def error_vs_a_plus(test_df, y, pa, conf_a, probs, name):
    pm, cm = probs.argmax(1), probs.max(1)
    involved = (pa != y) | (pm != y)
    cat = np.where((pa != y) & (pm == y), "fixed by model (A+ wrong, model right)",
          np.where((pa == y) & (pm != y), "new error (A+ right, model wrong)",
          np.where(pa == pm, "both wrong (same prediction)", "both wrong (different predictions)")))
    df = pd.DataFrame({"model": name, "text": test_df["cleaned_text"].to_numpy(),
                       "language": test_df["language_type"].to_numpy(),
                       "actual": [CLASS_NAMES[i] for i in y], "prediction": [CLASS_NAMES[i] for i in pm],
                       "confidence": cm, "A+ prediction": [CLASS_NAMES[i] for i in pa], "A+ confidence": conf_a,
                       "error_category": cat})[involved]
    trans = []
    for a, b in [("Positive", "Neutral"), ("Neutral", "Positive"), ("Neutral", "Negative"), ("Negative", "Neutral"),
                 ("Negative", "Positive"), ("Positive", "Negative")]:
        ia, ib = CLASS_NAMES.index(a), CLASS_NAMES.index(b)
        a_err = (y == ia) & (pa == ib)
        trans.append({"model": name, "confusion": f"{a} -> {b}", "A+ count": int(a_err.sum()),
                      "model count": int(((y == ia) & (pm == ib)).sum()),
                      "A+ errors fixed by model": int((a_err & (pm == y)).sum())})
    return df, trans


SERVED_ID = {"model_a_plus": "tfidf_plus_multilingual", "model_a": "tfidf_multilingual"}


def select_final(cands, results, stats, yv) -> dict:
    """Final model: best VALIDATION rank, simplicity gate (needs >= 0.005 validation macro F1 to justify a more complex
    model), then a test go/no-go against A+ (McNemar p < 0.05 AND bootstrap 95% CI of the macro F1 difference > 0)."""
    reasons = []
    pool = [c for c in cands if c["val"] is not None]
    ranked = sorted(pool, key=lambda c: rank_key(full_metrics(yv, c["val"].argmax(1))), reverse=True)
    complexity = {"tfidf": 0, None: 0, "transformer": 1, "ensemble": 2, "bias": 2}
    choice = ranked[0]
    reasons.append(f"Highest validation macro F1: {choice['name']} ({choice['best_val']:.4f}).")
    for c in ranked[1:]:
        if complexity[c.get("kind")] < complexity[choice.get("kind")] and \
                fast_macro_f1(yv, choice["val"].argmax(1)) - fast_macro_f1(yv, c["val"].argmax(1)) < 0.005:
            reasons.append(f"{choice['name']} beats the simpler {c['name']} by < 0.005 validation macro F1 -> prefer "
                           f"the simpler model.")
            choice = c
    if choice["id"] != "model_a_plus":
        st = stats.get(f"{choice['name']} vs Multilingual A+")
        ok = st and st["mcnemar"]["p_value"] < 0.05 and st["bootstrap"]["macro_f1_diff_ci95"][0] > 0
        if not ok:
            reasons.append(f"{choice['name']} is not significantly better than A+ on test (McNemar / bootstrap) -> keep A+.")
            choice = next(c for c in cands if c["id"] == "model_a_plus")
        else:
            reasons.append(f"Significantly better than A+ on test: McNemar p={st['mcnemar']['p_value']:.3g}, macro F1 "
                           f"difference 95% CI [{st['bootstrap']['macro_f1_diff_ci95'][0]:+.4f}, "
                           f"{st['bootstrap']['macro_f1_diff_ci95'][1]:+.4f}].")
    kind = choice.get("kind")
    if kind == "transformer":
        served = f"{choice['family']}_phase3"
    elif kind == "ensemble":
        served = "ensemble_phase3"
    elif kind == "bias":
        base = next(c for c in cands if c["id"] == choice["base_id"])
        served = SERVED_ID.get(base["id"]) or (f"{base['family']}_phase3" if base.get("kind") == "transformer"
                                               else "ensemble_phase3")
    else:
        served = SERVED_ID.get(choice["id"], "tfidf_plus_multilingual")
    m = results[choice["id"]]["m"]
    needs = sorted({c["family"] for c in cands if c.get("kind") == "transformer"
                    and (served == f"{c['family']}_phase3" or (served == "ensemble_phase3"
                                                               and c["id"] in (json.loads((EXP / "ensemble_validation_tuned" / "config.json").read_text())["member_ids"]
                                                                               if (EXP / "ensemble_validation_tuned" / "config.json").exists() else [])))})
    return {"selected_model_name": choice["name"], "selected_id": choice["id"], "served_model_id": served,
            "class_log_bias": choice.get("log_bias"), "temperature": choice.get("temperature"),
            "requires_transformer_weights": needs,
            "test": {k: m[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
            "validation_macro_f1": choice["best_val"], "reasoning": reasons,
            "rule": select_final.__doc__.strip()}


if __name__ == "__main__":
    main()
