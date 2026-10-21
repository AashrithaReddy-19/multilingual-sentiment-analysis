"""Steps 8-12: one Transformer experiment on the fixed multilingual split (Colab GPU; CPU only for --smoke-rows).

TRAIN mode (default) fits on train.csv, evaluates on validation.csv every epoch, keeps the best epoch by validation
macro F1 (early stopping patience 2) and writes VALIDATION outputs only. test.csv is not opened.

    python train_transformer.py --data-dir DATA --out-dir RUNS --exp-id xlmr_lr2e-5 --model xlm-roberta-base \
        --lr 2e-5 --epochs 5 --max-length 128 [--loss ce|wce|focal --gamma 2.0] [--lang-tokens]
        [--class-weight-scheme inverse|sqrt_inverse|effective] [--sampler random|lang_balanced|lang_sqrt]
        [--text-variant v1_current|v3_repeat_norm|v4_repeat_norm_emoticons] [--scheduler linear|cosine]
        [--freeze-lower-layers N --freeze-mode epoch1|permanent]

Runs with --lang-tokens also write validation predictions where the language token comes from the app's heuristic
detector instead of the dataset metadata ("as served", detected_lang/), because that is all the web app has.

TEST mode is run ONLY for configurations already selected on validation:

    python train_transformer.py --data-dir DATA --out-dir RUNS --exp-id xlmr_lr2e-5 --predict-test MODEL_DIR

--smoke-rows N is only for checking that the code runs; its outputs are flagged and are not results.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent, HERE):            # backend/ (local) or package root (Colab)
    if (cand / "sentiment_core").exists():
        sys.path.insert(0, str(cand))
from sentiment_core.experiment_io import full_metrics, save_eval_files  # noqa: E402
from sentiment_core.labels import ID2LABEL, LABEL2ID  # noqa: E402
from sentiment_core.language import detect_language  # noqa: E402
from sentiment_core.text_variants import VARIANTS, apply_variant  # noqa: E402

SEED = 42
LANG_TOKENS = {"english": "[LANG_ENGLISH]", "telugu": "[LANG_TELUGU]", "code_mixed": "[LANG_CODE_MIXED]"}


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(data_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"{split}.csv", encoding="utf-8-sig", keep_default_na=False, na_values=[""],
                       low_memory=False)


def model_inputs(df: pd.DataFrame, lang_tokens: bool, input_col: str = "cleaned_text", variant: str = "v1_current",
                 lang_source: str = "metadata") -> list[str]:
    """Model input text. The variant only changes the representation; the frozen rows/labels are untouched.
    lang_source: "metadata" = dataset language_type (training/selection), "detected" = the app's heuristic detector."""
    text = pd.Series(apply_variant(df[input_col].astype(str), variant), index=df.index)
    if lang_tokens:
        langs = df["language_type"] if lang_source == "metadata" else \
            df["cleaned_text"].astype(str).map(detect_language)
        text = langs.map(LANG_TOKENS) + " " + text
    return text.tolist()


def class_weights(labels, scheme: str, beta: float = 0.9999) -> np.ndarray:
    """Per-class loss weights from TRAINING label counts, scaled so the count-weighted mean weight is 1."""
    n = np.bincount(np.asarray(labels, dtype=int), minlength=3).astype(np.float64)
    if scheme == "inverse":
        w = 1 / n
    elif scheme == "sqrt_inverse":
        w = 1 / np.sqrt(n)
    elif scheme == "effective":                       # Cui et al. 2019, effective number of samples
        w = (1 - beta) / (1 - beta ** n)
    else:
        raise ValueError(scheme)
    return (w * n.sum() / (w * n).sum()).astype(np.float32)


def language_sample_weights(langs, scheme: str) -> np.ndarray:
    """Per-row sampling weights for the TRAIN set only: P(row) ~ n_lang^-alpha.
    lang_balanced (alpha 1): each language 1/3 of the draws; lang_sqrt (alpha 0.5): language share ~ sqrt(rows)."""
    langs = pd.Series(np.asarray(langs))
    n = langs.map(langs.value_counts()).to_numpy(dtype=np.float64)
    alpha = {"lang_balanced": 1.0, "lang_sqrt": 0.5}[scheme]
    return n ** -alpha


def model_prefix(model_name: str) -> str:
    return "passage: " if "multilingual-e5" in model_name.lower() else ""


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--loss", choices=["ce", "wce", "focal"], default="ce")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--class-weight-scheme", choices=["inverse", "sqrt_inverse", "effective"], default="inverse",
                    help="class weights for --loss wce, from training label counts")
    ap.add_argument("--sampler", choices=["random", "lang_balanced", "lang_sqrt"], default="random",
                    help="training-set sampling only; validation/test are never resampled")
    ap.add_argument("--text-variant", choices=sorted(VARIANTS), default="v1_current")
    ap.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    ap.add_argument("--lang-tokens", action="store_true")
    ap.add_argument("--input-col", choices=["cleaned_text", "context_text"], default="cleaned_text",
                    help="context_text = Model C; refused unless rows carry real earlier-message context")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="CE/WCE only")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2, help="early stopping patience (epochs)")
    ap.add_argument("--freeze-lower-layers", type=int, default=0,
                    help="freeze embeddings + the lowest N encoder layers (see --freeze-mode)")
    ap.add_argument("--freeze-mode", choices=["epoch1", "permanent"], default="epoch1",
                    help="epoch1 = frozen during epoch 1 then unfrozen (gradual unfreezing); permanent = never trained")
    ap.add_argument("--no-fp16", action="store_true", help="fp32 even on CUDA (e.g. mDeBERTa is unstable in fp16)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--keep-model", action="store_true", help="keep the best model in OUT/EXP/model")
    ap.add_argument("--predict-test", type=Path, default=None, help="MODEL_DIR of a validation-selected model")
    ap.add_argument("--smoke-rows", type=int, default=0)
    return ap.parse_args(argv)


def tokenizer_for(name_or_dir, lang_tokens: bool):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(name_or_dir), use_fast=True)
    if lang_tokens:
        tok.add_special_tokens({"additional_special_tokens": list(LANG_TOKENS.values())})
    return tok


def to_dataset(df, tok, max_length, lang_tokens, with_labels=True, input_col="cleaned_text", prefix="",
               variant="v1_current", lang_source="metadata"):
    import datasets
    cols = {"text": [prefix + text for text in model_inputs(df, lang_tokens, input_col, variant, lang_source)]}
    if with_labels:
        cols["labels"] = df["label"].astype(int).tolist()
    ds = datasets.Dataset.from_dict(cols)
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_length), batched=True,
                  remove_columns=["text"])


def predict_probs(model, tok, ds, batch_size):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    dev = next(model.parameters()).device
    model.eval()
    out = []
    dl = DataLoader(ds.remove_columns([c for c in ds.column_names if c == "labels"]), batch_size=batch_size,
                    collate_fn=DataCollatorWithPadding(tok))
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(**b).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def overfitting_signal(hist: pd.DataFrame) -> str | None:
    """Validation macro F1 fell for 2 consecutive epochs while training loss kept falling."""
    if len(hist) < 3 or hist["train_loss"].isna().any():
        return None
    f, loss = hist["eval_macro_f1"].to_numpy(), hist["train_loss"].to_numpy()
    for i in range(2, len(hist)):
        if f[i] < f[i - 1] < f[i - 2] and loss[i] < loss[i - 1] < loss[i - 2]:
            return f"val macro F1 fell at epochs {hist['epoch'].iloc[i - 1]:g}-{hist['epoch'].iloc[i]:g} while train loss fell"
    return None


def lower_layer_params(model, n: int, embeddings: bool = True) -> list:
    base = getattr(model, model.base_model_prefix)
    return (list(base.embeddings.parameters()) if embeddings else []) + \
        [p for layer in base.encoder.layer[:n] for p in layer.parameters()]


def freeze_lower_layers_callback(model, n: int):
    """Freeze embeddings + the lowest n encoder layers during epoch 1, then train everything."""
    from transformers import TrainerCallback

    params = lower_layer_params(model, n)

    class FreezeLowerLayers(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            for p in params:
                p.requires_grad_(False)

        def on_epoch_end(self, args, state, control, **kw):
            for p in params:
                p.requires_grad_(True)

    return FreezeLowerLayers()


def run_test(args) -> dict:
    """Score a validation-selected model ONCE on test.csv."""
    import torch
    from transformers import AutoModelForSequenceClassification
    run = args.out_dir / args.exp_id
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if (run / "test_metrics.json").exists():
        raise SystemExit(f"{run} already has test results; the test split is evaluated only once per config.")
    test = read(args.data_dir, "test")
    if args.smoke_rows:
        test = test.groupby("label", group_keys=False).head(args.smoke_rows // 3)
    tok = tokenizer_for(args.predict_test, False)          # saved tokenizer already contains any language tokens
    prefix = cfg.get("input_prefix", model_prefix(cfg["model"]))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.predict_test))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    variant = cfg.get("text_variant", "v1_current")
    probs = predict_probs(model, tok, to_dataset(test, tok, cfg["max_length"], cfg["lang_tokens"], False,
                                                 cfg.get("input_col", "cleaned_text"), prefix, variant),
                          args.eval_batch_size)
    m = save_eval_files(run, test["label"], probs, test, "test", f"{args.exp_id}")
    if cfg["lang_tokens"]:            # as served: language token from the app's heuristic detector
        det = predict_probs(model, tok, to_dataset(test, tok, cfg["max_length"], True, False,
                                                   cfg.get("input_col", "cleaned_text"), prefix, variant, "detected"),
                            args.eval_batch_size)
        save_eval_files(run / "detected_lang", test["label"], det, test, "test", f"{args.exp_id} (detected language)")
    m["smoke_test"] = bool(args.smoke_rows)
    (run / "test_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"[{args.exp_id}] TEST macro F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")
    return m


def run_train(args) -> dict:
    import torch
    from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding, EarlyStoppingCallback,
                              Trainer, TrainingArguments, set_seed)

    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.smoke_rows:
        raise SystemExit("No CUDA GPU: full Transformer runs are GPU-only (use Colab).")
    set_seed(SEED)
    run = args.out_dir / args.exp_id
    if (run / "validation_metrics.json").exists():
        print(f"[{args.exp_id}] already finished; skipping.")
        return json.loads((run / "validation_metrics.json").read_text(encoding="utf-8"))
    run.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint_dir or (run / "_checkpoints")
    shutil.rmtree(ckpt, ignore_errors=True)

    train, val = read(args.data_dir, "train"), read(args.data_dir, "validation")
    split_hashes = {s: sha256(args.data_dir / f"{s}.csv") for s in ("train", "validation", "test")}
    if args.smoke_rows:
        train, val = (d.groupby("label", group_keys=False).head(args.smoke_rows // 3).reset_index(drop=True)
                      for d in (train, val))

    if args.input_col == "context_text":
        from sentiment_core.context import has_real_context
        if not (has_real_context(train) and has_real_context(val)):
            raise SystemExit("True context-aware sentiment training could not be performed because the available "
                             "datasets do not contain reliable conversation structure.")
    tok = tokenizer_for(args.model, args.lang_tokens)
    if args.input_col == "context_text":
        tok.add_special_tokens({"additional_special_tokens": tok.additional_special_tokens +
                                ["[PREVIOUS_2]", "[PREVIOUS_1]", "[CURRENT]"]})
    prefix = model_prefix(args.model)
    tv = args.text_variant
    tr = to_dataset(train, tok, args.max_length, args.lang_tokens, input_col=args.input_col, prefix=prefix, variant=tv)
    va = to_dataset(val, tok, args.max_length, args.lang_tokens, input_col=args.input_col, prefix=prefix, variant=tv)

    class_w = class_weights(train["label"], args.class_weight_scheme)     # from TRAINING frequencies only
    val_lang = val["language_type"].to_numpy()

    def compute_metrics(ev):
        logits, labels = ev                      # eval order == validation.csv order (sequential eval sampler)
        m = full_metrics(labels, np.argmax(logits, axis=-1), val_lang)
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                **{f"{k}_{c}": v[k] for c, v in m["per_class"].items() for k in ("precision", "recall", "f1")},
                **{f"{lt}_{k}": v[k] for lt, v in m["per_language"].items() for k in ("accuracy", "macro_f1")}}

    loss_kind, gamma, smoothing = args.loss, args.gamma, args.label_smoothing
    if loss_kind == "focal" and smoothing:
        raise SystemExit("label smoothing is only defined here for ce/wce")

    row_w = language_sample_weights(train["language_type"], args.sampler) if args.sampler != "random" else None

    class LossTrainer(Trainer):
        def _get_train_sampler(self, *a, **kw):
            if row_w is None:
                return super()._get_train_sampler(*a, **kw)
            from torch.utils.data import WeightedRandomSampler   # same epoch size, language-reweighted draws
            return WeightedRandomSampler(torch.as_tensor(row_w, dtype=torch.double), num_samples=len(row_w),
                                         replacement=True, generator=torch.Generator().manual_seed(SEED))

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.float()
            w = torch.tensor(class_w, device=logits.device) if loss_kind == "wce" else None
            if loss_kind == "focal":
                logp = torch.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
                loss = (-(1 - logp.exp()) ** gamma * logp).mean()
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=w, label_smoothing=smoothing)
            return (loss, outputs) if return_outputs else loss

    params = inspect.signature(TrainingArguments.__init__).parameters
    fp16 = use_cuda and not args.no_fp16
    kw = dict(output_dir=str(ckpt), num_train_epochs=1 if args.smoke_rows else args.epochs, learning_rate=args.lr,
              weight_decay=args.weight_decay, save_strategy="epoch", load_best_model_at_end=True,
              metric_for_best_model="macro_f1", greater_is_better=True, per_device_train_batch_size=args.batch_size,
              gradient_accumulation_steps=args.grad_accum, per_device_eval_batch_size=args.eval_batch_size,
              fp16=fp16, seed=SEED, logging_strategy="epoch", save_total_limit=2, report_to="none",
              dataloader_num_workers=2 if use_cuda else 0, lr_scheduler_type=args.scheduler)
    kw["evaluation_strategy" if "evaluation_strategy" in params else "eval_strategy"] = "epoch"
    if row_w is not None:
        pass                                # the language sampler replaces length grouping
    elif "group_by_length" in params:       # batches of similar length -> much less padding, same data per epoch
        kw["group_by_length"] = True
    elif "train_sampling_strategy" in params:   # transformers v5 name
        kw["train_sampling_strategy"] = "group_by_length"
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = args.warmup_ratio
    else:                                   # transformers v5: a float < 1 in warmup_steps is a ratio
        kw["warmup_steps"] = args.warmup_ratio
    if "data_seed" in params:
        kw["data_seed"] = SEED

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3, id2label=ID2LABEL,
                                                               label2id=LABEL2ID)
    if len(tok) > model.get_input_embeddings().num_embeddings:     # only grow (added special tokens)
        model.resize_token_embeddings(len(tok))
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    frozen_embeddings = False
    if args.freeze_lower_layers and args.freeze_mode == "epoch1":
        callbacks.append(freeze_lower_layers_callback(model, args.freeze_lower_layers))
        frozen_embeddings = True
    elif args.freeze_lower_layers:            # permanent; keep embeddings trainable if new language tokens were added
        frozen_embeddings = not args.lang_tokens
        for p in lower_layer_params(model, args.freeze_lower_layers, embeddings=frozen_embeddings):
            p.requires_grad_(False)
    tok_kw = {"processing_class": tok} if "processing_class" in inspect.signature(Trainer.__init__).parameters \
        else {"tokenizer": tok}
    trainer = LossTrainer(model=model, args=TrainingArguments(**kw), train_dataset=tr, eval_dataset=va,
                          data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                          callbacks=callbacks, **tok_kw)
    t0 = time.time()
    trainer.train()
    secs = round(time.time() - t0, 1)
    hist = pd.DataFrame([h for h in trainer.state.log_history if "eval_macro_f1" in h])
    train_loss = {round(h["epoch"]): h["loss"] for h in trainer.state.log_history if "loss" in h and "eval_loss" not in h}
    hist.insert(1, "train_loss", hist["epoch"].round().map(train_loss))
    hist.to_csv(run / "validation_history.csv", index=False)
    best_epoch = float(hist.loc[hist["eval_macro_f1"].idxmax(), "epoch"])
    overfit = overfitting_signal(hist)

    probs = predict_probs(trainer.model, tok, va, args.eval_batch_size)   # best epoch (load_best_model_at_end)
    m = save_eval_files(run, val["label"], probs, val, "validation", f"{args.exp_id} ({args.model})")
    if args.lang_tokens:                 # as served: language token from the app's heuristic detector
        va_det = to_dataset(val, tok, args.max_length, True, input_col=args.input_col, prefix=prefix, variant=tv,
                            lang_source="detected")
        md = save_eval_files(run / "detected_lang", val["label"],
                             predict_probs(trainer.model, tok, va_det, args.eval_batch_size),
                             val, "validation", f"{args.exp_id} (detected language)")
        print(f"[{args.exp_id}] VAL macro F1 with detected-language tokens (as served): {md['macro_f1']:.4f}")
    if abs(m["macro_f1"] - float(trainer.state.best_metric)) > 2e-3:
        print(f"WARNING: re-scored validation macro F1 {m['macro_f1']:.4f} != trainer best {trainer.state.best_metric:.4f}")
    cfg = {"experiment_id": args.exp_id, "model": args.model, "learning_rate": args.lr, "epochs_max": args.epochs,
           "max_length": args.max_length, "loss": args.loss, "focal_gamma": args.gamma if args.loss == "focal" else None,
           "class_weights": class_w.tolist() if args.loss == "wce" else None,
           "class_weight_scheme": args.class_weight_scheme if args.loss == "wce" else None,
           "lang_tokens": args.lang_tokens, "sampler": args.sampler, "text_variant": args.text_variant,
           "scheduler": args.scheduler, "freeze_mode": args.freeze_mode if args.freeze_lower_layers else None,
           "frozen_embeddings": frozen_embeddings,
           "batch_size": args.batch_size, "grad_accum": args.grad_accum,
           "effective_batch_size": args.batch_size * args.grad_accum, "weight_decay": args.weight_decay,
           "warmup_ratio": args.warmup_ratio, "label_smoothing": args.label_smoothing,
           "freeze_lower_layers_epoch1": args.freeze_lower_layers,
           "early_stopping_patience": args.patience, "metric_for_best_model": "macro_f1", "seed": SEED, "fp16": fp16,
           "no_fp16_flag": bool(args.no_fp16),
           "group_by_length": row_w is None, "input_col": args.input_col,
           "input_prefix": prefix,
           "input": args.input_col + (" with language token" if args.lang_tokens else ""),
           "best_epoch": best_epoch, "epochs_run": float(trainer.state.epoch), "train_seconds": secs,
           "best_checkpoint": f"epoch {best_epoch:g} ({trainer.state.best_model_checkpoint}); intermediate checkpoints "
                              "deleted, best weights kept per model family by colab_runner",
           "still_improving_at_last_epoch": bool(best_epoch >= float(trainer.state.epoch) - 1e-6),
           "overfitting_signal": overfit,
           "trainer_best_val_macro_f1": float(trainer.state.best_metric),
           "device": torch.cuda.get_device_name(0) if use_cuda else "cpu", "split_sha256": split_hashes,
           "smoke_test": bool(args.smoke_rows), "test_used": False}
    (run / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if args.keep_model:
        trainer.save_model(str(run / "model"))
        tok.save_pretrained(str(run / "model"))
    shutil.rmtree(ckpt, ignore_errors=True)
    print(f"[{args.exp_id}] best epoch {best_epoch:g}: VAL macro F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} "
          f"P {m['macro_precision']:.4f} R {m['macro_recall']:.4f} ({secs / 60:.1f} min)")
    return m


def main(argv: list[str] | None = None) -> dict:
    args = parse(argv)
    return run_test(args) if args.predict_test else run_train(args)


if __name__ == "__main__":
    main()
