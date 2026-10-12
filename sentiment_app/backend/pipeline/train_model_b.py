"""Phase 2B: XLM-RoBERTa (xlm-roberta-base) without context, on the prepared multilingual split.

Intended for a CUDA GPU (Colab). Exact settings: max_length 128, 5 epochs, lr 2e-5, weight decay 0.01,
warmup ratio 0.1, evaluation + checkpoint every epoch, best model by validation macro F1, early stopping
patience 2, fp16 only on CUDA, seed 42.

    python train_model_b.py --data-dir DATA --out-dir OUT [--checkpoint-dir CKPT] [--smoke-rows N]

--smoke-rows is only for checking that the code runs; its outputs are not results.
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
from sentiment_core.evaluation import evaluate_and_save  # noqa: E402
from sentiment_core.labels import ID2LABEL, LABEL2ID, MODEL_C_SKIP  # noqa: E402

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
SEED = 42


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--train-batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--smoke-rows", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding, EarlyStoppingCallback,
                              Trainer, TrainingArguments, XLMRobertaTokenizerFast, set_seed)
    import datasets

    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.smoke_rows:
        raise SystemExit("No CUDA GPU: the full multilingual Model B run is GPU-only (it would take days on CPU).")
    set_seed(SEED)
    out = args.out_dir
    if (out / "metrics_model_b.json").exists():
        raise SystemExit(f"{out} already has Model B results; refusing to overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint_dir or (out / "_checkpoints")

    read = lambda s: pd.read_csv(args.data_dir / f"{s}.csv", encoding="utf-8-sig", keep_default_na=False,  # noqa: E731
                                 na_values=[""], low_memory=False)
    train, val, test = read("train"), read("validation"), read("test")
    split_hashes = {s: sha256(args.data_dir / f"{s}.csv") for s in ("train", "validation", "test")}
    if args.smoke_rows:
        train, val, test = (d.groupby("label", group_keys=False).head(args.smoke_rows // 3) for d in (train, val, test))

    tok = XLMRobertaTokenizerFast.from_pretrained(MODEL_NAME)

    def to_ds(df):
        ds = datasets.Dataset.from_pandas(pd.DataFrame({"text": df["cleaned_text"].astype(str),
                                                        "labels": df["label"].astype(int)}), preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LENGTH), batched=True,
                      remove_columns=["text"])

    tr, va, te = to_ds(train), to_ds(val), to_ds(test)

    def compute_metrics(ev):
        logits, labels = ev
        p = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, p),
                "macro_f1": f1_score(labels, p, average="macro", labels=[0, 1, 2], zero_division=0),
                "weighted_f1": f1_score(labels, p, average="weighted", labels=[0, 1, 2], zero_division=0),
                "precision": precision_score(labels, p, average="macro", labels=[0, 1, 2], zero_division=0),
                "recall": recall_score(labels, p, average="macro", labels=[0, 1, 2], zero_division=0)}

    params = inspect.signature(TrainingArguments.__init__).parameters
    kw = dict(output_dir=str(ckpt), num_train_epochs=1 if args.smoke_rows else 5, learning_rate=2e-5,
              weight_decay=0.01, save_strategy="epoch", load_best_model_at_end=True,
              metric_for_best_model="macro_f1", greater_is_better=True,
              per_device_train_batch_size=args.train_batch_size, per_device_eval_batch_size=args.eval_batch_size,
              fp16=use_cuda, seed=SEED, logging_strategy="epoch", save_total_limit=2, report_to="none",
              dataloader_num_workers=0)
    kw["evaluation_strategy" if "evaluation_strategy" in params else "eval_strategy"] = "epoch"
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = 0.1
    else:                                   # transformers v5: a float < 1 in warmup_steps is a ratio
        kw["warmup_steps"] = 0.1
    if "data_seed" in params:
        kw["data_seed"] = SEED

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, id2label=ID2LABEL,
                                                               label2id=LABEL2ID)
    tok_kw = {"processing_class": tok} if "processing_class" in inspect.signature(Trainer.__init__).parameters \
        else {"tokenizer": tok}
    trainer = Trainer(model=model, args=TrainingArguments(**kw), train_dataset=tr, eval_dataset=va,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], **tok_kw)
    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0, 1)
    hist = pd.DataFrame([h for h in trainer.state.log_history if "eval_macro_f1" in h])
    hist.to_csv(out / "validation_history_model_b.csv", index=False)
    best_val = float(trainer.state.best_metric)

    trainer.save_model(str(out / "xlm_roberta_multilingual"))
    tok.save_pretrained(str(out / "tokenizer_multilingual"))
    logits = trainer.predict(te).predictions
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    m = evaluate_and_save(
        tag="model_b", model_name="XLM-RoBERTa (no context, multilingual)", input_col="cleaned_text",
        test_df=test, probs=probs, out_dir=out, best_val_macro_f1=best_val,
        extra={"params": {"model": MODEL_NAME, "max_length": MAX_LENGTH, "epochs_max": kw["num_train_epochs"],
                          "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.1,
                          "early_stopping_patience": 2, "metric_for_best_model": "macro_f1", "seed": SEED,
                          "fp16": use_cuda, "train_batch_size": args.train_batch_size},
               "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
               "train_seconds": train_seconds, "best_checkpoint": trainer.state.best_model_checkpoint,
               "epochs_run": float(trainer.state.epoch), "split_sha256": split_hashes,
               "smoke_test": bool(args.smoke_rows), "model_c": MODEL_C_SKIP})
    if ckpt.exists():
        shutil.rmtree(ckpt, ignore_errors=True)
    print(json.dumps({k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1", "best_validation_macro_f1")}, indent=1))
    return m


if __name__ == "__main__":
    main()
