"""Frozen sentence embeddings (CPU) from the BASE intfloat/multilingual-e5-small (not the fine-tuned one, so train
embeddings are not fitted to the training labels). Mean pooling over the attention mask, L2-normalised, "query: "
prefix (E5 convention for classification inputs), max length 128. Cached once per split.

    python -m pipeline.cpu_frozen_embeddings
Output: artifacts/multilingual/experiments/cpu_embed/{train,validation,test}.npy (float32, rows in split order)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402

OUT = paths.MULTI / "experiments" / "cpu_embed"
MODEL = "intfloat/multilingual-e5-small"


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(10)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    for split in ("validation", "train", "test"):
        dest = OUT / f"{split}.npy"
        if dest.exists():
            print(split, "cached"); continue
        texts = pd.read_csv(paths.MULTI / "data" / f"{split}.csv", encoding="utf-8-sig",
                            low_memory=False)["cleaned_text"].astype(str).tolist()
        order = np.argsort([len(t) for t in texts])            # length-sorted batches: much less padding
        out = np.zeros((len(texts), model.config.hidden_size), np.float32)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                idx = order[i:i + 64]
                enc = tok(["query: " + texts[j] for j in idx], truncation=True, max_length=128, padding=True,
                          return_tensors="pt")
                h = model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1)
                out[idx] = torch.nn.functional.normalize(e, dim=-1).numpy()
                if i % 6400 == 0:
                    print(f"{split} {i}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
        np.save(dest, out)
        print(f"{split} done {out.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
