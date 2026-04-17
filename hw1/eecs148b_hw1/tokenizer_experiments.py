"""
HW §2.7 tokenizer experiments. Run from repo `hw1/`:

  uv run eecs148b_hw1/tokenizer_experiments.py

Requires trained BPE (`train_bpe_tinystories.py`) and TinyStories files under `data/`.
"""

import random
from pathlib import Path
import numpy as np
from tokenizer import Tokenizer

SPECIAL = "<|endoftext|>"
HW1 = Path(__file__).resolve().parents[1]
DATA = HW1 / "data"
VOCAB = HW1 / "artifacts" / "tinystories_bpe" / "vocab.pkl"
MERGES = HW1 / "artifacts" / "tinystories_bpe" / "merges.pkl"
OUT_DIR = HW1 / "artifacts" / "tinystories_tokenized"
TRAIN_PATH = DATA / "TinyStoriesV2-GPT4-train.txt"
VALID_PATH = DATA / "TinyStoriesV2-GPT4-valid.txt"

def load_tokenizer() -> Tokenizer:
    if not VOCAB.is_file() or not MERGES.is_file():
        raise SystemExit(f"Missing {VOCAB} or {MERGES}. Run train_bpe_tinystories.py first.")
    return Tokenizer.from_files(str(VOCAB), str(MERGES), special_tokens=[SPECIAL])


def task_a(tok: Tokenizer) -> None:
    if not TRAIN_PATH.is_file():
        raise SystemExit(f"Missing {TRAIN_PATH} (see README.md).")

    text = TRAIN_PATH.read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split(SPECIAL) if d.strip()]
    if len(docs) < 10:
        raise SystemExit(f"Expected at least 10 non-empty documents; found {len(docs)}.")

    sample = random.sample(docs, 10)
    joined = SPECIAL.join(sample)
    raw = joined.encode("utf-8")
    ids = tok.encode(joined)

    bytes_per_token = len(raw) / len(ids) # num bytes / num tokens
    print("(a) 10 random documents (seed=0), UTF-8 bytes / token IDs:")
    print(f"    bytes={len(raw)}, tokens={len(ids)}, bytes/token ≈ {bytes_per_token:.3f}")


def task_b(tok: Tokenizer) -> None:
    for path in (TRAIN_PATH, VALID_PATH):
        if not path.is_file():
            raise SystemExit(f"Missing {path} (see README.md).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    max_id = max(tok.vocab)

    if max_id >= 2**16:
        raise SystemExit(f"Max token id {max_id} does not fit uint16.")

    for path, name in ((TRAIN_PATH, "train_tokens.npy"), (VALID_PATH, "valid_tokens.npy")):
        ids = tok.encode(path.read_text(encoding="utf-8"))
        arr = np.asarray(ids, dtype=np.uint16)
        out = OUT_DIR / name
        np.save(out, arr)
        print(f"(b) Wrote {len(arr):,} ids to {out} (uint16, max_id={max_id})")


def main() -> None:
    tok = load_tokenizer()
    task_a(tok)
    task_b(tok)


if __name__ == "__main__":
    main()
