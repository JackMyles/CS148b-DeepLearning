"""
Train byte-level BPE on the TinyStories corpus (HW1 problem train_bpe_tinystories).

From the `hw1/` directory:
  uv run eecs148b_hw1/train_bpe_tinystories.py

Loads with:
  vocab = pickle.load(open(vocab_path, "rb"))   # dict[int, bytes]
  merges = pickle.load(open(merges_path, "rb")) # list[tuple[bytes, bytes]]
"""

import argparse
import pickle
from pathlib import Path
from train_bpe import train_bpe

SPECIAL = "<|endoftext|>"


def main() -> None:
    default_train = Path("data/TinyStoriesV2-GPT4-train.txt")
    default_vocab = Path("artifacts/tinystories_bpe/vocab.pkl")
    default_merges = Path("artifacts/tinystories_bpe/merges.pkl")

    parser = argparse.ArgumentParser(description="Train BPE on TinyStories and pickle vocab and merges separately.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_train,
        help="TinyStories training plaintext (default: data/TinyStoriesV2-GPT4-train.txt).",
    )
    parser.add_argument(
        "--vocab-output",
        type=Path,
        default=default_vocab,
        help="Pickle path for vocab dict (default: artifacts/tinystories_bpe/vocab.pkl).",
    )
    parser.add_argument(
        "--merges-output",
        type=Path,
        default=default_merges,
        help="Pickle path for merges list (default: artifacts/tinystories_bpe/merges.pkl).",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Final vocabulary size (includes specials).")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}\nDownload TinyStories into ./data/ (see README.md).")

    args.vocab_output.parent.mkdir(parents=True, exist_ok=True)
    args.merges_output.parent.mkdir(parents=True, exist_ok=True)

    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=[SPECIAL],
    )

    with args.vocab_output.open("wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with args.merges_output.open("wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {len(vocab)} vocab entries to {args.vocab_output}")
    print(f"Wrote {len(merges)} merges to {args.merges_output}")

    longest = max(vocab.values(), key=len)
    print(f"Longest token: {len(longest)} bytes, repr = {longest!r}")
    try:
        print(f"  UTF-8 decode: {longest.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print("  (not valid UTF-8 as a whole string)")


if __name__ == "__main__":
    main()
