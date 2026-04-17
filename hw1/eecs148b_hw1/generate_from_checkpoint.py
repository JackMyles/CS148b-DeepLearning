"""
Generate text from a saved TransformerLM checkpoint.

Usage (from hw1/):
    uv run eecs148b_hw1/generate_from_checkpoint.py \
        --checkpoint checkpoints/best_run/final.pt \
        --output artifacts/generated_text.txt
"""

import argparse
from pathlib import Path

import torch

from eecs148b_hw1.decoding import decode
from eecs148b_hw1.tokenizer import Tokenizer
from eecs148b_hw1.transformer_lm import TransformerLM

SPECIAL = "<|endoftext|>"
VOCAB_PKL = Path("artifacts/tinystories_bpe/vocab.pkl")
MERGES_PKL = Path("artifacts/tinystories_bpe/merges.pkl")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a trained LM checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/generated_text.txt"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    args = p.parse_args()

    tok = Tokenizer.from_files(str(VOCAB_PKL), str(MERGES_PKL), special_tokens=[SPECIAL])
    eos_id = tok._bytes_to_id[SPECIAL.encode("utf-8")]

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    prompt_ids = tok.encode(args.prompt)
    generated_ids = decode(
        model,
        prompt=prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_id,
    )
    text = tok.decode(generated_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Generated {len(generated_ids)} tokens → {args.output}")
    print("--- generated text ---")
    print(text)
    print("--- end ---")


if __name__ == "__main__":
    main()
