"""
Training loop for the Transformer LM on TinyStories.

Usage (from hw1/):
    uv run eecs148b_hw1/training_together.py          # defaults
    uv run eecs148b_hw1/training_together.py --help    # see all flags
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from eecs148b_hw1.cross_entropy import cross_entropy
from eecs148b_hw1.data_loading import get_batch
from eecs148b_hw1.experiment_log import ExperimentLog
from eecs148b_hw1.transformer_lm import TransformerLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tokens(path: Path) -> np.ndarray:
    """Load a .npy token array via memory-mapped I/O."""
    return np.load(path, mmap_mode="r")


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_steps: int,
) -> float:
    model.eval()
    total = 0.0
    for _ in range(eval_steps):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        total += loss.item()
    model.train()
    return total / eval_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train a Transformer LM on TinyStories.")

    # Paths
    p.add_argument("--train-tokens", type=Path,
                    default=Path("artifacts/tinystories_tokenized/train_tokens.npy"))
    p.add_argument("--val-tokens", type=Path,
                    default=Path("artifacts/tinystories_tokenized/valid_tokens.npy"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))

    # Model
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=2048)

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=1000)

    # Misc
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--run-name", type=str, default="default")
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument("--overfit", action="store_true",
                    help="Fix a single batch and train on it repeatedly (sanity check).")
    p.add_argument("--no-layernorm", action="store_true",
                    help="Remove all LayerNorms (replace with Identity).")
    p.add_argument("--no-pos-emb", action="store_true",
                    help="Remove positional embeddings (NoPE).")

    args = p.parse_args()

    # ---- Data ----
    train_data = load_tokens(args.train_tokens)
    val_data = load_tokens(args.val_tokens)

    # ---- Model ----
    device = args.device
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        no_layernorm=args.no_layernorm,
        no_pos_emb=args.no_pos_emb,
    ).to(device)
    n_params = sum(pa.numel() for pa in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Checkpoint dir ----
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Experiment log ----
    log = ExperimentLog(
        run_name=args.run_name,
        log_dir=args.log_dir,
        config=vars(args),
    )

    # ---- Fixed batch for overfit mode ----
    fixed_batch = None
    if args.overfit:
        fixed_batch = get_batch(train_data, args.batch_size, args.context_length, device)
        print("Overfit mode: training on a single fixed batch.")

    # ---- Training loop ----
    model.train()
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        x, y = fixed_batch if fixed_batch is not None else get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -- Console + log --
        if step == 1 or step % args.log_interval == 0:
            elapsed = time.time() - t0
            train_loss = loss.item()
            ppl = math.exp(train_loss)
            print(
                f"step {step:>6d} | train loss {train_loss:.4f} | "
                f"ppl {ppl:.2f} | time {elapsed:.1f}s"
            )
            log.record(step=step, train_loss=train_loss, train_ppl=ppl, wallclock=elapsed)

        # -- Validation --
        if step % args.eval_interval == 0:
            val_loss = estimate_loss(
                model, val_data, args.batch_size, args.context_length,
                device, args.eval_steps,
            )
            val_ppl = math.exp(val_loss)
            print(f"         >>> val loss {val_loss:.4f} | val ppl {val_ppl:.2f}")
            log.record(step=step, val_loss=val_loss, val_ppl=val_ppl)

        # -- Checkpoint --
        if step % args.save_interval == 0:
            ckpt_path = args.checkpoint_dir / f"step_{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"         saved checkpoint → {ckpt_path}")

    # Final save
    final_path = args.checkpoint_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final checkpoint → {final_path}")

    # Save log (plotting is done separately to avoid overwriting curated plots)
    try:
        log_path = log.save()
        print(f"Saved run log → {log_path}")
    except Exception as e:
        print(f"Warning: could not save log: {e}")


if __name__ == "__main__":
    main()
