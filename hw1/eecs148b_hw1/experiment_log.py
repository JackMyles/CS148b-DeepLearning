"""
Lightweight experiment logger.

Stores per-run config + metrics as JSON-lines (one file per run) and
provides a helper to plot loss curves from saved logs.

Usage:
    from eecs148b_hw1.experiment_log import ExperimentLog

    log = ExperimentLog(run_name="baseline", log_dir="logs", config={...})
    log.record(step=100, train_loss=3.21, train_ppl=24.8, wallclock=12.5)
    log.record(step=500, val_loss=2.95, val_ppl=19.1)
    log.save()                       # flush to disk
    ExperimentLog.plot("logs")       # render curves for every run in the dir
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path


class ExperimentLog:
    def __init__(
        self,
        run_name: str,
        log_dir: str | Path = "logs",
        config: dict | None = None,
    ) -> None:
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.entries: list[dict] = []
        self._start = time.time()

        self._path = self.log_dir / f"{self.run_name}.jsonl"
        safe_config = {k: str(v) if isinstance(v, Path) else v for k, v in (self.config).items()}
        with open(self._path, "w") as f:
            f.write(json.dumps({"_config": safe_config}) + "\n")

    def record(self, **kwargs) -> None:
        """Append one row of metrics and flush to disk immediately."""
        self.entries.append(kwargs)
        with open(self._path, "a") as f:
            f.write(json.dumps(kwargs) + "\n")

    def save(self) -> Path:
        """Rewrite the full log (config + entries). Mostly a no-op now since
        record() already appends, but keeps the file tidy if entries were modified."""
        safe_config = {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()}
        with open(self._path, "w") as f:
            f.write(json.dumps({"_config": safe_config}) + "\n")
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")
        return self._path

    @staticmethod
    def load(path: str | Path) -> tuple[dict, list[dict]]:
        """Load a JSONL log. Returns (config, entries)."""
        config = {}
        entries: list[dict] = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if "_config" in row:
                    config = row["_config"]
                else:
                    entries.append(row)
        return config, entries

    @staticmethod
    def plot(log_dir: str | Path = "logs", out: str | Path | None = None,
             runs: list[str] | None = None) -> None:
        """
        Read .jsonl logs in *log_dir* and plot train/val loss curves.
        If *runs* is given, only include those run names (file stems).
        Saves to *out* (default: <log_dir>/loss_curves.png).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        log_dir = Path(log_dir)
        if out is None:
            out = log_dir / "loss_curves.png"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_loss, ax_ppl = axes

        for path in sorted(log_dir.glob("*.jsonl")):
            if runs is not None and path.stem not in runs:
                continue
            config, entries = ExperimentLog.load(path)
            name = path.stem

            steps_t = [e["step"] for e in entries if "train_loss" in e]
            train_loss = [e["train_loss"] for e in entries if "train_loss" in e]
            steps_v = [e["step"] for e in entries if "val_loss" in e]
            val_loss = [e["val_loss"] for e in entries if "val_loss" in e]

            if train_loss:
                ax_loss.plot(steps_t, train_loss, label=f"{name} train")
            if val_loss:
                ax_loss.plot(steps_v, val_loss, "--", label=f"{name} val")

            if train_loss:
                ax_ppl.plot(steps_t, [math.exp(l) for l in train_loss], label=f"{name} train")
            if val_loss:
                ax_ppl.plot(steps_v, [math.exp(l) for l in val_loss], "--", label=f"{name} val")

        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend(fontsize=7)

        ax_ppl.set_xlabel("Step")
        ax_ppl.set_ylabel("Perplexity")
        ax_ppl.set_title("Perplexity")
        ax_ppl.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved loss curves → {out}")
