"""Create a self-contained local fixture without downloading data or models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import ConfigError


def initialise_example(directory: str | Path) -> Path:
    root = Path(directory).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ConfigError(f"Refusing to populate non-empty example directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    models_dir = root / "models"
    datasets_dir = root / "datasets"
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    datasets_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(17)
    steps = np.arange(240, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "timestamp": [
                (datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=int(step))).isoformat()
                for step in steps
            ],
            "signal": 0.03 * steps + np.sin(steps / 5.0) + rng.normal(0, 0.02, size=len(steps)),
            "seasonal": np.cos(steps / 9.0) + rng.normal(0, 0.02, size=len(steps)),
        }
    )
    frame.to_csv(data_dir / "toy.csv", index=False)
    (datasets_dir / "toy.yaml").write_text(
        """name: toy
source:
  path: ../data/toy.csv
schema:
  time_column: timestamp
  target_columns: [signal, seasonal]
splits:
  train: 0.60
  val: 0.20
  test: 0.20
protocol:
  input_length: 24
  prediction_length: 6
  normalization: standard
""",
        encoding="utf-8",
    )
    (models_dir / "reference-linear.yaml").write_text(
        """name: reference-linear
kind: reference
enabled: true
adapter: reference_linear
mode: supervised
description: Deterministic protocol sanity check; not an official model or paper baseline.
parameters:
  ridge: 0.000001
""",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        """# TS-Repro quickstart

This directory is generated locally; it contains a deterministic toy CSV and a
reference-only model so the full experiment-manifest flow can be exercised
without network access or a GPU.

```bash
tsr run --model reference-linear --models-dir models --dataset toy --datasets-dir datasets --output-dir experiments
tsr verify experiments/<run-directory>
```
""",
        encoding="utf-8",
    )
    return root
