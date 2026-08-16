"""Run the catalog's 22 bridge contracts against a tiny local dataset.

This is a dependency-free fixture gate, not a model benchmark. It deliberately
sets ``TS_REPRO_TEST_MODE=1`` and enables catalog rows only in memory. Real
upstream closures use the same script through a pinned model runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ts_repro.config import list_catalog, load_catalog_config
from ts_repro.runner import run_experiment


MODELS = (
    "dlinear-official", "patchtst-official", "itransformer-official",
    "timemixer-official", "timesnet-official", "nhits-official",
    "sparsetsf", "fits", "pdf", "pathformer-official", "timekan", "xpatch",
    "patchmlp", "amplifier", "duet", "timebridge-official",
    "chronos-official", "timesfm-official", "moirai2-official", "ttm-official",
    "timemoe-official", "timer-official",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="bridge-verification")
    args = parser.parse_args()
    os.environ["TS_REPRO_TEST_MODE"] = "1"
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "fixture.csv"
    values = np.arange(240, dtype=np.float64)
    pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=len(values), freq="h"),
        "target_a": np.sin(values / 7.0) + values / 100.0,
        "target_b": np.cos(values / 11.0),
    }).to_csv(fixture, index=False)
    dataset = {
        "name": "bridge-fixture",
        "source": {"path": str(fixture), "format": "csv"},
        "schema": {"time_column": "time", "target_columns": ["target_a", "target_b"]},
        "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
        "protocol": {"input_length": 12, "prediction_length": 4, "normalization": "standard", "test_drop_last": False},
        "_config_path": str(output / "fixture.yaml"),
    }
    catalog_paths = {entry["name"]: entry["path"] for entry in list_catalog("models")}
    rows = []
    for name in MODELS:
        model = load_catalog_config(catalog_paths[name], "models")
        model["enabled"] = True
        result = run_experiment(model, dataset, output_dir=output / "runs", seed=2026)
        rows.append({"model": name, "mode": result.mode, "run_dir": str(result.run_dir), **result.metrics})
        print(f"verified {name}: {result.run_dir}", flush=True)
    (output / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"verified {len(rows)} model bridges; summary={output / 'summary.json'}")


if __name__ == "__main__":
    main()
