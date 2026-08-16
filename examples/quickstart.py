"""Create and run the canonical Forecast Ledger example."""

from __future__ import annotations

import argparse
from pathlib import Path

from ts_repro.config import load_catalog_config
from ts_repro.example import initialise_example
from ts_repro.runner import run_experiment
from ts_repro.visualization import build_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical Forecast Ledger case study")
    parser.add_argument("--output-dir", default="/tmp/forecast-ledger-demo")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = initialise_example(Path(args.output_dir).expanduser().resolve())
    model = load_catalog_config("reference-linear", "models", [root / "models"])
    dataset = load_catalog_config("toy", "datasets", [root / "datasets"])
    result = run_experiment(model, dataset, output_dir=root / "experiments", seed=args.seed)
    viewer = build_viewer(root / "experiments", root / "viewer")
    print(f"sealed run: {result.run_dir}")
    print(f"viewer: {viewer}")


if __name__ == "__main__":
    main()
