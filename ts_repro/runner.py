"""The end-to-end protocol runner and comparison coordinator."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .adapters import build_adapter
from .config import dump_yaml
from .data import PreparedDataset, prepare_dataset
from .errors import AdapterError, ConfigError
from .manifest import create_run_directory, seal_directory
from .metrics import point_metrics
from .reporting import experiment_report, write_json, write_result_csv
from .runtime import environment_snapshot, git_revision, peak_cuda_memory_mb


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    metrics: dict[str, float]
    runtime: dict[str, float | None]
    model_name: str
    dataset_name: str
    mode: str
    seed: int

    def comparison_row(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "mode": self.mode,
            "seed": self.seed,
            **self.metrics,
            **self.runtime,
            "run_directory": str(self.run_dir),
        }


def set_seed(seed: int) -> None:
    if seed < 0:
        raise ConfigError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _effective_config(model: dict[str, Any], dataset: dict[str, Any], seed: int, mode: str) -> dict[str, Any]:
    return {
        "ts_repro_format": "v0.1",
        "seed": seed,
        "mode": mode,
        "metric_epsilon": 1e-8,
        "model": {key: value for key, value in model.items() if key != "_config_path"},
        "dataset": {key: value for key, value in dataset.items() if key != "_config_path"},
    }


def _adapter_protocol(data: PreparedDataset, seed: int, mode: str) -> dict[str, Any]:
    return {
        "format": "ts-repro-adapter-v1",
        "seed": seed,
        "mode": mode,
        "target_columns": data.target_columns,
        "input_length": data.input_length,
        "prediction_length": data.prediction_length,
        "channels": len(data.target_columns),
        "prediction_shape": list(data.test_y.shape),
        "normalization": data.provenance["normalization"],
        "input_contract": {
            "train": list(data.train_x.shape),
            "val": list(data.val_x.shape),
            "test": list(data.test_x.shape),
            "target": "normalised values; test labels are intentionally withheld",
        },
    }


def _project_revision() -> dict[str, Any]:
    return git_revision(Path(__file__).resolve().parents[1])


def run_experiment(
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    output_dir: str | Path = "experiments",
    seed: int = 2026,
    mode_override: str | None = None,
    seal: bool = True,
) -> RunResult:
    """Run one model under one prepared protocol; never overwrites an existing run."""
    mode = mode_override or str(model_config.get("mode", "supervised"))
    if mode not in {"supervised", "zero-shot", "fine-tune"}:
        raise ConfigError("mode must be supervised, zero-shot, or fine-tune")
    set_seed(seed)
    data = prepare_dataset(dataset_config)
    effective_model_config = dict(model_config)
    effective_model_config["mode"] = mode
    run_dir = create_run_directory(output_dir, str(effective_model_config.get("name", "model")), data.name)
    stdout_path = run_dir / "stdout.log"
    try:
        effective = _effective_config(effective_model_config, dataset_config, seed, mode)
        (run_dir / "config.yaml").write_text(dump_yaml(effective), encoding="utf-8")
        write_json(run_dir / "dataset.json", data.provenance)
        write_json(run_dir / "environment.json", environment_snapshot())
        (run_dir / "commit.txt").write_text(json.dumps(_project_revision(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        protocol = _adapter_protocol(data, seed, mode)
        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as transcript, redirect_stdout(transcript), redirect_stderr(transcript):
            adapter = build_adapter(effective_model_config, run_dir, seed)
            adapter.fit(data.train_x, data.train_y, data.val_x, data.val_y, protocol)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with stdout_path.open("a", encoding="utf-8") as transcript, redirect_stdout(transcript), redirect_stderr(transcript):
            predictions_normalised = adapter.predict(data.test_x, protocol)
        inference_seconds = time.perf_counter() - started
        expected = data.test_y.shape
        if predictions_normalised.shape != expected:
            raise AdapterError(f"Adapter returned {predictions_normalised.shape}; expected {expected}")
        if not np.isfinite(predictions_normalised).all():
            raise AdapterError("Adapter returned non-finite predictions")
        predictions = data.inverse_transform(predictions_normalised)
        targets = data.inverse_transform(data.test_y)
        np.savez_compressed(run_dir / "predictions.npz", predictions=predictions, targets=targets)
        metrics = point_metrics(targets, predictions, epsilon=effective["metric_epsilon"])
        runtime: dict[str, float | None] = {
            "training_time_s": float(train_seconds),
            "inference_time_s": float(inference_seconds),
            "peak_cuda_memory_mb": peak_cuda_memory_mb(),
        }
        adapter_provenance = adapter.provenance()
        if adapter_provenance.get("logs"):
            with stdout_path.open("a", encoding="utf-8") as transcript:
                transcript.write("\n--- Adapter subprocess transcript ---\n")
                transcript.write("\n".join(str(line) for line in adapter_provenance["logs"]))
                transcript.write("\n")
        metric_payload = {
            "metrics": metrics,
            "metric_units": {"mse": "squared original units", "mae": "original units", "rmse": "original units", "mape": "percent", "smape": "percent"},
            "prediction_shape": list(predictions.shape),
            "evaluation_partition": "test",
        }
        write_json(run_dir / "metrics.json", metric_payload)
        write_json(run_dir / "runtime.json", runtime)
        write_json(run_dir / "model_provenance.json", adapter_provenance)
        write_result_csv(
            run_dir / "result.csv",
            {
                "model": str(effective_model_config["name"]),
                "dataset": data.name,
                "mode": mode,
                "seed": seed,
                **metrics,
                **runtime,
            },
        )
        (run_dir / "report.md").write_text(
            experiment_report(run_dir, effective_model_config, data.provenance, metrics, runtime, adapter_provenance, seed, mode),
            encoding="utf-8",
        )
        if seal:
            seal_directory(
                run_dir,
                {
                    "kind": "experiment",
                    "model": str(effective_model_config["name"]),
                    "dataset": data.name,
                    "mode": mode,
                    "seed": seed,
                },
            )
        return RunResult(run_dir, metrics, runtime, str(effective_model_config["name"]), data.name, mode, seed)
    except Exception:
        # Preserve the incomplete directory for diagnosis. It is deliberately not sealed or deleted.
        with stdout_path.open("a", encoding="utf-8") as transcript:
            transcript.write("\nTS-Repro run did not complete; this directory is unsealed diagnostic evidence.\n")
        raise
