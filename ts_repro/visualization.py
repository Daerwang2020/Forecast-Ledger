"""Build a small, dependency-light experiment browser from sealed run artifacts.

The viewer is intentionally a report over evidence that already exists on disk;
it does not rerun models, rank papers, or upload data anywhere.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


_METRICS = ("mse", "mae", "rmse", "mape", "smape")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _csv_row(path: Path) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return next(csv.DictReader(handle), {})
    except (OSError, StopIteration):
        return {}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _series(path: Path, max_points: int) -> dict[str, list[float]]:
    """Return one representative channel, downsampled for a responsive browser."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            prediction = np.asarray(archive["predictions"], dtype=float)
            target = np.asarray(archive["targets"], dtype=float)
    except (OSError, KeyError, ValueError):
        return {"target": [], "prediction": []}
    if prediction.shape != target.shape or prediction.size == 0:
        return {"target": [], "prediction": []}
    # Flatten windows while retaining the first channel. This is a diagnostic
    # trace, not a replacement for the official aggregate metrics.
    target_1d = target[..., 0].reshape(-1)
    prediction_1d = prediction[..., 0].reshape(-1)
    if len(target_1d) > max_points:
        indices = np.linspace(0, len(target_1d) - 1, max_points).round().astype(int)
        target_1d, prediction_1d = target_1d[indices], prediction_1d[indices]
    return {"target": target_1d.tolist(), "prediction": prediction_1d.tolist()}


def collect_runs(runs_dir: str | Path, max_points: int = 128) -> list[dict[str, Any]]:
    """Collect run cards and optional traces below ``runs_dir``.

    Directories without both ``metrics.json`` and ``predictions.npz`` are
    ignored, so incomplete diagnostics cannot accidentally appear as results.
    """
    root = Path(runs_dir).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        prediction_path = run_dir / "predictions.npz"
        if not prediction_path.is_file():
            continue
        metrics_payload = _read_json(metrics_path)
        metrics = metrics_payload.get("metrics", {})
        row = _csv_row(run_dir / "result.csv")
        provenance = _read_json(run_dir / "model_provenance.json")
        try:
            display_path = str(run_dir.relative_to(root))
        except ValueError:
            display_path = run_dir.name
        record: dict[str, Any] = {
            "id": run_dir.name,
            # Keep generated viewers portable and avoid embedding the host's
            # absolute filesystem path in a shareable HTML artifact.
            "path": display_path,
            "model": str(row.get("model") or provenance.get("model") or run_dir.name),
            "dataset": str(row.get("dataset") or "unknown"),
            "mode": str(row.get("mode") or "unknown"),
            "seed": row.get("seed"),
            "sealed": (run_dir / "manifest.json").is_file(),
            "metrics": {key: _number(metrics.get(key, row.get(key))) for key in _METRICS},
            "runtime": {
                "training_time_s": _number(row.get("training_time_s")),
                "inference_time_s": _number(row.get("inference_time_s")),
            },
            "series": _series(prediction_path, max_points),
        }
        records.append(record)
    return records


def build_viewer(runs_dir: str | Path, output_dir: str | Path, max_points: int = 128) -> Path:
    """Write a self-contained static viewer and return its index path."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).resolve().parents[1] / "viewer" / "index.html"
    html = template.read_text(encoding="utf-8")
    html = html.replace("const RUNS = [];", "const RUNS = " + json.dumps(collect_runs(runs_dir, max_points), ensure_ascii=False) + ";")
    index = output / "index.html"
    index.write_text(html, encoding="utf-8")
    return index
