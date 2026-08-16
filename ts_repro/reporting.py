"""Human- and paper-facing reports derived only from run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ("mse", "mae", "rmse", "mape", "smape")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_result_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _metric_table(metrics: dict[str, float]) -> str:
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for key in METRIC_KEYS:
        lines.append(f"| {key.upper()} | {metrics[key]:.8f} |")
    return "\n".join(lines)


def experiment_report(
    run_dir: Path,
    model: dict[str, Any],
    dataset: dict[str, Any],
    metrics: dict[str, float],
    runtime: dict[str, Any],
    adapter_provenance: dict[str, Any],
    seed: int,
    mode: str,
) -> str:
    kind = str(model.get("kind", "reference"))
    model_source = adapter_provenance.get("official_repository", "built-in reference adapter")
    checkpoint = adapter_provenance.get("official_checkpoint") or "not applicable"
    gpu = runtime.get("peak_cuda_memory_mb")
    gpu_display = "not available" if gpu is None else f"{gpu:.2f} MiB"
    return f"""# Experiment Card

## Identity

| Field | Value |
| --- | --- |
| Run directory | `{run_dir.name}` |
| Model | {model['name']} |
| Classification | {kind} |
| Dataset | {dataset['dataset_name']} |
| Mode | {mode} |
| Seed | {seed} |
| Model source | {model_source} |
| Checkpoint | {checkpoint} |

## Protocol

| Field | Value |
| --- | ---: |
| Input length | {dataset['input_length']} |
| Prediction length | {dataset['prediction_length']} |
| Target columns | {', '.join(dataset['target_columns'])} |
| Train / validation / test windows | {dataset['windows']['train']} / {dataset['windows']['val']} / {dataset['windows']['test']} |
| Normalization | {dataset['normalization']['method']} fit on train rows only |

## Test metrics (original data units)

{_metric_table(metrics)}

## Runtime

| Field | Value |
| --- | ---: |
| Training time | {runtime['training_time_s']:.4f} s |
| Inference time | {runtime['inference_time_s']:.4f} s |
| Peak CUDA allocated | {gpu_display} |

## Reproduction

The complete effective configuration, source/data provenance, environment,
subprocess transcript, predictions, metrics, and file digests are in this
directory. Verify before citing:

```bash
tsr verify {run_dir}
```
"""


def write_comparison(directory: Path, rows: list[dict[str, Any]]) -> None:
    keys = ["model", "dataset", "mode", "seed", *METRIC_KEYS, "training_time_s", "inference_time_s", "run_directory"]
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in keys} for row in rows])
    (directory / "comparison.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# Forecast Ledger comparison",
        "",
        "All rows reference separately sealed experiment directories. This is a summary, not a leaderboard.",
        "",
        "| Model | MSE | MAE | RMSE | MAPE | sMAPE | Run |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    latex = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Model & MSE & MAE & RMSE & MAPE & sMAPE " + r"\\",
        "\\midrule",
    ]
    for row in rows:
        display_row = dict(row)
        display_row["run"] = Path(str(row["run_directory"])).name
        markdown.append(
            "| {model} | {mse:.8f} | {mae:.8f} | {rmse:.8f} | {mape:.4f} | {smape:.4f} | `{run}` |".format(**display_row)
        )
        latex.append(
            ("{model} & {mse:.6f} & {mae:.6f} & {rmse:.6f} & {mape:.3f} & {smape:.3f} " + r"\\").format(**row)
        )
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    (directory / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (directory / "comparison.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
