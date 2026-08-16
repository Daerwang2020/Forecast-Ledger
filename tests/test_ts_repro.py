from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

from ts_repro.config import list_catalog, load_catalog_config
from ts_repro.errors import DatasetError, ManifestError
from ts_repro.example import initialise_example
from ts_repro.manifest import create_run_directory, seal_directory, verify_directory
from ts_repro.reporting import write_comparison
from ts_repro.runner import run_experiment
from ts_repro.preflight import preflight_catalog
from ts_repro.visualization import build_viewer, collect_runs


def _configs(tmp_path: Path) -> tuple[dict, dict]:
    values = np.arange(180, dtype=float)
    pd.DataFrame(
        {
            "time": [f"2024-01-{index:03d}" for index in range(len(values))],
            "first": values * 0.1 + np.sin(values / 4),
            "second": np.cos(values / 7),
        }
    ).to_csv(tmp_path / "series.csv", index=False)
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(
        """name: fixture
source:
  path: series.csv
schema:
  time_column: time
  target_columns: [first, second]
splits: {train: 0.6, val: 0.2, test: 0.2}
protocol: {input_length: 12, prediction_length: 4, normalization: standard}
""",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        """name: fixture-reference
kind: reference
adapter: reference_linear
mode: supervised
parameters: {ridge: 0.000001}
""",
        encoding="utf-8",
    )
    return load_catalog_config(str(model_path), "models"), load_catalog_config(str(dataset_path), "datasets")


def test_reference_run_is_sealed_and_verifiable(tmp_path: Path) -> None:
    model, dataset = _configs(tmp_path)
    result = run_experiment(model, dataset, output_dir=tmp_path / "experiments", seed=42)
    expected = {
        "config.yaml",
        "dataset.json",
        "environment.json",
        "metrics.json",
        "runtime.json",
        "commit.txt",
        "stdout.log",
        "result.csv",
        "report.md",
        "predictions.npz",
        "manifest.json",
    }
    assert expected.issubset({path.name for path in result.run_dir.iterdir()})
    assert verify_directory(result.run_dir)["valid"] is True
    with np.load(result.run_dir / "predictions.npz", allow_pickle=False) as archive:
        assert archive["predictions"].shape == archive["targets"].shape
    metrics = json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    assert set(metrics) == {"mse", "mae", "rmse", "mape", "smape"}


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    model, dataset = _configs(tmp_path)
    result = run_experiment(model, dataset, output_dir=tmp_path / "experiments")
    target = result.run_dir / "metrics.json"
    target.chmod(0o644)
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="verification failed"):
        verify_directory(result.run_dir)


def test_command_adapter_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, dataset = _configs(tmp_path)
    fixture = Path(__file__).resolve().parents[1] / "ts_repro" / "testing_adapter.py"
    monkeypatch.setenv("TS_REPRO_FIXTURE_ROOT", str(tmp_path))
    model_path = tmp_path / "command_model.yaml"
    model_path.write_text(
        f"""name: command-fixture
kind: external
adapter: command
mode: supervised
official_repository: https://example.invalid/official-fixture
repository_dir: ${{TS_REPRO_FIXTURE_ROOT}}
official_revision: unversioned-fixture
commands:
  train: [\"{{python}}\", {fixture}, train]
  predict: [\"{{python}}\", {fixture}, predict]
""",
        encoding="utf-8",
    )
    model = load_catalog_config(str(model_path), "models")
    result = run_experiment(model, dataset, output_dir=tmp_path / "experiments", seed=9)
    assert verify_directory(result.run_dir)["valid"] is True
    assert "fixture predict complete" in (result.run_dir / "stdout.log").read_text(encoding="utf-8")
    assert (result.run_dir / "adapter_input" / "bridges" / "predict_1_testing_adapter.py").is_file()


def test_initialise_example_and_catalog_listing(tmp_path: Path) -> None:
    root = initialise_example(tmp_path / "quickstart")
    model = load_catalog_config("reference-linear", "models", [root / "models"])
    dataset = load_catalog_config("toy", "datasets", [root / "datasets"])
    result = run_experiment(model, dataset, output_dir=root / "experiments")
    assert result.metrics["mse"] >= 0
    assert verify_directory(result.run_dir)["artifact_count"] >= 10


def test_visualization_collects_sealed_run_and_builds_static_viewer(tmp_path: Path) -> None:
    model, dataset = _configs(tmp_path)
    result = run_experiment(model, dataset, output_dir=tmp_path / "experiments", seed=17)
    records = collect_runs(tmp_path / "experiments")
    assert len(records) == 1
    assert records[0]["sealed"] is True
    assert records[0]["series"]["target"]
    viewer = build_viewer(tmp_path / "experiments", tmp_path / "viewer", max_points=16)
    html = viewer.read_text(encoding="utf-8")
    assert "fixture-reference" in html
    assert "const RUNS = [" in html
    assert result.run_dir.name in html


def test_visualize_cli_accepts_empty_directory(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-m", "ts_repro", "visualize", "--runs-dir", str(tmp_path / "empty"), "--output-dir", str(tmp_path / "viewer")],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.stdout.strip().endswith("index.html")
    assert (tmp_path / "viewer" / "index.html").is_file()


def test_builtin_catalogs_parse_and_preflight_reports_readiness(tmp_path: Path) -> None:
    assert len(list_catalog("datasets")) == 16
    assert len(list_catalog("models")) == 23
    model, dataset = _configs(tmp_path)
    models = tmp_path / "models"
    datasets = tmp_path / "datasets"
    models.mkdir()
    datasets.mkdir()
    (models / "fixture-reference.yaml").write_text(
        """name: fixture-reference\nkind: reference\nadapter: reference_linear\nmode: supervised\n""",
        encoding="utf-8",
    )
    (datasets / "fixture.yaml").write_text(
        """name: fixture\nsource: {path: ../series.csv}\nschema: {time_column: time, target_columns: [first, second]}\nsplits: {train: 0.6, val: 0.2, test: 0.2}\nprotocol: {input_length: 12, prediction_length: 4, normalization: standard}\n""",
        encoding="utf-8",
    )
    # The fixture files deliberately exercise the same readiness path as `tsr doctor`.
    assert any(item["name"] == model["name"] and item["status"] == "ready" for item in preflight_catalog("models", [str(models)]))
    assert any(item["name"] == dataset["name"] and item["status"] == "ready" for item in preflight_catalog("datasets", [str(datasets)]))


def test_comparison_exports_and_seals(tmp_path: Path) -> None:
    model, dataset = _configs(tmp_path)
    result = run_experiment(model, dataset, output_dir=tmp_path / "experiments")
    comparison = create_run_directory(tmp_path / "comparisons", "comparison", "fixture")
    write_comparison(comparison, [result.comparison_row()])
    seal_directory(comparison, {"kind": "comparison"})
    assert verify_directory(comparison)["valid"] is True
    assert "\\begin{tabular}" in (comparison / "comparison.tex").read_text(encoding="utf-8")


def test_explicit_splits_reject_overlap_that_would_leak_future_rows(tmp_path: Path) -> None:
    for name, start in (("train", 0), ("val", 10), ("test", 40)):
        values = np.arange(start, start + 30)
        pd.DataFrame({"time": values, "target": values.astype(float)}).to_csv(tmp_path / f"{name}.csv", index=False)
    dataset_path = tmp_path / "overlapping.yaml"
    dataset_path.write_text(
        """name: overlapping
paths: {train: train.csv, val: val.csv, test: test.csv}
schema: {time_column: time, target_columns: [target]}
protocol: {input_length: 4, prediction_length: 2, normalization: standard}
""",
        encoding="utf-8",
    )
    config = load_catalog_config(str(dataset_path), "datasets")
    with pytest.raises(DatasetError, match="overlapping or reordered"):
        run_experiment(load_catalog_config("reference-linear", "models"), config, output_dir=tmp_path / "experiments")


def test_npz_and_gift_eval_dataset_adapters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    np.savez_compressed(tmp_path / "pems.npz", data=np.arange(120 * 3 * 2, dtype=float).reshape(120, 3, 2))
    npz_path = tmp_path / "pems.yaml"
    npz_path.write_text(
        """name: pems-fixture
source: {path: pems.npz, format: npz, array_key: data, array_channel: 0}
schema: {time_column: __index__, target_columns: all}
splits: {train: 0.6, val: 0.2, test: 0.2}
protocol: {input_length: 6, prediction_length: 3, normalization: standard, test_drop_last: false}
""", encoding="utf-8")
    result = run_experiment(load_catalog_config("reference-linear", "models"), load_catalog_config(str(npz_path), "datasets"), output_dir=tmp_path / "runs")
    assert result.metrics["mae"] >= 0

    class FakeDataset:
        prediction_length = 3
        training_dataset = [{"target": np.arange(40, dtype=float)}]
        validation_dataset = [{"target": np.arange(50, dtype=float)}]
        test_data = types.SimpleNamespace(input=[{"target": np.arange(20, dtype=float)}], label=[{"target": np.arange(3, dtype=float)}])
        def __init__(self, *args, **kwargs): pass
    gift_module, data_module = types.ModuleType("gift_eval"), types.ModuleType("gift_eval.data")
    data_module.Dataset = FakeDataset
    monkeypatch.setitem(sys.modules, "gift_eval", gift_module)
    monkeypatch.setitem(sys.modules, "gift_eval.data", data_module)
    gift_path = tmp_path / "gift.yaml"
    gift_path.write_text(
        """name: gift-fixture
source: {format: gift_eval, dataset: fixture, term: short}
protocol: {input_length: 6, prediction_length: 3, normalization: standard, test_drop_last: false}
""", encoding="utf-8")
    gift_result = run_experiment(load_catalog_config("reference-linear", "models"), load_catalog_config(str(gift_path), "datasets"), output_dir=tmp_path / "runs")
    assert gift_result.metrics["mse"] >= 0


def test_composite_time_and_exact_duplicate_policy(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "Datum": ["01.01.2020"] * 61,
            "Von": [f"{index // 4:02d}:{(index % 4) * 15:02d}" for index in range(60)] + ["14:45"],
            "MW": np.arange(60, dtype=float).tolist() + [59.0],
        }
    )
    rows.to_csv(tmp_path / "solar.csv", index=False)
    dataset_path = tmp_path / "solar.yaml"
    dataset_path.write_text(
        """name: solar-fixture
source: {path: solar.csv, format: csv, deduplicate_exact: true}
schema: {time_columns: [Datum, Von], target_columns: [MW]}
splits: {train: 0.6, val: 0.2, test: 0.2}
protocol: {input_length: 6, prediction_length: 3, normalization: standard, test_drop_last: false}
""",
        encoding="utf-8",
    )
    result = run_experiment(
        load_catalog_config("reference-linear", "models"),
        load_catalog_config(str(dataset_path), "datasets"),
        output_dir=tmp_path / "runs",
    )
    assert result.metrics["mae"] >= 0
    provenance = json.loads((result.run_dir / "dataset.json").read_text(encoding="utf-8"))
    assert provenance["source_files"]["source"]["deduplicated_rows"] == 1


def test_all_model_official_bridges_contract_in_fixture_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every catalog model exposes the production bridge contract.

    The real upstream implementation is exercised by the A100 verification
    job.  Local tests use the explicit fixture mode so they do not download or
    import 22 optional research stacks.
    """
    monkeypatch.setenv("TS_REPRO_TEST_MODE", "1")
    bridge_root = Path(__file__).resolve().parents[1] / "bridges"
    assert not (bridge_root / "npz_bridge.py").exists()
    selectors = [bridge for bridge in bridge_root.glob("*_bridge.py") if not bridge.name.startswith("._")]
    assert all("npz_bridge" not in bridge.read_text(encoding="utf-8") for bridge in selectors)
    models = [
        "dlinear", "patchtst", "itransformer", "timemixer", "timesnet", "nhits",
        "sparsetsf", "fits", "pdf", "pathformer", "timekan", "xpatch", "patchmlp",
        "amplifier", "duet", "timebridge", "chronos", "timesfm", "moirai2", "ttm",
        "timemoe", "timer",
    ]
    rng = np.random.default_rng(17)
    protocol = {
        "format": "ts-repro-adapter-v1",
        "input_length": 12,
        "prediction_length": 4,
        "channels": 2,
        "prediction_shape": [4, 4, 2],
        "mode": "supervised",
    }
    for model in models:
        input_dir = tmp_path / model / "input"
        output_dir = tmp_path / model / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir()
        protocol_path = tmp_path / model / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        np.savez_compressed(input_dir / "train.npz", x=rng.normal(size=(24, 12, 2)), y=rng.normal(size=(24, 4, 2)))
        np.savez_compressed(input_dir / "test.npz", x=rng.normal(size=(4, 12, 2)))
        environment = {
            **dict(os.environ),
            "TS_REPRO_INPUT_DIR": str(input_dir),
            "TS_REPRO_OUTPUT_DIR": str(output_dir),
            "TS_REPRO_PROTOCOL_PATH": str(protocol_path),
        }
        bridge = bridge_root / f"{model}_bridge.py"
        subprocess.run([sys.executable, str(bridge), "train"], env=environment, check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, str(bridge), "predict"], env=environment, check=True, capture_output=True, text=True)
        with np.load(output_dir / "predictions.npz", allow_pickle=False) as archive:
            assert archive["predictions"].shape == (4, 4, 2)


def test_foundation_model_official_bridges_complete_zero_shot_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TS_REPRO_TEST_MODE", "1")
    bridge_root = Path(__file__).resolve().parents[1] / "bridges"
    for model in ("chronos", "timesfm", "moirai2", "ttm", "timemoe", "timer"):
        input_dir = tmp_path / model / "input"
        output_dir = tmp_path / model / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir()
        protocol_path = tmp_path / model / "protocol.json"
        protocol_path.write_text(json.dumps({
            "format": "ts-repro-adapter-v1", "input_length": 12,
            "prediction_length": 4, "channels": 2,
            "prediction_shape": [4, 4, 2], "mode": "zero-shot",
        }), encoding="utf-8")
        np.savez_compressed(input_dir / "test.npz", x=np.random.default_rng(19).normal(size=(4, 12, 2)))
        environment = {
            **dict(os.environ),
            "TS_REPRO_INPUT_DIR": str(input_dir),
            "TS_REPRO_OUTPUT_DIR": str(output_dir),
            "TS_REPRO_PROTOCOL_PATH": str(protocol_path),
        }
        bridge = bridge_root / f"{model}_bridge.py"
        subprocess.run([sys.executable, str(bridge), "predict"], env=environment, check=True, capture_output=True, text=True)
        with np.load(output_dir / "predictions.npz", allow_pickle=False) as archive:
            assert archive["predictions"].shape == (4, 4, 2)
