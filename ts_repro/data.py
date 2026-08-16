"""Transparent CSV loading, chronological splits, and window construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_base_dir
from .errors import ConfigError, DatasetError


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    target_columns: list[str]
    input_length: int
    prediction_length: int
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    provenance: dict[str, Any]

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std.reshape(1, 1, -1) + self.mean.reshape(1, 1, -1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str, base_dir: Path) -> Path:
    expanded = os.path.expandvars(value)
    if "${" in expanded or "$" in expanded:
        raise DatasetError(f"Unresolved environment variable in dataset path: {value}")
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _required(mapping: dict[str, Any], key: str, label: str) -> Any:
    value = mapping.get(key)
    if value is None:
        raise ConfigError(f"{label} is required")
    return value


def _load_csv(
    path: Path,
    time_column: str,
    target_columns: list[str],
    source: dict[str, Any] | None = None,
    time_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise DatasetError(f"Dataset CSV does not exist: {path}")
    source = source or {}
    data_format = str(source.get("format", "csv"))
    try:
        if data_format == "csv":
            frame = pd.read_csv(path)
        elif data_format == "whitespace":
            frame = pd.read_csv(path, sep=r"\s+", header=None)
            frame.columns = [f"value_{index:04d}" for index in range(frame.shape[1])]
        elif data_format == "npz":
            key = str(source.get("array_key", "data"))
            with np.load(path, allow_pickle=False) as archive:
                if key not in archive.files:
                    raise DatasetError(f"NPZ {path} has no array '{key}'")
                values = np.asarray(archive[key])
            if values.ndim == 3:
                channel = int(source.get("array_channel", 0))
                values = values[:, :, channel]
            if values.ndim != 2:
                raise DatasetError(f"NPZ data must become [time, channels], got {values.shape}")
            frame = pd.DataFrame(values, columns=[f"value_{index:04d}" for index in range(values.shape[1])])
        else:
            raise DatasetError(f"Unsupported source.format '{data_format}'")
    except Exception as exc:  # pandas normalises several parser failures
        raise DatasetError(f"Cannot read CSV {path}: {exc}") from exc
    if time_column == "__index__":
        frame.insert(0, time_column, np.arange(len(frame), dtype=np.int64))
    if time_columns:
        missing_time = [column for column in time_columns if column not in frame.columns]
        if missing_time:
            raise DatasetError(f"{path} is missing declared time columns: {missing_time}")
        try:
            combined = frame[time_columns].astype(str).agg(" ".join, axis=1)
            frame[time_column] = pd.to_datetime(combined, dayfirst=True, errors="raise")
        except Exception as exc:
            raise DatasetError(f"Cannot parse composite timestamp columns {time_columns} in {path}: {exc}") from exc
    if not target_columns:
        target_columns.extend(column for column in frame.columns if column != time_column)
    required = [time_column, *target_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetError(f"{path} is missing declared columns: {missing}")
    if frame.empty:
        raise DatasetError(f"Dataset CSV is empty: {path}")
    if frame[time_column].isna().any():
        raise DatasetError(f"Timestamp column contains missing values: {path}")
    duplicate_mask = frame[time_column].duplicated(keep=False)
    deduplicated_rows = 0
    if duplicate_mask.any():
        if not bool(source.get("deduplicate_exact", False)):
            raise DatasetError(f"Timestamp column must be unique: {path}")
        for timestamp in frame.loc[duplicate_mask, time_column].unique():
            group = frame.loc[frame[time_column] == timestamp]
            if len(group.drop_duplicates()) != 1:
                raise DatasetError(f"Duplicate timestamp has conflicting values: {path}")
        before = len(frame)
        frame = frame.drop_duplicates(subset=[time_column], keep="first")
        deduplicated_rows = before - len(frame)
    frame = frame.loc[:, required].copy()
    try:
        frame = frame.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    except Exception as exc:
        raise DatasetError(f"Timestamp column is not sortable in {path}: {exc}") from exc
    for column in target_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    values = frame[target_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise DatasetError(f"Target columns must be finite (no NaN/inf): {path}")
    info = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "first_timestamp": str(frame[time_column].iloc[0]),
        "last_timestamp": str(frame[time_column].iloc[-1]),
        "format": data_format,
        "deduplicated_rows": deduplicated_rows,
    }
    return frame, info


def _build_windows(history: np.ndarray, segment: np.ndarray, input_length: int, prediction_length: int) -> tuple[np.ndarray, np.ndarray]:
    full = np.concatenate((history, segment), axis=0)
    first_target = len(history)
    final_target = len(full) - prediction_length
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for target_start in range(first_target, final_target + 1):
        input_start = target_start - input_length
        if input_start < 0:
            continue
        xs.append(full[input_start:target_start])
        ys.append(full[target_start : target_start + prediction_length])
    if not xs:
        raise DatasetError(
            "No complete windows for a split. Increase split rows or reduce input_length/prediction_length."
        )
    return np.stack(xs), np.stack(ys)


def _fractional_splits(values: np.ndarray, split: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(split.get("rows"), dict):
        rows = split["rows"]
        try:
            counts = [int(rows[name]) for name in ("train", "val", "test")]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError("splits.rows must declare integer train, val, and test counts") from exc
        if min(counts) <= 0 or sum(counts) > len(values):
            raise DatasetError("splits.rows must be positive and fit within the source data")
        train_end, val_end, test_end = counts[0], counts[0] + counts[1], sum(counts)
        return values[:train_end], values[train_end:val_end], values[val_end:test_end]
    try:
        train_fraction = float(_required(split, "train", "splits.train"))
        val_fraction = float(_required(split, "val", "splits.val"))
        test_fraction = float(_required(split, "test", "splits.test"))
    except (TypeError, ValueError) as exc:
        raise ConfigError("splits.train/val/test must be numeric fractions") from exc
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0, atol=1e-9) or min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ConfigError("splits.train, splits.val, and splits.test must be positive and sum to 1.")
    train_end = int(len(values) * train_fraction)
    val_end = int(len(values) * (train_fraction + val_fraction))
    if min(train_end, val_end - train_end, len(values) - val_end) <= 0:
        raise DatasetError("Configured split leaves an empty partition.")
    return values[:train_end], values[train_end:val_end], values[val_end:]


def _normalise(
    train: np.ndarray, val: np.ndarray, test: np.ndarray, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if method == "none":
        mean = np.zeros(train.shape[1], dtype=np.float64)
        std = np.ones(train.shape[1], dtype=np.float64)
        return train, val, test, mean, std
    if method != "standard":
        raise ConfigError("normalization must be either 'standard' or 'none'.")
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (train - mean) / std, (val - mean) / std, (test - mean) / std, mean, std


def _gift_target(entry: Any) -> np.ndarray:
    values = np.asarray(entry["target"], dtype=np.float64)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim == 2:
        return values.T
    raise DatasetError(f"GIFT-Eval target must be 1D or 2D, got {values.shape}")


def _prepare_gift_eval(config: dict[str, Any], source: dict[str, Any]) -> PreparedDataset:
    """Use GIFT-Eval's own Dataset split object; do not reimplement its splitting."""
    protocol = config.get("protocol")
    if not isinstance(protocol, dict):
        raise ConfigError("GIFT-Eval datasets require a protocol mapping")
    input_length = int(_required(protocol, "input_length", "protocol.input_length"))
    try:
        from gift_eval.data import Dataset  # type: ignore
    except ImportError as exc:
        raise DatasetError("GIFT-Eval support requires the official gift-eval package in this environment.") from exc
    dataset_name = str(_required(source, "dataset", "source.dataset"))
    term = str(source.get("term", "short"))
    storage_env_var = str(source.get("storage_env_var", "GIFT_EVAL"))
    supplied_root = source.get("storage_path")
    if supplied_root is not None:
        os.environ[storage_env_var] = str(supplied_root)
    gift = Dataset(dataset_name, term=term, to_univariate=bool(source.get("to_univariate", False)), storage_env_var=storage_env_var)
    prediction_length = int(getattr(gift, "prediction_length"))
    declared_horizon = protocol.get("prediction_length")
    if declared_horizon is not None and int(declared_horizon) != prediction_length:
        raise DatasetError(f"GIFT-Eval declared horizon {prediction_length} disagrees with protocol prediction_length {declared_horizon}")
    train_series = [_gift_target(entry) for entry in gift.training_dataset]
    val_series = [_gift_target(entry) for entry in gift.validation_dataset]
    inputs = [_gift_target(entry) for entry in gift.test_data.input]
    labels = [_gift_target(entry) for entry in gift.test_data.label]
    if not train_series or not val_series or not inputs or len(inputs) != len(labels):
        raise DatasetError("GIFT-Eval returned empty or misaligned training/validation/test iterables")
    channels = train_series[0].shape[1]
    if any(series.shape[1] != channels for series in [*train_series, *val_series, *inputs, *labels]):
        raise DatasetError("GIFT-Eval entries have inconsistent target dimensionality")
    raw_train = np.concatenate(train_series, axis=0)
    mean = raw_train.mean(axis=0)
    std = np.where(raw_train.std(axis=0) == 0, 1.0, raw_train.std(axis=0))
    normalise = lambda array: (array - mean) / std if protocol.get("normalization", "standard") == "standard" else array
    if protocol.get("normalization", "standard") not in {"standard", "none"}:
        raise ConfigError("normalization must be either 'standard' or 'none'.")
    if protocol.get("normalization", "standard") == "none":
        mean, std = np.zeros(channels), np.ones(channels)
    train_windows = [_build_windows(np.empty((0, channels)), normalise(series), input_length, prediction_length) for series in train_series]
    val_windows = [_build_windows(np.empty((0, channels)), normalise(series), input_length, prediction_length) for series in val_series]
    test_x = np.stack([normalise(series)[-input_length:] for series in inputs])
    test_y = np.stack([normalise(series) for series in labels])
    if test_y.shape[1] != prediction_length or test_x.shape[1] != input_length:
        raise DatasetError("GIFT-Eval test_data does not match the configured context/horizon")
    return PreparedDataset(
        name=str(config["name"]), target_columns=[f"target_{index:04d}" for index in range(channels)],
        input_length=input_length, prediction_length=prediction_length,
        train_x=np.concatenate([item[0] for item in train_windows]), train_y=np.concatenate([item[1] for item in train_windows]),
        val_x=np.concatenate([item[0] for item in val_windows]), val_y=np.concatenate([item[1] for item in val_windows]),
        test_x=test_x, test_y=test_y, mean=mean, std=std,
        provenance={"dataset_name": str(config["name"]), "source": "gift_eval", "gift_dataset": dataset_name,
                    "term": term, "storage_env_var": storage_env_var, "storage_path": os.getenv(storage_env_var),
                    "input_length": input_length, "prediction_length": prediction_length,
                    "target_columns": [f"target_{index:04d}" for index in range(channels)],
                    "windows": {"train": int(sum(len(x[0]) for x in train_windows)), "val": int(sum(len(x[0]) for x in val_windows)), "test": len(test_x)},
                    "normalization": {"method": protocol.get("normalization", "standard"), "fit_partition": "gift_eval.training_dataset", "mean": mean.tolist(), "std": std.tolist()}},
    )


def prepare_dataset(config: dict[str, Any]) -> PreparedDataset:
    name = str(_required(config, "name", "dataset name"))
    base_dir = config_base_dir(config)
    source = config.get("source")
    if isinstance(source, dict) and source.get("format") == "gift_eval":
        return _prepare_gift_eval(config, source)
    schema = config.get("schema")
    if not isinstance(schema, dict):
        raise ConfigError("dataset schema must be a mapping")
    time_columns_raw = schema.get("time_columns")
    time_columns: list[str] | None = None
    if time_columns_raw is not None:
        if not isinstance(time_columns_raw, list) or not time_columns_raw or not all(isinstance(column, str) for column in time_columns_raw):
            raise ConfigError("schema.time_columns must be a non-empty list of strings")
        time_columns = list(time_columns_raw)
        time_column = "__composite_time__"
    else:
        time_column = str(_required(schema, "time_column", "schema.time_column"))
    target_columns_raw = _required(schema, "target_columns", "schema.target_columns")
    if target_columns_raw == "all":
        target_columns: list[str] = []
    elif isinstance(target_columns_raw, list) and target_columns_raw and all(isinstance(column, str) for column in target_columns_raw):
        target_columns = list(target_columns_raw)
    else:
        raise ConfigError("schema.target_columns must be 'all' or a non-empty list of strings")
    protocol = config.get("protocol")
    if not isinstance(protocol, dict):
        raise ConfigError("dataset protocol must be a mapping")
    try:
        input_length = int(_required(protocol, "input_length", "protocol.input_length"))
        prediction_length = int(_required(protocol, "prediction_length", "protocol.prediction_length"))
    except (TypeError, ValueError) as exc:
        raise ConfigError("protocol input_length and prediction_length must be integers") from exc
    if input_length <= 0 or prediction_length <= 0:
        raise ConfigError("protocol input_length and prediction_length must be positive")

    provenance: dict[str, Any] = {
        "dataset_name": name,
        "config_path": config["_config_path"],
        "time_column": time_column,
        "time_columns": time_columns,
        "target_columns": target_columns,
        "input_length": input_length,
        "prediction_length": prediction_length,
    }
    if isinstance(config.get("paths"), dict):
        paths = config["paths"]
        split_frames: list[pd.DataFrame] = []
        source_info: dict[str, Any] = {}
        for split_name in ("train", "val", "test"):
            path_value = _required(paths, split_name, f"paths.{split_name}")
            if not isinstance(path_value, str):
                raise ConfigError(f"paths.{split_name} must be a CSV path string")
            frame, info = _load_csv(_resolve_path(path_value, base_dir), time_column, target_columns, time_columns=time_columns)
            split_frames.append(frame)
            source_info[split_name] = info
        for prior_name, following_name, prior, following in zip(
            ("train", "val"), ("val", "test"), split_frames, split_frames[1:]
        ):
            try:
                ordered = bool(prior[time_column].iloc[-1] < following[time_column].iloc[0])
            except TypeError as exc:
                raise DatasetError(
                    f"Cannot verify chronological boundary between explicit {prior_name} and {following_name} files."
                ) from exc
            if not ordered:
                raise DatasetError(
                    f"Explicit {following_name} split must start strictly after the {prior_name} split ends; "
                    "overlapping or reordered files would leak future observations."
                )
        train, val, test = (frame[target_columns].to_numpy(dtype=np.float64) for frame in split_frames)
        provenance["split_strategy"] = "explicit_files"
        provenance["source_files"] = source_info
    else:
        source = config.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ConfigError("dataset must specify either paths.{train,val,test} or source.path")
        frame, info = _load_csv(_resolve_path(source["path"], base_dir), time_column, target_columns, source, time_columns)
        split = config.get("splits")
        if not isinstance(split, dict):
            raise ConfigError("single-source datasets require a splits mapping")
        train, val, test = _fractional_splits(frame[target_columns].to_numpy(dtype=np.float64), split)
        provenance["split_strategy"] = "contiguous_fractions"
        provenance["split_definition"] = split.get("rows") or {key: float(split[key]) for key in ("train", "val", "test")}
        provenance["source_files"] = {"source": info}

    train_n, val_n, test_n, mean, std = _normalise(
        train, val, test, str(protocol.get("normalization", "standard"))
    )
    train_x, train_y = _build_windows(np.empty((0, train.shape[1])), train_n, input_length, prediction_length)
    val_x, val_y = _build_windows(train_n, val_n, input_length, prediction_length)
    test_x, test_y = _build_windows(np.concatenate((train_n, val_n)), test_n, input_length, prediction_length)
    provenance["rows"] = {"train": len(train), "val": len(val), "test": len(test)}
    provenance["windows"] = {"train": len(train_x), "val": len(val_x), "test": len(test_x)}
    provenance["test_drop_last"] = bool(protocol.get("test_drop_last", False))
    if provenance["test_drop_last"]:
        raise ConfigError("TS-Repro forbids protocol.test_drop_last=true: every eligible test window must be scored.")
    provenance["normalization"] = {
        "method": str(protocol.get("normalization", "standard")),
        "fit_partition": "train",
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    return PreparedDataset(
        name=name,
        target_columns=target_columns,
        input_length=input_length,
        prediction_length=prediction_length,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        mean=mean,
        std=std,
        provenance=json.loads(json.dumps(provenance)),
    )
