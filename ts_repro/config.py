"""Strict, small YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError


PACKAGE_ROOT = Path(__file__).resolve().parent
CATALOG_ROOT = PACKAGE_ROOT / "catalog"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration must be a YAML mapping: {path}")
    value = deepcopy(value)
    value["_config_path"] = str(path)
    return value


def dump_yaml(value: dict[str, Any]) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def _candidate_paths(name: str, kind: str, extra_dirs: list[str | Path] | None) -> list[Path]:
    supplied = Path(name).expanduser()
    candidates: list[Path] = []
    if supplied.suffix in {".yaml", ".yml"} or supplied.exists():
        candidates.append(supplied)
    suffixes = (".yaml", ".yml")
    for directory in extra_dirs or []:
        for suffix in suffixes:
            candidates.append(Path(directory).expanduser() / f"{name}{suffix}")
    for suffix in suffixes:
        candidates.append(CATALOG_ROOT / kind / f"{name}{suffix}")
    return candidates


def load_catalog_config(
    name: str,
    kind: str,
    extra_dirs: list[str | Path] | None = None,
) -> dict[str, Any]:
    if kind not in {"models", "datasets"}:
        raise ValueError(f"Unsupported catalog kind: {kind}")
    searched: list[Path] = []
    for candidate in _candidate_paths(name, kind, extra_dirs):
        candidate = candidate.expanduser()
        searched.append(candidate)
        if candidate.is_file():
            return load_yaml(candidate)
    joined = "\n  ".join(str(path) for path in searched)
    raise ConfigError(f"Unknown {kind[:-1]} '{name}'. Looked in:\n  {joined}")


def list_catalog(kind: str, extra_dirs: list[str | Path] | None = None) -> list[dict[str, str]]:
    paths: list[Path] = [CATALOG_ROOT / kind]
    paths.extend(Path(directory).expanduser() for directory in extra_dirs or [])
    entries: dict[str, Path] = {}
    for directory in paths:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.y*ml")):
            if path.name.startswith("._"):
                continue
            entries.setdefault(path.stem, path)
    result: list[dict[str, str]] = []
    for name, path in sorted(entries.items()):
        config = load_yaml(path)
        result.append(
            {
                "name": str(config.get("name", name)),
                "path": str(path),
                "kind": str(config.get("kind", "dataset" if kind == "datasets" else "unknown")),
                "enabled": str(config.get("enabled", True)).lower(),
                "priority": str(config.get("priority", "")),
            }
        )
    return result


def config_base_dir(config: dict[str, Any]) -> Path:
    raw = config.get("_config_path")
    if not raw:
        raise ConfigError("Configuration is missing its source path.")
    return Path(str(raw)).resolve().parent
