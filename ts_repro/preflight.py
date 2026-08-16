"""Read-only readiness checks for the data and model catalogs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .adapters.base import validate_model_config
from .config import load_catalog_config, list_catalog
from .data import prepare_dataset
from .errors import TSReproError
from .runtime import git_revision


def _status(kind: str, name: str, config: dict[str, Any]) -> dict[str, str]:
    """Return a machine-readable readiness result without creating a run."""
    base = {"kind": kind[:-1], "name": name, "config_path": str(config.get("_config_path", ""))}
    try:
        if kind == "datasets":
            prepare_dataset(config)
        else:
            validate_model_config(config)
    except TSReproError as exc:
        return {**base, "status": "blocked", "reason": str(exc)}
    return {**base, "status": "ready", "reason": ""}


def preflight_catalog(
    kind: str,
    extra_dirs: list[str] | None = None,
) -> list[dict[str, str]]:
    """Check every catalog entry, sequentially, without training or downloads."""
    if kind not in {"models", "datasets"}:
        raise ValueError(f"Unsupported catalog type: {kind}")
    results: list[dict[str, str]] = []
    for entry in list_catalog(kind, extra_dirs):
        name = entry["name"]
        try:
            # A catalog label is intentionally independent from its filename
            # (for example, ``chronos-official-template.yaml``).  Load the
            # discovered path directly so the preflight sees every entry.
            config = load_catalog_config(entry["path"], kind, extra_dirs)
            results.append(_status(kind, name, config))
        except TSReproError as exc:
            results.append({"kind": kind[:-1], "name": name, "status": "blocked", "reason": str(exc)})
    return results


def inspect_model_binding(config: dict[str, Any]) -> dict[str, Any]:
    """Inspect external binding files without enabling or running a model."""
    repository_value = os.path.expandvars(str(config.get("repository_dir", "")))
    repository = Path(repository_value).expanduser()
    binding: dict[str, Any] = {
        "repository_dir": str(repository),
        "checkout": "missing",
        "revision": "missing",
        "bridge": "missing",
        "checkpoint": "not_declared",
    }
    if "${" in repository_value or "$" in repository_value:
        binding["checkout"] = "root_unresolved"
    elif repository.is_dir():
        binding["checkout"] = "present"
        observed = git_revision(repository)
        actual = observed.get("revision")
        expected = str(config.get("official_revision", ""))
        binding["observed_revision"] = actual
        binding["revision"] = "match" if actual and expected and (actual.startswith(expected) or expected.startswith(actual)) else "mismatch"
    commands = config.get("commands", {})
    bridge_paths: list[str] = []
    for phase in ("train", "predict"):
        command = commands.get(phase) if isinstance(commands, dict) else None
        if isinstance(command, list):
            for item in command[1:]:
                expanded = os.path.expandvars(str(item))
                if expanded.endswith((".py", ".sh")):
                    bridge_paths.append(expanded)
    if bridge_paths and all(Path(path).expanduser().is_file() for path in bridge_paths):
        binding["bridge"] = "present"
    binding["bridge_paths"] = bridge_paths
    checkpoint = config.get("official_checkpoint")
    if checkpoint:
        checkpoint_path = Path(os.path.expandvars(str(checkpoint))).expanduser()
        binding["checkpoint"] = "local" if checkpoint_path.is_file() else "declared_nonlocal"
        binding["checkpoint_id"] = str(checkpoint)
    return binding
