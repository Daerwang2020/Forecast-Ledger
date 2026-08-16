"""Runtime and source provenance snapshots for experiment cards."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def git_revision(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    command = ["git", "-C", str(path), "rev-parse", "HEAD"]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return {"path": str(path), "revision": None, "available": False}
    revision = completed.stdout.strip() if completed.returncode == 0 else None
    return {"path": str(path), "revision": revision, "available": revision is not None}


def package_versions(packages: tuple[str, ...] = ("numpy", "pandas", "PyYAML", "torch")) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(),
    }
    try:
        import torch  # type: ignore

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except ImportError:
        payload["torch"] = {"installed": False}
    return payload


def peak_cuda_memory_mb() -> float | None:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    except ImportError:
        return None
