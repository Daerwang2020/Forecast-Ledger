"""Creation, sealing, and verification of local experiment manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid
from typing import Any

from .errors import ManifestError


MANIFEST_NAME = "manifest.json"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return cleaned or "unnamed"


def create_run_directory(output_dir: str | Path, model_name: str, dataset_name: str) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ManifestError(f"Experiment output root is not a directory: {root}")
    stem = f"{utc_timestamp()}_{safe_name(model_name)}_{safe_name(dataset_name)}_{uuid.uuid4().hex[:8]}"
    run_dir = root / stem
    try:
        run_dir.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise ManifestError(f"Refusing to overwrite an existing experiment: {run_dir}") from exc
    return run_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_index(directory: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(directory).as_posix()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def _make_read_only(directory: Path) -> None:
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o444)
    directory.chmod(0o555)


def seal_directory(directory: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ManifestError(f"Cannot seal a missing directory: {directory}")
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.exists():
        raise ManifestError(f"Directory already has a manifest and will not be overwritten: {directory}")
    payload = {
        "format": "ts-repro-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_intent": "read-only artifacts plus content hashes; verify before using a result.",
        "metadata": metadata or {},
        "artifacts": artifact_index(directory),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _make_read_only(directory)
    return manifest_path


def verify_directory(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory).resolve()
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ManifestError(f"No {MANIFEST_NAME} in {directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Malformed manifest: {manifest_path}") from exc
    if manifest.get("format") != "ts-repro-manifest-v1":
        raise ManifestError("Unsupported manifest format")
    expected = {item["path"]: item for item in manifest.get("artifacts", [])}
    actual = {item["path"]: item for item in artifact_index(directory)}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    changed = sorted(
        path
        for path in set(expected).intersection(actual)
        if expected[path]["sha256"] != actual[path]["sha256"] or expected[path]["bytes"] != actual[path]["bytes"]
    )
    if missing or unexpected or changed:
        details = {"missing": missing, "unexpected": unexpected, "changed": changed}
        raise ManifestError(f"Manifest verification failed: {json.dumps(details, sort_keys=True)}")
    return {
        "valid": True,
        "directory": str(directory),
        "artifact_count": len(expected),
        "created_at_utc": manifest.get("created_at_utc"),
    }


def is_sealed(directory: Path) -> bool:
    return (directory / MANIFEST_NAME).is_file() and not bool(directory.stat().st_mode & stat.S_IWUSR)
