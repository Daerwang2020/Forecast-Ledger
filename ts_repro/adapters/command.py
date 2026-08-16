"""External-command adapter for unmodified official repositories."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from .base import ForecastAdapter
from ..errors import AdapterError, ConfigError
from ..runtime import git_revision


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CommandAdapter(ForecastAdapter):
    def __init__(self, config: dict[str, Any], run_dir: Path, seed: int) -> None:
        super().__init__(config, run_dir, seed)
        self.input_dir = run_dir / "adapter_input"
        self.output_dir = run_dir / "adapter_output"
        self.input_dir.mkdir(parents=True, exist_ok=False)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.protocol_path = self.input_dir / "protocol.json"
        repository_value = os.path.expandvars(str(config["repository_dir"]))
        self.repository_dir = Path(repository_value).expanduser().resolve()
        if not self.repository_dir.is_dir():
            raise ConfigError(f"Official repository_dir does not exist: {self.repository_dir}")
        observed = git_revision(self.repository_dir)
        expected = str(config.get("official_revision", "unversioned-external-adapter"))
        actual = observed.get("revision")
        if config.get("kind") == "official" and actual is None:
            raise ConfigError(f"Official repository_dir is not a readable Git checkout: {self.repository_dir}")
        if actual is not None and not actual.startswith(expected) and not expected.startswith(actual):
            raise ConfigError(
                f"Official checkout revision mismatch: expected {expected}, observed {actual}. Pin the intended checkout."
            )
        self.observed_repository = observed
        self.bridge_snapshots = self._snapshot_local_command_files()

    def _snapshot_local_command_files(self) -> list[dict[str, str]]:
        """Copy invoked local bridge scripts into the sealed evidence directory."""
        snapshot_dir = self.input_dir / "bridges"
        snapshots: list[dict[str, str]] = []
        for phase in ("train", "predict"):
            for index, item in enumerate(self._command(phase)):
                if index == 0:  # interpreter / executable, not an adapter bridge
                    continue
                candidate = Path(item).expanduser()
                if not candidate.is_absolute():
                    candidate = self.repository_dir / candidate
                if candidate.suffix not in {".py", ".sh"}:
                    continue
                if not candidate.is_file():
                    continue
                if candidate.stat().st_size > 5 * 1024 * 1024:
                    continue
                snapshot_dir.mkdir(exist_ok=True)
                destination = snapshot_dir / f"{phase}_{index}_{candidate.name}"
                if not destination.exists():
                    shutil.copy2(candidate, destination)
                snapshots.append(
                    {
                        "phase": phase,
                        "source": str(candidate.resolve()),
                        "snapshot": str(destination.relative_to(self.run_dir)),
                        "sha256": _sha256(destination),
                    }
                )
        return snapshots

    def _checkpoint_provenance(self) -> dict[str, Any] | None:
        checkpoint = self.config.get("official_checkpoint")
        if checkpoint is None:
            return None
        declared = str(checkpoint)
        path = Path(declared).expanduser()
        if path.is_file():
            return {"declared": declared, "path": str(path.resolve()), "sha256": _sha256(path)}
        return {"declared": declared, "sha256": None, "note": "non-local checkpoint identifier; resolve it to an immutable upstream revision."}

    def _write_protocol(self, protocol: dict[str, Any]) -> None:
        self.protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _command(self, phase: str) -> list[str]:
        raw = self.config.get("commands", {}).get(phase)
        if raw is None:
            return []
        if isinstance(raw, str):
            command = shlex.split(raw)
        elif isinstance(raw, list) and all(isinstance(item, (str, int, float)) for item in raw):
            command = [str(item) for item in raw]
        else:
            raise ConfigError(f"commands.{phase} must be a command string or list")
        placeholders = {
            "python": sys.executable,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "protocol_path": str(self.protocol_path),
            "run_dir": str(self.run_dir),
            "repository_dir": str(self.repository_dir),
            "seed": str(self.seed),
            "phase": phase,
        }
        try:
            expanded: list[str] = []
            for item in command:
                # Catalogs use shell-style ${VAR} paths.  `str.format_map`
                # would otherwise mistake the inner braces for a TS-Repro
                # placeholder before `expandvars` gets a chance to run.
                env_tokens: dict[str, str] = {}

                def protect(match: re.Match[str]) -> str:
                    token = f"__TSR_ENV_{len(env_tokens)}__"
                    env_tokens[token] = match.group(0)
                    return token

                protected = re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", protect, item)
                rendered = protected.format_map(placeholders)
                for token, expression in env_tokens.items():
                    rendered = rendered.replace(token, expression)
                expanded.append(os.path.expandvars(rendered))
            return expanded
        except KeyError as exc:
            raise ConfigError(f"Unknown command placeholder {exc} in commands.{phase}") from exc

    def _run(self, phase: str) -> None:
        command = self._command(phase)
        if not command:
            return
        environment = os.environ.copy()
        environment.update(
            {
                "TS_REPRO_INPUT_DIR": str(self.input_dir),
                "TS_REPRO_OUTPUT_DIR": str(self.output_dir),
                "TS_REPRO_PROTOCOL_PATH": str(self.protocol_path),
                "TS_REPRO_RUN_DIR": str(self.run_dir),
                "TS_REPRO_PHASE": phase,
                "TS_REPRO_SEED": str(self.seed),
            }
        )
        completed = subprocess.run(
            command,
            cwd=self.repository_dir,
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        transcript = (
            f"$ {' '.join(shlex.quote(item) for item in command)}\n"
            f"[phase={phase}; returncode={completed.returncode}]\n"
            f"{completed.stdout}{completed.stderr}"
        )
        self.logs.append(transcript)
        if completed.returncode != 0:
            raise AdapterError(f"Official {phase} command failed with exit code {completed.returncode}.")

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        protocol: dict[str, Any],
    ) -> None:
        self._write_protocol(protocol)
        mode = self.config.get("mode", "supervised")
        if mode == "zero-shot":
            self.logs.append("zero-shot mode: training command skipped")
            return
        np.savez_compressed(self.input_dir / "train.npz", x=train_x, y=train_y)
        np.savez_compressed(self.input_dir / "val.npz", x=val_x, y=val_y)
        self._run("train")

    def predict(self, test_x: np.ndarray, protocol: dict[str, Any]) -> np.ndarray:
        self._write_protocol(protocol)
        np.savez_compressed(self.input_dir / "test.npz", x=test_x)
        prediction_path = self.output_dir / "predictions.npz"
        if prediction_path.exists():
            raise AdapterError("Official adapter output already contains predictions.npz before predict.")
        self._run("predict")
        if not prediction_path.is_file():
            raise AdapterError("Official predict command did not write adapter_output/predictions.npz")
        try:
            with np.load(prediction_path, allow_pickle=False) as archive:
                if "predictions" not in archive.files:
                    raise AdapterError("predictions.npz must contain an array named 'predictions'")
                predictions = np.asarray(archive["predictions"], dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise AdapterError(f"Cannot read official predictions.npz: {exc}") from exc
        expected = tuple(protocol["prediction_shape"])
        if predictions.shape != expected:
            raise AdapterError(f"Official predictions shape {predictions.shape} does not equal required {expected}")
        if not np.isfinite(predictions).all():
            raise AdapterError("Official predictions contain NaN or infinity")
        return predictions

    def provenance(self) -> dict[str, Any]:
        payload = super().provenance()
        payload.update(
            {
                "classification": "official_command_adapter"
                if self.config.get("kind") == "official"
                else "external_command_adapter",
                "official_repository": self.config.get("official_repository"),
                "declared_revision": self.config.get("official_revision"),
                "observed_repository": self.observed_repository,
                "official_checkpoint": self._checkpoint_provenance(),
                "commands": self.config.get("commands"),
                "bridge_snapshots": self.bridge_snapshots,
            }
        )
        return payload
