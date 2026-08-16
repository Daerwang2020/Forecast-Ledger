"""Adapter interface. Adapters receive only standardised window arrays."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import AdapterError, ConfigError


class ForecastAdapter(ABC):
    def __init__(self, config: dict[str, Any], run_dir: Path, seed: int) -> None:
        self.config = config
        self.run_dir = run_dir
        self.seed = seed
        self.logs: list[str] = []

    @abstractmethod
    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        protocol: dict[str, Any],
    ) -> None:
        """Fit or prepare a model without observing test labels."""

    @abstractmethod
    def predict(self, test_x: np.ndarray, protocol: dict[str, Any]) -> np.ndarray:
        """Return normalised test predictions of exactly the required shape."""

    def provenance(self) -> dict[str, Any]:
        return {"adapter": self.config.get("adapter"), "logs": self.logs}


def validate_model_config(config: dict[str, Any]) -> None:
    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("model name is required")
    adapter = config.get("adapter")
    if adapter not in {"reference_linear", "command"}:
        raise ConfigError("model adapter must be 'reference_linear' or 'command'")
    mode = config.get("mode", "supervised")
    if mode not in {"supervised", "zero-shot", "fine-tune"}:
        raise ConfigError("model mode must be supervised, zero-shot, or fine-tune")
    if config.get("enabled", True) is False:
        reason = config.get("disabled_reason", "not configured")
        raise ConfigError(f"Model '{name}' is disabled: {reason}")
    if adapter == "command":
        if not isinstance(config.get("repository_dir"), str) or not config["repository_dir"].strip():
            raise ConfigError(f"Command adapter '{name}' must declare repository_dir")
        commands = config.get("commands")
        if not isinstance(commands, dict) or not commands.get("predict"):
            raise ConfigError("Command adapters require commands.predict")
        if mode != "zero-shot" and not commands.get("train"):
            raise ConfigError("Supervised/fine-tune command adapters require commands.train")
    if config.get("kind") == "official":
        for field in ("official_repository", "repository_dir", "official_revision"):
            if not isinstance(config.get(field), str) or not config[field].strip():
                raise ConfigError(f"Official model '{name}' must declare {field}")
        if adapter != "command":
            raise ConfigError("Official models must use the command adapter to preserve source boundaries")
        if mode in {"zero-shot", "fine-tune"} and not isinstance(config.get("official_checkpoint"), str):
            raise ConfigError(f"{mode} official model '{name}' must declare official_checkpoint")


def build_adapter(config: dict[str, Any], run_dir: Path, seed: int) -> ForecastAdapter:
    validate_model_config(config)
    adapter_name = config["adapter"]
    if adapter_name == "reference_linear":
        from .linear import ReferenceLinearAdapter

        return ReferenceLinearAdapter(config, run_dir, seed)
    if adapter_name == "command":
        from .command import CommandAdapter

        return CommandAdapter(config, run_dir, seed)
    raise AdapterError(f"No adapter implementation for {adapter_name}")
