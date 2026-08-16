"""Deterministic reference adapter for pipeline verification only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import ForecastAdapter
from ..errors import AdapterError


class ReferenceLinearAdapter(ForecastAdapter):
    """Ridge regression from a flattened context to a flattened horizon.

    This intentionally modest adapter exists to exercise the complete protocol
    without pretending to be a publication model or official implementation.
    """

    def __init__(self, config: dict[str, Any], run_dir: Path, seed: int) -> None:
        super().__init__(config, run_dir, seed)
        self.ridge = float(config.get("parameters", {}).get("ridge", 1e-6))
        if self.ridge < 0:
            raise AdapterError("reference_linear ridge must be non-negative")
        self.weights: np.ndarray | None = None

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        protocol: dict[str, Any],
    ) -> None:
        features = train_x.reshape(len(train_x), -1).astype(np.float64)
        targets = train_y.reshape(len(train_y), -1).astype(np.float64)
        design = np.concatenate((features, np.ones((len(features), 1))), axis=1)
        # Solve the augmented least-squares system rather than normal equations:
        # it is materially more stable for overlapping forecast windows.
        if self.ridge:
            penalty = np.sqrt(self.ridge) * np.eye(design.shape[1], dtype=np.float64)
            penalty[-1, -1] = 0.0  # do not regularise the intercept
            design = np.concatenate((design, penalty), axis=0)
            targets = np.concatenate((targets, np.zeros((penalty.shape[0], targets.shape[1]))), axis=0)
        self.weights, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
        checkpoint = self.run_dir / "reference_linear_weights.npz"
        np.savez_compressed(checkpoint, weights=self.weights, ridge=np.asarray([self.ridge]))
        self.logs.append(f"reference_linear fitted {len(train_x)} training windows; checkpoint={checkpoint.name}")

    def predict(self, test_x: np.ndarray, protocol: dict[str, Any]) -> np.ndarray:
        if self.weights is None:
            raise AdapterError("reference_linear.predict called before fit")
        features = test_x.reshape(len(test_x), -1).astype(np.float64)
        design = np.concatenate((features, np.ones((len(features), 1))), axis=1)
        # ``einsum`` avoids a spurious Accelerate/BLAS overflow warning seen on
        # some macOS builds for otherwise small, finite matrices.
        flat = np.einsum("ij,jk->ik", design, self.weights)
        expected = tuple(protocol["prediction_shape"])
        try:
            prediction = flat.reshape(expected)
        except ValueError as exc:
            raise AdapterError(f"Reference prediction cannot be reshaped to {expected}") from exc
        return prediction.astype(np.float64)

    def provenance(self) -> dict[str, Any]:
        payload = super().provenance()
        payload.update(
            {
                "classification": "reference_only",
                "warning": "This is a deterministic pipeline sanity-check adapter, not an official model.",
                "parameters": {"ridge": self.ridge},
                "checkpoint": "reference_linear_weights.npz",
            }
        )
        return payload
