"""Small, public-safe Hugging Face Space for the Forecast Ledger case study."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ts_repro.config import load_catalog_config
from ts_repro.example import initialise_example
from ts_repro.manifest import verify_directory
from ts_repro.runner import run_experiment


def run_case(seed: int) -> tuple[object | None, dict[str, object], str]:
    """Run only the generated toy case; no user data or model weights enter the Space."""
    try:
        root = initialise_example(Path(tempfile.mkdtemp(prefix="forecast-ledger-space-")))
        model = load_catalog_config("reference-linear", "models", [root / "models"])
        dataset = load_catalog_config("toy", "datasets", [root / "datasets"])
        result = run_experiment(model, dataset, output_dir=root / "experiments", seed=int(seed))
        verify_directory(result.run_dir)
        with np.load(result.run_dir / "predictions.npz", allow_pickle=False) as archive:
            target = np.asarray(archive["targets"], dtype=float)[..., 0].reshape(-1)
            prediction = np.asarray(archive["predictions"], dtype=float)[..., 0].reshape(-1)
        fig, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        axis.plot(target, color="#17212b", linewidth=1.8, label="target")
        axis.plot(prediction, color="#e87722", linewidth=1.8, linestyle="--", label="prediction")
        axis.set_title("Forecast Ledger · sealed reference run")
        axis.set_xlabel("test-window points")
        axis.set_ylabel("original units")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=2)
        summary = {
            "evidence_state": "valid sealed evidence",
            "model": result.model_name,
            "dataset": result.dataset_name,
            "seed": result.seed,
            "metrics": result.metrics,
            "runtime": result.runtime,
            "artifacts": sorted(path.name for path in result.run_dir.iterdir()),
        }
        return fig, summary, "Run complete. The manifest verified every artifact before this result was shown."
    except Exception as exc:  # pragma: no cover - exercised by the hosted runtime
        return None, {"error": f"{type(exc).__name__}: {exc}"}, "The demo failed before producing evidence."


with gr.Blocks(title="Forecast Ledger Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# Forecast Ledger

**Freeze the protocol. Run the seam. Seal the evidence.**

This browser demo uses only a generated toy series and the deterministic
`reference-linear` sanity adapter. It is designed to make the evidence flow
visible without downloading a checkpoint or uploading user data.
"""
    )
    with gr.Row():
        seed = gr.Slider(0, 9999, value=2026, step=1, label="Seed")
        run = gr.Button("Run sealed example", variant="primary")
    status = gr.Markdown("Click **Run sealed example** to create a fresh run card.")
    with gr.Row():
        plot = gr.Plot(label="Target vs prediction")
        details = gr.JSON(label="Evidence card")
    run.click(run_case, inputs=seed, outputs=[plot, details, status])
    gr.Markdown(
        "Source: [Forecast-Ledger on GitHub](https://github.com/Daerwang2020/Forecast-Ledger) · "
        "The full viewer and official bridges remain local-first."
    )


if __name__ == "__main__":
    demo.launch()
