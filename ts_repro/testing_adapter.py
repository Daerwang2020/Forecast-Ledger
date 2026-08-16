"""A deterministic subprocess fixture used only by the test suite.

It validates the command-adapter exchange contract without claiming to be an
official forecasting model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "predict"))
    args = parser.parse_args()
    input_dir = Path(os.environ["TS_REPRO_INPUT_DIR"])
    output_dir = Path(os.environ["TS_REPRO_OUTPUT_DIR"])
    if args.phase == "train":
        with np.load(input_dir / "train.npz", allow_pickle=False) as archive:
            assert archive["x"].ndim == 3 and archive["y"].ndim == 3
        print("fixture train complete")
        return
    with np.load(input_dir / "test.npz", allow_pickle=False) as archive:
        x = archive["x"]
    protocol = __import__("json").loads((input_dir / "protocol.json").read_text(encoding="utf-8"))
    horizon = int(protocol["prediction_length"])
    predictions = np.repeat(x[:, -1:, :], horizon, axis=1)
    np.savez_compressed(output_dir / "predictions.npz", predictions=predictions)
    print("fixture predict complete")


if __name__ == "__main__":
    main()
