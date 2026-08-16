# Contributing to Forecast Ledger

Small, reviewable contributions are preferred. The project is an evidence
tool, so reproducibility metadata is part of the feature—not an afterthought.

## Before opening a change

```bash
cd ts-repro
python -m pip install -e '.[dev]'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q
```

Do not commit datasets, checkpoints, generated experiment directories, or
macOS `._*` sidecar files.

## Adding a model

1. Add a catalog row with the official repository URL, revision, bridge, and
   checkpoint policy.
2. Keep upstream code in its own checkout; do not paste a reimplementation into
   `ts-repro`.
3. Document the train/predict command or API and the output shape.
4. Add a small contract fixture and run the bridge verification gate.
5. State clearly whether evidence is an interface closure or a full-series
   scientific result.

## Adding a visualization

Prefer a view over existing sealed artifacts. New charts must say which metric
and partition they use, remain usable on a narrow screen, and never imply a
ranking when the underlying protocol is not matched.
