# Official command adapters

The adapter boundary keeps model-specific code outside TS-Repro. It is designed
for repositories that already expose a training/evaluation command and for thin
bridge scripts that live adjacent to the official checkout.

## Required model fields

```yaml
name: patchtst-official
kind: official
enabled: true
adapter: command
mode: supervised
official_repository: https://github.com/yuqinie98/PatchTST
repository_dir: /absolute/path/to/PatchTST
official_revision: <pinned-commit-SHA-or-unambiguous-prefix>
commands:
  train: [python, /absolute/path/to/patchtst_bridge.py, train]
  predict: [python, /absolute/path/to/patchtst_bridge.py, predict]
```

`repository_dir` must already exist. Catalog bindings use
`${TS_REPRO_MODEL_ROOT}/<model>`; the adapter expands this variable at runtime.
TS-Repro never clones, changes, or installs into that repository. If it is a Git
checkout, its observed revision is written to the manifest and must match
`official_revision`; a mismatch is a hard error.

## Exchange files

Before `train`, the runner creates `<run>/adapter_input/train.npz` and
`val.npz`. Before `predict`, it creates `<run>/adapter_input/test.npz`. Each
contains arrays named `x` with shape `[windows, input_length, channels]` and
`y` with shape `[windows, prediction_length, channels]`; `test.npz` contains
`x` and no test labels. `protocol.json` records column order and all dimensions.

The subprocess receives these environment variables and equivalent command
placeholders:

```text
TS_REPRO_INPUT_DIR / {input_dir}
TS_REPRO_OUTPUT_DIR / {output_dir}
TS_REPRO_PROTOCOL_PATH / {protocol_path}
TS_REPRO_RUN_DIR / {run_dir}
TS_REPRO_PHASE / {phase}
TS_REPRO_SEED / {seed}
```

The prediction command must write exactly one usable file:

```text
<output_dir>/predictions.npz  # array key: predictions
```

Its shape must exactly equal `[test_windows, prediction_length, channels]`, and
values must be finite. The runner refuses anything else. The array is in the
normalised space; TS-Repro inverse-transforms it before calculating metrics.

## Minimal bridge outline

```python
inputs = np.load(Path(os.environ["TS_REPRO_INPUT_DIR"]) / "train.npz")
# Convert x/y only at the official CLI boundary; do not edit upstream model code.
# Call the pinned official program with its documented parameters.

test_x = np.load(Path(os.environ["TS_REPRO_INPUT_DIR"]) / "test.npz")["x"]
predictions = official_predict(test_x)
np.savez_compressed(Path(os.environ["TS_REPRO_OUTPUT_DIR"]) / "predictions.npz",
                    predictions=predictions)
```

The bridge itself is recorded as part of the command string and copied into the
experiment input directory when it is a local file. Keep bridge versions under
source control and add a small fixture test before treating a model as ready.

## Current production bridges

`bridges/<model>_bridge.py` is a model selector for the shared production
implementation in `bridges/official_bridge.py`. In production it imports the
pinned upstream `Model` class or official foundation API, performs one real
optimization step where the catalog is supervised, persists the official state,
and writes the official prediction tensor. There is no substitute ridge or
continuation path. The only fixture path is `TS_REPRO_TEST_MODE=1`, reserved for
local contract tests and never used by a scientific run.

For the exact model-family mapping and closure evidence, see
[production-bridges.md](production-bridges.md).

Two upstream entrypoint details are part of the current verification record:
FITS uses `run_longExp_F.py`/`exp_main_F.py`, while the TimesNet repository
directs users to `thuml/Time-Series-Library` for the runnable implementation.
