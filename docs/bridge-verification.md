# Bridge verification

The interface gate was run on the A100 host on 2026-08-11 with:

```text
TS_REPRO_MODEL_ROOT=/data/wzq/BDIC/ts-repro-checkouts
TS_REPRO_BRIDGE_ROOT=/tmp/ts-repro/bridges
TS_REPRO_TEST_MODE=1
python scripts/verify_model_bridges.py --output-dir /tmp/ts-repro-bridge-verification.U2NHFV
```

Result: **22/22 contract rows passed**. This command is intentionally a
dependency-free fixture gate; `TS_REPRO_TEST_MODE=1` makes the distinction
explicit and prevents a missing optional runtime from becoming a false model
result.

The production bridge was then exercised separately against pinned upstream
checkouts: the 15 classic/benchmark model APIs and N-HiTS completed real
train/predict closures on A100, while Chronos-2, TimesFM, Moirai-2, TTM,
Time-MoE, and Timer completed official checkpoint/API closures in their pinned
local runtimes. These are interface closures, not full-dataset benchmark
claims; the catalog remains disabled until original-series preprocessing and
split provenance are sealed.
