# Data suites

## LTSF regression suite

`TS_REPRO_DATA_ROOT` is the only machine-specific setting. The catalog uses
the BDIC layout `ETT/`, `Electricity/`, `Traffic/`, `Weather/`, `Exchange/`,
`ILI/`, `Solar/`, and `PEMS/`; copy a YAML file and edit only `source.path` if
your existing layout differs. For the current A100 checkout, set
`TS_REPRO_DATA_ROOT=/data/wzq/BDIC/data` (the lowercase `/data/wzq/bdic/data`
checkout has the same layout). Data are never copied into TS-Repro.

ETT uses the canonical 12/4/4-month row boundaries. The other CSV datasets use
chronological 70/10/20 boundaries; Solar reads the numeric `MW` column from
BDIC's `Solar/solar.csv` in source-row order (its vendor time fields contain
non-time markers); Weather explicitly allows the one identical duplicate row
present in the source and records that deduplication. PEMS reads `data[:, :, 0]`
from the common `.npz` representation. Every resolved
CSV/NPZ digest, row count, dimensionality, effective split, and normalization
fit is written to `dataset.json`.

`protocol.test_drop_last` must stay false. TS-Repro does not batch the test
windows internally and rejects configurations that request dropping any final
test batch.

## GIFT-Eval

Install and configure the official project, including `GIFT_EVAL` as its data
root. TS-Repro imports `gift_eval.data.Dataset(name, term, to_univariate,
storage_env_var)` and treats its official split object as the source of truth:

- `training_dataset` supplies supervised training windows;
- `validation_dataset` supplies validation windows;
- `test_data.input` and `test_data.label` supply the paired evaluation origins
  and labels.

The adapter records the GIFT dataset name, term, storage variable, official
prediction length, target dimensionality, and window counts. It never creates
a replacement GIFT split and does not copy GIFT data into an experiment.
