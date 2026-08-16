# Evaluation protocol

TS-Repro makes the surrounding evaluation contract explicit. A dataset config
is the only source of preprocessing and split decisions.

## Dataset contract

A dataset either specifies one source CSV and contiguous split fractions, or
three explicit `paths.train`, `paths.val`, and `paths.test` CSVs. Each CSV must
contain a unique, sortable timestamp column and the declared finite numeric
target columns.

For a single source, rows are sorted by timestamp and split in order. TS-Repro
does not shuffle, fill missing values, resample, deduplicate, or infer target
columns. Any of those decisions must be made upstream and recorded in the
dataset YAML.

## Windows and information boundary

For input length `L` and prediction length `H`, each forecast at target index
`t` is built as:

```text
input  = values[t-L : t]
target = values[t : t+H]
```

Training targets lie entirely in the train partition. Validation and test
targets lie entirely in their own partitions, while their input history may use
only preceding rows. This mirrors real forecasting: past train/validation
observations are available when predicting a later partition; future targets
never enter an input window.

## Normalization

`standard` normalization uses only train rows, independently for each target
column: `(x - train_mean) / train_std`. Zero standard deviations are replaced
by `1.0` and recorded. `none` leaves values untouched. Metrics are always
computed after inverse transformation, in the original data units.

## Metrics

MSE, MAE, RMSE, MAPE, and sMAPE pool every forecast origin, horizon, and target
value. MAPE and sMAPE are percentages. Their denominator is clamped by
`metric_epsilon` (default `1e-8`), which is saved in `config.yaml`.

## Modes

- `supervised`: the adapter receives train and validation windows, then predicts test windows.
- `fine-tune`: the same data boundary as supervised; the adapter may load a documented official checkpoint before adaptation.
- `zero-shot`: the adapter receives no training signal. Its model YAML must pin the official checkpoint.

The mode is evidence metadata, not a license to change a model's internals.
