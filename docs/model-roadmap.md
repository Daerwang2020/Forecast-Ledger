# Model registry and admission rule

| Priority | Models | Role |
| --- | --- | --- |
| P0 | DLinear, PatchTST, iTransformer, TimeMixer, TimesNet, N-HiTS | The mandatory regression panel: simple linear, strong patch Transformer, inverted Transformer, multi-scale MLP, temporal CNN, and non-Transformer hierarchical interpolation. |
| P1 | SparseTSF, FITS, PDF, Pathformer, TimeKAN, xPatch, PatchMLP, Amplifier, DUET, TimeBridge | Recent architectural routes; each enters only after an upstream commit and a bridge test are pinned. |
| P2 | Chronos-2, TimesFM, Moirai-2, TTM, Time-MoE, Timer/Timer-XL | Foundation-model protocol group, evaluated separately for zero-shot and fine-tune information conditions. |

An entry is **not supported** merely because it is named here. Admission requires:

1. official or research-maintained upstream URL and immutable commit;
2. bridge contract test on a public fixture;
3. a sealed run demonstrating no dropped test windows;
4. declared checkpoint identity and leakage statement for pretrained models.

Every P0/P1/P2 name now has a disabled catalog adapter with a pinned upstream
revision and production bridge in [model-bindings.md](model-bindings.md).
PatchMLP is explicitly marked `benchmark_reference` because its paper has no
public author checkout; the DUET/TFB baseline class is not relabeled as official
paper code. Rows remain disabled until the full-series preprocessing, split,
checkpoint, and sealed evidence are present. This keeps the registry complete
without laundering an interface closure into an official result.
