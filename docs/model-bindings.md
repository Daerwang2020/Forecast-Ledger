# P0/P1/P2 model bindings

All rows below have a public upstream URL and a pinned Git revision. On the
current A100 host the checkout root is:

```text
/data/wzq/BDIC/ts-repro-checkouts
```

Set `TS_REPRO_MODEL_ROOT` to that directory and set `TS_REPRO_BRIDGE_ROOT` to
`ts-repro/bridges` (or a copied, reviewed bridge directory). Every row now has
a production bridge that calls its pinned upstream API. A bridge closure is an
interface result; full-series preprocessing and split provenance are still
required before enabling a scientific table.

| Model | Priority | Upstream | Pinned revision | Checkpoint | Bridge state |
| --- | --- | --- | --- | --- | --- |
| DLinear | P0 | `cure-lab/LTSF-Linear` | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | — | official Model API closure |
| PatchTST | P0 | `yuqinie98/PatchTST` | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | — | official Model API closure |
| iTransformer | P0 | `thuml/iTransformer` | `c2426e68ca13f74aaec08045c5c724d8ad328124` | — | official Model API closure |
| TimeMixer | P0 | `kwuking/TimeMixer` | `e24610583b36fdd8c76cc17a8df4e65759a5f460` | — | official Model API closure |
| TimesNet | P0 | `thuml/Time-Series-Library` (TimesNet README points here) | `4e938a1767106324dd753b2a44832bf870a0252e` | — | official Model API closure |
| N-HiTS | P0 | `Nixtla/neuralforecast` | `ad099fce08e1f4e36cdbf89301f69bd3b820fd41` | — | official NeuralForecast API closure |
| SparseTSF | P1 | `lss-1138/SparseTSF` | `b8c2740eecc84d8095ffce49ba5acafe68e53bb8` | — | official Model API closure |
| FITS | P1 | `VEWOXIC/FITS` | `d040bb015b6299da26d879b90dd19c80fb72c160` | — | official Model API closure; CLI entrypoint separately recorded |
| PDF | P1 | `Hank0626/PDF` | `6c654a1f53f036ba4efa7706986370a812a85917` | — | official Model API closure; CLI plotting remains opt-in |
| Pathformer | P1 | `decisionintelligence/pathformer` | `ea85d82932215e171357da47b3bc82d502344758` | — | official Model API closure |
| TimeKAN | P1 | `huangst21/TimeKAN` | `3a7c366a9e8547fd8840c5d27f25ee3e30615e33` | — | official Model API closure |
| xPatch | P1 | `stitsyuk/xPatch` | `d12eecaa11409109582f5e2ffdebcc2cffd47b3e` | — | official Model API closure |
| PatchMLP | P1 | TFB/DUET benchmark baseline | `a0087ed7da48218504237d37165d0cb401c942c3` | — | official benchmark class closure |
| Amplifier | P1 | `aikunyi/amplifier` | `6cc089312254a0eeda7767342f690fd4536a1758` | — | official Model API closure |
| DUET | P1 | `decisionintelligence/DUET` | `dcc6e6780a9138731b64b9b5398a94a1d97033f0` | — | official benchmark class closure |
| TimeBridge | P1 | `Hank0626/TimeBridge` | `0f9a83fbc3e1260c9ddd527c522dff0ce4b9554b` | — | official Model API closure |
| Chronos-2 | P2 | `amazon-science/chronos-forecasting` | `7dc4435706a4454feb79df44ca9f33631f3027bf` | `amazon/chronos-2` | checkpoint + official `predict_df` closure |
| TimesFM | P2 | `google-research/timesfm` | `3dae50b20d7a724981e8ea36cda75578f80dd2dc` | `google/timesfm-2.5-200m-pytorch` | checkpoint + official `forecast` closure |
| Moirai-2 | P2 | `SalesforceAIResearch/uni2ts` | `cfd46d4510ed8896f263116f32928eede05b0a75` | `Salesforce/moirai-2.0-R-small` | checkpoint + official module forward pass |
| TTM | P2 | `ibm-granite/granite-tsfm` | `9739fa59b61bd9f15cbfb06e5dc3dab28c72ee8d` | `ibm-granite/granite-timeseries-ttm-r2` | checkpoint + official model forward pass |
| Time-MoE | P2 | `Time-MoE/Time-MoE` | `915bfda4c78a544d62a2bec6ab22948423059236` | `Maple728/TimeMoE-50M` | official generation pass with Transformers 4.40.1 |
| Timer / Timer-XL | P2 | `thuml/Large-Time-Series-Model` | `1ff8d1afc073182e6d46022069ff32470ab47945` | `thuml/timer-base-84m` | official generation pass with Transformers 4.40.1 |

The disabled state is deliberate. The bridge closures execute the upstream API,
but a scientific run also needs the original-series mapping, official split,
dependency lock, and checkpoint provenance. `tsr doctor models --bindings`
reports checkout, bridge, and checkpoint state separately.
