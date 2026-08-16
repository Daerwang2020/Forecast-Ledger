# Real upstream bridge verification

This record is intentionally separate from the 22/22 fixture contract result.
It records whether the pinned upstream checkout or package crossed a minimal
execution boundary on the A100 host (`/data/wzq/BDIC/data`) or in the pinned
local checkpoint runtime.

| Model | Result | Evidence / adjustment |
| --- | --- | --- |
| DLinear | PASS | 1-epoch train + test; NumPy 2 compatibility launcher required |
| PatchTST | PASS | 1-epoch train + test; nested-script import path fixed in launcher |
| iTransformer | PASS | 1-epoch train + test |
| TimeMixer | PASS WITH WORKAROUND | official `label_len=0` setting avoids an upstream target/output length mismatch; standard nonzero label length still fails in validation |
| TimesNet | PASS | official TimesNet README points to `thuml/Time-Series-Library`; pinned `4e938a1…` checkout completed 1-epoch ETTh1 train + test (`mse 0.4784`, `mae 0.4597`) |
| N-HiTS | PASS | NeuralForecast official API fit + predict; force single device/CPU for bridge closure |
| SparseTSF | PASS | 1-epoch train + test; `model_type=linear` is required (the attempted `M` value is invalid) |
| FITS | PASS | the official `FITS` registry is in `exp_main_F.py`, not `exp_main.py`; `run_longExp_F.py` completed 1 epoch + ETTh1 test (`mse 1.0275`, `mae 0.6765`); launcher preserves the ragged `x.npy` artifact after metrics |
| PDF | PASS WITH PLOT SUPPRESSION | 1 epoch + full ETTh1 test completed (`mse 0.7950`, `mae 0.5957`); `TS_REPRO_SKIP_VISUAL=1` disables only per-window PDF rendering, leaving prediction metrics intact |
| Pathformer | PASS | official 3-layer patch list and expert-count settings required |
| TimeKAN | PASS WITH WORKAROUND | `label_len=0` plus official downsampling avoids the same target slicing issue |
| xPatch | PASS | 1-epoch train + test |
| Amplifier | PASS | 1-epoch train + test |
| DUET | PASS | direct model forward returns `[2, 24, 7]` on a tiny tensor |
| TimeBridge | PASS | 1-epoch train + test |
| Chronos-2 | PASS | local checkpoint + official `Chronos2Pipeline.predict_df` returned an 8-step, 13-column forecast |
| TimesFM | PASS | local checkpoint + official `TimesFM_2p5_200M_torch.forecast` returned point `(1, 8)` and quantile `(1, 8, 10)` outputs |
| Moirai-2 | PASS | local checkpoint + official `Moirai2Module` forward returned `(1, 40, 576)` distribution parameters |
| TTM | PASS | local checkpoint + official `TinyTimeMixerForPrediction` returned `(1, 96, 1)` |
| Time-MoE | PASS | local checkpoint + official remote-code `generate()` returned `(1, 72)` under isolated Transformers 4.40.1 |
| Timer / Timer-XL | PASS | corrected `thuml/timer-base-84m` checkpoint + official `generate()` returned `(1, 8)` under isolated Transformers 4.40.1 |

`PASS` here means only that the official code can execute a minimal train or
forward/predict closure. It is not a benchmark claim, and the workarounds are
recorded rather than silently changing the paper configuration.
