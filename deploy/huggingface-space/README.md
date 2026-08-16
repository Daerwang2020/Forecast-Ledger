---
title: Forecast Ledger Demo
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
---

# Forecast Ledger demo

This Space serves a pre-generated, self-contained viewer for the deterministic
five-minute Forecast Ledger case study. It creates a toy series locally during
publication, applies the reference-only linear adapter, and embeds the sealed
run summary with target/prediction traces in a static HTML page.

## What to inspect

- **Method contract** — dataset, frozen protocol, adapter seam, predictions,
  and sealed card.
- **Run ledger** — model, dataset, metrics, mode, and evidence state.
- **Trace** — target and prediction values from the test windows.
- **Evidence boundary** — this is a reproducibility demonstration, not a
  leaderboard or a paper result.

## Continue to the project

- [GitHub repository](https://github.com/Daerwang2020/Forecast-Ledger)
- [Five-minute case study](https://github.com/Daerwang2020/Forecast-Ledger/tree/main/examples)
- [Documentation](https://github.com/Daerwang2020/Forecast-Ledger/tree/main/docs)
- [Curated collection](https://huggingface.co/collections/Ziqianwwww/fair-time-series-forecasting-reproduction-6a81415d8bdcaba4b5ba3195)

The reference adapter is a protocol sanity check, not a paper baseline. The
production project and all official model bridges live at
https://github.com/Daerwang2020/Forecast-Ledger.
