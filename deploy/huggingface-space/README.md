---
title: Forecast Ledger Demo
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# Forecast Ledger demo

This Space runs the deterministic five-minute Forecast Ledger case study in a
browser. It creates a toy series locally, applies the reference-only linear
adapter, and returns a sealed run summary with target/prediction traces.

The reference adapter is a protocol sanity check, not a paper baseline. The
production project and all official model bridges live at
https://github.com/Daerwang2020/Forecast-Ledger.
