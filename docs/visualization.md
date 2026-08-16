# Evidence visualization

Forecast Ledger includes a static, local-first browser for completed run cards.
It is intentionally an evidence view, not a leaderboard or a new metric
implementation.

```bash
cd ts-repro
tsr visualize --runs-dir experiments --output-dir viewer
python -m http.server 8000 --directory viewer
```

Open `http://localhost:8000`. The browser provides:

- a compact method diagram from dataset and frozen protocol to official
  adapter, predictions, and sealed card;
- model/dataset/metric filters over completed runs;
- a run table that distinguishes sealed evidence from unsealed diagnostics;
- an inspectable target-vs-prediction trace for the selected run.

The trace is downsampled only for display. Metrics remain those stored in
`metrics.json`, and the browser never changes or uploads the underlying run.
Runs without both `metrics.json` and `predictions.npz` are omitted. This keeps
failed or partial executions visible in the filesystem for diagnosis without
mistaking them for results.

For a custom integration, `ts_repro.visualization.collect_runs()` returns the
same JSON-friendly records used by the HTML viewer.
