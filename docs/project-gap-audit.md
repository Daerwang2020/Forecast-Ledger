# Project gap audit

This is the maintenance view of Forecast Ledger, not a model leaderboard. It
records what we borrowed from mature repositories and what we intentionally do
not copy.

| Reference | Strong practice observed | Forecast Ledger status |
| --- | --- | --- |
| [TSLib](https://github.com/thuml/Time-Series-Library) | Unified task entry points, Docker guidance, quick tests, and a visible project structure. | Closed for the core path with `tsr init-example`, CI, wheel smoke, and a docs index; upstream model code remains upstream. |
| [TFB](https://github.com/decisionintelligence/TFB) | Explicit fair-benchmark framing, locked Docker requirements, dataset coverage, scripts, and bilingual onboarding. | Protocol and catalogs are explicit; a full Docker image and bilingual docs remain optional follow-ups. |
| [ProbTS](https://github.com/microsoft/ProbTS) | Config-driven runs, varied horizons, foundation-model checkpoint preparation, and reproducibility scripts. | YAML catalog and checkpoint provenance exist; varied-horizon and probabilistic outputs are still a future interface extension. |
| [BasicTS](https://github.com/GestaltCogTeam/BasicTS) | Three-line quick start, modular taskflow/callback design, device support, and unified logging. | The local-first runner is intentionally smaller; CI and the run viewer now cover the public onboarding gap without adding a service backend. |
| [TIME](https://github.com/zqiao11/TIME) | Task-centric fresh datasets, processed-data conventions, and per-task evaluation. | LTSF and GIFT-Eval mappings are present; a public task registry/download layer is still missing. |
| [Aim](https://github.com/aimhubio/aim) | Queryable run explorer and artifact-oriented visual inspection at scale. | The static viewer is portable and private; comparison ingestion and large-run indexing are the next UX seam, not a reason to add telemetry. |

## What was upgraded now

- Python matrix CI (3.10–3.12) plus package build/install smoke tests.
- Wheel-safe inclusion of the viewer template and catalog data.
- `CITATION.cff`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, and a release checklist.
- Viewer evidence states: `valid`, `invalid`, and `unsealed`; a manifest file is
  no longer treated as trustworthy without hash verification.

## Remaining high-value work

1. Add a small task registry that records frequency, horizon, license, source,
   and split provenance without downloading data into Git.
2. Ingest sealed comparison summaries into the viewer so model × dataset ×
   horizon is queryable while preserving the no-leaderboard policy.
3. Add optional Docker images only for pinned official runtimes; keep the core
   package dependency-light.

These are deliberately sequenced after package/evidence integrity. They add
Leverage without weakening Locality or making the project another model zoo.
