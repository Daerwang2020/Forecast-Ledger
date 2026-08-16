# Reference repositories

Forecast Ledger borrows conventions, not model source code. These projects
are useful reference points for maintainers and contributors:

| Repository | Reusable lesson | Forecast Ledger boundary |
| --- | --- | --- |
| [Time-Series-Library](https://github.com/thuml/Time-Series-Library) | A broad, coherent collection of forecasting implementations and scripts. | Keep official implementations in their own checkout; call them through a bridge. |
| [TFB](https://github.com/decisionintelligence/TFB) | Explicit benchmark scripts and attention to evaluation differences. | Freeze the protocol and record every split/loader choice in the run card. |
| [ProbTS](https://github.com/microsoft/ProbTS) | One benchmark surface spanning point, probabilistic, and foundation models. | Preserve the identity of the official API instead of silently replacing it. |
| [BasicTS](https://github.com/GestaltCogTeam/BasicTS) | Standardized dataset/model experiment organization. | Reuse coverage and naming ideas without copying a second framework into core. |
| [Aim](https://github.com/aimhubio/aim) | Local experiment browsing and run-oriented visual inspection. | Keep the viewer static and artifact-backed; no service or telemetry is required. |
| [DVC VS Code extension](https://github.com/iterative/vscode-dvc) | Treat data and experiment state as inspectable, versioned artifacts. | Use manifests and provenance files, while leaving storage and Git workflow to the researcher. |
| [Evidently](https://github.com/evidentlyai/evidently) | Reusable evaluation/monitoring views. | Keep scientific metrics defined by the frozen protocol, not by a dashboard default. |

The common thread is separation of concerns: a framework can make execution
convenient while a result ledger preserves what actually happened. Forecast
Ledger makes that separation explicit through the `catalog → protocol →
bridge → manifest → viewer` path.
