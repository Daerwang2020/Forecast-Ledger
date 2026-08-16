# Forecast Ledger documentation

Start here when adding a model, dataset, or result view.

1. [Protocol](protocol.md) — the data split, normalization, and metric contract.
2. [Data suites](data-suites.md) — LTSF datasets and the GIFT-Eval mapping.
3. [Model bindings](model-bindings.md) — pinned checkout, revision, bridge, and checkpoint.
4. [Production bridge architecture](production-bridges.md) — adapter seams and the 22-model panel.
5. [Visualization](visualization.md) — build and use the local evidence browser.
6. [Reference repositories](reference-repositories.md) — what we learned from established projects.
7. [Bridge verification](bridge-verification.md) — interface-gate evidence and its limits.
8. [Release checklist](release-checklist.md) — the quality gates used before publishing.
9. [Project gap audit](project-gap-audit.md) — what was borrowed, closed, and intentionally deferred.
10. [Canonical case study](../examples/README.md) — the five-minute Python and CLI path.
11. [Discoverability](discoverability.md) — repository metadata and release practices that help useful work get found.

## Design principles

- The **Module** is the protocol runner, catalog, bridge, manifest, or viewer.
- An **Interface** is a stable file/CLI contract; an **Implementation** stays in
  the official upstream checkout whenever possible.
- A **Seam** is where TS-Repro hands control to an external model. The seam is
  narrow, logged, and provenance-bearing.
- **Depth** is kept low in the core: the viewer consumes artifacts instead of
  becoming a second experiment tracker.
- **Leverage** comes from reusing upstream APIs and established benchmark
  conventions; **Locality** comes from keeping evidence on the researcher’s
  machine.

## Contribution path

Read [CONTRIBUTING.md](../CONTRIBUTING.md), add a small fixture test first,
then add the model/dataset binding and documentation in the same change.
