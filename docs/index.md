# Forecast Ledger documentation

Start here when adding a model, dataset, or result view.

1. [Canonical case study](../examples/README.md) — the five-minute Python and CLI path.
2. [Protocol](protocol.md) — the data split, normalization, and metric contract.
3. [Visualization](visualization.md) — build and use the local evidence browser.
4. [Data suites](data-suites.md) — LTSF datasets and the GIFT-Eval mapping.
5. [Model bindings](model-bindings.md) — pinned checkout, revision, bridge, and checkpoint.
6. [Production bridge architecture](production-bridges.md) — adapter seams and the 22-model panel.
7. [Bridge verification](bridge-verification.md) — interface-gate evidence and its limits.
8. [Reference repositories](reference-repositories.md) — what we learned from established projects.
9. [Release checklist](release-checklist.md) — the quality gates used before publishing.
10. [Project gap audit](project-gap-audit.md) — what was borrowed, closed, and intentionally deferred.
11. [Discoverability](discoverability.md) — repository metadata and release practices that help useful work get found.
12. [Hugging Face publication](huggingface.md) — publish the public-safe browser demo and collection.

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
