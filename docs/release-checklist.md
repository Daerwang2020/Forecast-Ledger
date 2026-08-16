# Release checklist

Use this checklist for a tagged release or a substantial bridge change.

- [ ] `python -m pytest` passes in the supported Python matrix.
- [ ] `python -m build` produces both sdist and wheel.
- [ ] The wheel installs into a clean target and `python -m ts_repro --help` works.
- [ ] `tsr init-example` followed by one reference run and `tsr verify` works.
- [ ] `tsr visualize` renders a viewer from a sealed run card.
- [ ] No datasets, model weights, `experiments/`, `._*`, credentials, or local absolute paths are staged.
- [ ] Every official model binding has a repository, revision, bridge/API, and checkpoint policy.
- [ ] `CITATION.cff`, `SECURITY.md`, and the README version/scope are current.
- [ ] Interface closure is not described as a scientific benchmark result.

The release gate is deliberately artifact-based: a green test suite is
necessary, but the sealed run and package smoke checks are what make the
release useful to another researcher.
