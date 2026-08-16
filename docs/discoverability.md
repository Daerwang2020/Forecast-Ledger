# Discoverability and release practice

Forecast Ledger should be discoverable because it is useful, not because it
tries to manufacture attention. The strongest open-source projects make the
first screen answer three questions quickly: what problem does this solve, can
I run it now, and what evidence will I get back?

## Repository metadata

Set the GitHub About description to:

> Manifest-backed, local-first reproducibility layer for fair time-series forecasting — official model bridges, sealed run cards, and interactive evidence views.

Use lowercase, hyphenated topics that match the words researchers actually
search for:

```text
time-series
time-series-forecasting
time-series-foundation-models
forecasting
benchmarking
reproducibility
experiment-tracking
machine-learning
deep-learning
pytorch
scientific-computing
research-software
open-source
data-science
python
```

GitHub documents topics as a discovery mechanism and recommends lowercase,
hyphenated labels; the repository can have up to 20 topics. See [Classifying
your repository with topics](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics?apiVersion=2022-11-28).

Also set a social preview image using the same visual language as
`docs/assets/forecast-ledger-teaser.svg`, with the words “Forecast Ledger” and
“sealed evidence for time-series forecasting” large enough to read in a feed.
GitHub’s [repository customization guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository)
covers the About box, social preview, README, citation, license, and community
files.

## The first five minutes

Keep the README order stable:

1. One-sentence promise and teaser figure.
2. The canonical case study with a copy-paste command.
3. The output tree and one viewer screenshot or animated preview.
4. The model/data coverage and explicit boundaries.
5. Installation, contribution, citation, and security links.

The [canonical case study](../examples/README.md) is intentionally small and
offline. It is a good issue-report attachment, workshop demo, and pull-request
acceptance test because it produces ordinary, inspectable files rather than a
hosted service or a leaderboard claim.

## Release and community loop

- Pin the first stable release and keep its example reproducible.
- Add a short release note that shows the viewer output, names the supported
  protocol, and links to the case study.
- Link the repository from papers, project pages, forecasting reading lists,
  and relevant “awesome” collections; ask for a documentation issue or a small
  reproducibility report rather than asking for a star.
- Keep CI, issue forms, pull-request guidance, `CITATION.cff`, and the license
  visible. They lower the cost of trusting and reusing the project.
- Add one screenshot/GIF when the interactive viewer changes materially. The
  static SVGs remain the durable, offline fallback.
- Use canonical terms in prose: “time-series forecasting”, “reproducibility”,
  “official model bridge”, “experiment manifest”, and “evidence viewer”. Avoid
  invented marketing vocabulary that makes search harder.

GitHub’s [discoverability guide](https://docs.github.com/en/get-started/exploring-projects-on-github/discovering-projects-on-github)
describes Explore, search, topics, stars, and following as the normal routes by
which users find projects. This project should earn those signals through a
clear artifact and helpful releases, not through automated engagement or
keyword stuffing.

## Manual GitHub checklist

The repository content is versioned here; the following three metadata edits
still need to be made in the GitHub web UI (or through an authenticated GitHub
API client):

- About description: paste the one-line description above.
- Topics: paste the 15 topics above, then save.
- Social preview: upload a 1280×640 PNG derived from the teaser figure.

The rest of the discoverability surface is already in the repository: a clear
README, a runnable example, docs, CI, contribution and security policy, issue
forms, and citation metadata.

The public [Hugging Face evidence viewer](https://huggingface.co/spaces/Ziqianwwww/forecast-ledger-demo)
now provides the shareable visual entry point; link it from release notes and
community posts instead of sending readers directly to a long setup section.
