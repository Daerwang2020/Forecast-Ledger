# Hugging Face publication

The repository contains a public-safe Gradio Space in
`deploy/huggingface-space/`. It runs only the generated toy case and the
deterministic `reference-linear` sanity adapter. It never uploads local data,
model weights, or experiment artifacts.

## Publish with an API token

Create a Hugging Face user token with permission to create and write to a
Space, then run:

```bash
cd ts-repro
export HF_TOKEN='hf_...'
python -m pip install huggingface_hub
python scripts/publish_huggingface.py
```

The default Space name is `<your-namespace>/forecast-ledger-demo`. To publish
under a different name:

```bash
HF_SPACE_ID=<your-namespace>/<space-name> \
python scripts/publish_huggingface.py
```

The publisher also creates or reuses a collection named **Fair Time-Series
Forecasting Reproduction** and adds the Space to it. Use `--no-collection` if
you only want the Space.

The Hub supports public Dataset Cards, Model Cards, Spaces, and Collections as
discoverable project surfaces. See the [Hub documentation](https://huggingface.co/docs/hub)
and [Collections guide](https://huggingface.co/docs/hub/en/collections).
