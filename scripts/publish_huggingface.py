"""Publish the public Forecast Ledger demo Space and optional collection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

# Allow the script to be invoked as `python scripts/publish_huggingface.py`
# from a source checkout, where Python otherwise puts only `scripts/` on
# sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ts_repro.config import load_catalog_config
from ts_repro.example import initialise_example
from ts_repro.runner import run_experiment
from ts_repro.visualization import build_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the Forecast Ledger Hugging Face Space")
    parser.add_argument("--space-id", default=os.getenv("HF_SPACE_ID"))
    parser.add_argument("--collection-title", default="Fair Time-Series Forecasting Reproduction")
    parser.add_argument("--no-collection", action="store_true")
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) before publishing.")

    api = HfApi(token=token)
    identity = api.whoami()
    namespace = str(identity.get("name") or identity.get("id"))
    if not namespace:
        raise SystemExit("The Hugging Face token did not return an account name.")
    space_id = args.space_id or f"{namespace}/forecast-ledger-demo"
    space_dir = Path(__file__).resolve().parents[1] / "deploy" / "huggingface-space"
    api.create_repo(space_id, repo_type="space", space_sdk="static", exist_ok=True, token=token)
    with tempfile.TemporaryDirectory(prefix="forecast-ledger-space-") as temporary:
        staging = Path(temporary) / "space"
        staging.mkdir()
        shutil.copy2(space_dir / "README.md", staging / "README.md")
        root = initialise_example(Path(temporary) / "case")
        model = load_catalog_config("reference-linear", "models", [root / "models"])
        dataset = load_catalog_config("toy", "datasets", [root / "datasets"])
        run_experiment(model, dataset, output_dir=root / "experiments", seed=2026)
        build_viewer(root / "experiments", staging)
        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=staging,
            commit_message="Publish Forecast Ledger static evidence viewer",
            token=token,
        )
    print(f"Space published: https://huggingface.co/spaces/{space_id}")

    if not args.no_collection:
        collection = api.create_collection(
            args.collection_title,
            namespace=namespace,
            description="Curated resources for fair and reproducible time-series forecasting.",
            exists_ok=True,
            token=token,
        )
        api.add_collection_item(collection.slug, space_id, "space", exists_ok=True, token=token)
        collection_url = getattr(collection, "url", f"https://huggingface.co/collections/{collection.slug}")
        print(f"Collection updated: {collection_url}")


if __name__ == "__main__":
    main()
