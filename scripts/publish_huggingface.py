"""Publish the public Forecast Ledger demo Space and optional collection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


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
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True, token=token)
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=space_dir,
        commit_message="Publish Forecast Ledger browser demo",
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
        print(f"Collection updated: https://huggingface.co/collections/{namespace}/{collection.slug}")


if __name__ == "__main__":
    main()
