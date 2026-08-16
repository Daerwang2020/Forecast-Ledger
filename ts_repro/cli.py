"""Command line entry point for TS-Repro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_catalog_config, list_catalog
from .errors import TSReproError
from .example import initialise_example
from .manifest import create_run_directory, seal_directory, verify_directory
from .preflight import inspect_model_binding, preflight_catalog
from .reporting import write_comparison
from .runner import run_experiment
from .visualization import build_viewer


def _common_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, help="Dataset catalog name or YAML path")
    parser.add_argument("--models-dir", action="append", default=[], help="Additional directory containing model YAML files")
    parser.add_argument("--datasets-dir", action="append", default=[], help="Additional directory containing dataset YAML files")
    parser.add_argument("--output-dir", default="experiments", help="New experiment output root")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", choices=("supervised", "zero-shot", "fine-tune"))
    parser.add_argument("--no-seal", action="store_true", help="Leave output writable for debugging; it has no manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsr", description="Manifest-backed fair forecasting evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list", help="List built-in or extra catalog entries")
    listing.add_argument("kind", choices=("models", "datasets"))
    listing.add_argument("--dir", action="append", default=[], help="Additional catalog directory")
    initialise = subparsers.add_parser("init-example", help="Create a deterministic, self-contained quickstart")
    initialise.add_argument("directory")
    run = subparsers.add_parser("run", help="Run one model and create a sealed experiment card")
    run.add_argument("--model", required=True, help="Model catalog name or YAML path")
    _common_run_options(run)
    compare = subparsers.add_parser("compare", help="Run models and seal a summary directory")
    compare.add_argument("models", nargs="+", help="Model catalog names or YAML paths")
    _common_run_options(compare)
    verify = subparsers.add_parser("verify", help="Verify all hashes in a sealed output directory")
    verify.add_argument("directory")
    visualize = subparsers.add_parser("visualize", help="Build a local interactive browser from completed run cards")
    visualize.add_argument("--runs-dir", default="experiments", help="Directory containing completed experiment cards")
    visualize.add_argument("--output-dir", default="viewer", help="Directory for the self-contained HTML viewer")
    visualize.add_argument("--max-points", type=int, default=128, help="Maximum trace points embedded per run")
    doctor = subparsers.add_parser("doctor", help="Read-only readiness check; it never trains or downloads")
    doctor.add_argument("kind", choices=("models", "datasets", "all"), default="all", nargs="?")
    doctor.add_argument("--models-dir", action="append", default=[], help="Additional directory containing model YAML files")
    doctor.add_argument("--datasets-dir", action="append", default=[], help="Additional directory containing dataset YAML files")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable results")
    doctor.add_argument("--bindings", action="store_true", help="For models, inspect checkout, revision, bridge, and checkpoint binding")
    return parser


def _load_pair(args: argparse.Namespace, model_name: str):
    model = load_catalog_config(model_name, "models", args.models_dir)
    dataset = load_catalog_config(args.dataset, "datasets", args.datasets_dir)
    return model, dataset


def _compare(args: argparse.Namespace) -> Path:
    rows = []
    for model_name in args.models:
        model, dataset = _load_pair(args, model_name)
        result = run_experiment(
            model,
            dataset,
            output_dir=args.output_dir,
            seed=args.seed,
            mode_override=args.mode,
            seal=not args.no_seal,
        )
        rows.append(result.comparison_row())
        print(f"completed {result.model_name}: {result.run_dir}")
    if args.no_seal:
        comparison_root = Path(args.output_dir).expanduser().resolve() / "comparisons-unsealed"
        comparison_root.mkdir(parents=True, exist_ok=True)
        comparison_dir = comparison_root / "latest"
        comparison_dir.mkdir(exist_ok=False)
    else:
        comparison_dir = create_run_directory(Path(args.output_dir).expanduser().resolve() / "comparisons", "comparison", args.dataset)
    write_comparison(comparison_dir, rows)
    if not args.no_seal:
        seal_directory(comparison_dir, {"kind": "comparison", "runs": [row["run_directory"] for row in rows]})
    return comparison_dir


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            for entry in list_catalog(args.kind, args.dir):
                priority = f"priority={entry['priority']}" if entry["priority"] else ""
                print("\t".join(item for item in (entry["name"], entry["kind"], priority, f"enabled={entry['enabled']}", entry["path"]) if item))
            return
        if args.command == "init-example":
            print(initialise_example(args.directory))
            return
        if args.command == "verify":
            print(json.dumps(verify_directory(args.directory), indent=2, sort_keys=True))
            return
        if args.command == "visualize":
            if args.max_points < 8:
                raise TSReproError("--max-points must be at least 8")
            print(build_viewer(args.runs_dir, args.output_dir, args.max_points))
            return
        if args.command == "doctor":
            kinds = ("models", "datasets") if args.kind == "all" else (args.kind,)
            results = []
            for kind in kinds:
                extra_dirs = args.models_dir if kind == "models" else args.datasets_dir
                kind_results = preflight_catalog(kind, extra_dirs)
                if args.bindings and kind == "models":
                    for result in kind_results:
                        config_name = result.get("config_path", result["name"])
                        config = load_catalog_config(config_name, kind, args.models_dir)
                        result["binding"] = inspect_model_binding(config)
                results.extend(kind_results)
            if args.json:
                print(json.dumps(results, indent=2, ensure_ascii=False))
            else:
                for result in results:
                    reason = f"\t{result['reason']}" if result["reason"] else ""
                    print(f"{result['kind']}\t{result['name']}\t{result['status']}{reason}")
                ready = sum(result["status"] == "ready" for result in results)
                print(f"summary\t{ready} ready\t{len(results) - ready} blocked")
            return
        if args.command == "run":
            model, dataset = _load_pair(args, args.model)
            result = run_experiment(
                model,
                dataset,
                output_dir=args.output_dir,
                seed=args.seed,
                mode_override=args.mode,
                seal=not args.no_seal,
            )
            print(result.run_dir)
            return
        if args.command == "compare":
            print(_compare(args))
            return
        raise AssertionError(f"Unhandled command: {args.command}")
    except TSReproError as exc:
        print(f"tsr: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
