"""Run a pinned upstream script with narrow NumPy compatibility aliases.

The old forecasting repositories in the catalog predate NumPy 2.0 and often
refer to removed aliases such as ``np.Inf``.  This launcher keeps the checkout
unchanged while making that compatibility boundary explicit in the smoke
evidence.  It is not a model implementation or a benchmark runner.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
import types


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--script", required=True)
    args, forwarded = parser.parse_known_args()
    cwd = Path(args.cwd).expanduser().resolve()
    script = (cwd / args.script).resolve()
    if not cwd.is_dir() or not script.is_file():
        raise SystemExit(f"missing upstream cwd/script: {cwd} {script}")
    import numpy as np

    # FITS' official test script saves a list of input windows with one
    # shorter final batch. NumPy 2.x rejects that ragged list instead of
    # materialising an object array, after metrics have already succeeded.
    # Preserve the upstream artifact intent without touching the checkout.
    _numpy_save = np.save

    def _save_ragged_compat(file, arr, *save_args, **save_kwargs):
        try:
            return _numpy_save(file, arr, *save_args, **save_kwargs)
        except ValueError as exc:
            if "inhomogeneous" not in str(exc):
                raise
            save_kwargs.setdefault("allow_pickle", True)
            return _numpy_save(
                file, np.asarray(arr, dtype=object), *save_args, **save_kwargs
            )

    np.save = _save_ragged_compat

    # Time-Series-Library imports the optional M4 archive helper eagerly even
    # for ETTh/other CSV datasets. Keep those unrelated datasets runnable when
    # patool is absent; an actual M4 run still raises a clear ImportError at
    # the extraction boundary instead of pretending the dependency exists.
    try:
        import patoolib  # type: ignore  # noqa: F401
    except ImportError:
        patoolib = types.ModuleType("patoolib")

        def _missing_patool(*_args, **_kwargs):
            raise ImportError("patool is required for M4 archive extraction")

        patoolib.extract_archive = _missing_patool  # type: ignore[attr-defined]
        sys.modules["patoolib"] = patoolib

    aliases = {
        "Inf": "inf", "float": "float64", "int": "int64",
        "bool": "bool_", "complex": "complex128", "object": "object_",
    }
    for old, current in aliases.items():
        # ``hasattr`` triggers NumPy's deprecation warning for aliases such
        # as ``np.bool``; inspect the module dictionary instead.
        if old not in np.__dict__:
            setattr(np, old, getattr(np, current))
    os.chdir(cwd)
    # A number of official repositories keep their runnable script one level
    # below the checkout root and rely on that directory being on sys.path.
    sys.path.insert(0, str(script.parent))
    if os.getenv("TS_REPRO_SKIP_VISUAL") == "1":
        # PDF's upstream test loop renders one PDF per test window (2,857
        # files on ETTh1) and otherwise turns a model smoke test into a
        # plotting benchmark. Keep metrics and predictions, skip only plots.
        try:
            import utils.tools as _upstream_tools

            _upstream_tools.visual = lambda *_args, **_kwargs: None
        except ImportError:
            pass
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
