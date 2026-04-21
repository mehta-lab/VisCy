"""CLI entry point for the Dynacell application.

Routes Lightning subcommands (fit, predict, test, validate) to
``viscy_utils.cli.main()`` and Hydra subcommands (evaluate, report)
to their respective entry points.

Usage
-----
cd applications/dynacell/configs/examples
uv run dynacell fit -c unetvit3d/fit.yml
uv run dynacell evaluate io.pred_path=... target_name=sec61b
uv run dynacell report results_dirs.ModelA=/path/to/results
"""

import importlib
import sys
from pathlib import Path

_HYDRA_COMMANDS: dict[str, tuple[str, str, str]] = {
    "evaluate": ("dynacell.evaluation.pipeline", "evaluate_model", "eval"),
    "precompute-gt": ("dynacell.evaluation.precompute_cli", "precompute_gt", "eval"),
    "report": ("dynacell.reporting.cli", "generate_report", "report"),
}

# Config-group instances with HPC-specific paths (target/predict_set/feature_extractor
# values, benchmark leaves) live outside the Python package so the wheel ships only
# schema + path-free references. Editable installs / repo checkouts expose these
# through hydra.searchpath; wheel installs without the repo simply don't see them,
# and external users provide their own groups via --config-dir.
_EXTERNAL_CONFIGS_SUBPATH = ("configs", "evaluation")


def _external_configs_dir() -> Path | None:
    """Return the external eval configs dir if it sits next to this checkout."""
    root = Path(__file__).resolve().parent.parent.parent  # applications/dynacell
    candidate = root.joinpath(*_EXTERNAL_CONFIGS_SUBPATH)
    return candidate if candidate.is_dir() else None


def _inject_external_configs(argv: list[str]) -> list[str]:
    """Append a hydra.searchpath override so external configs are discoverable.

    Appended (not prepended) so Hydra's argparse-based CLI doesn't treat the
    override as a positional placed before diagnostic flags like ``-c job``.
    """
    ext = _external_configs_dir()
    if ext is None:
        return argv
    return argv + [f"hydra.searchpath=[file://{ext}]"]


def main_cli():
    """Console script entry point for ``dynacell`` command."""
    if len(sys.argv) >= 2 and sys.argv[1] in _HYDRA_COMMANDS:
        command = sys.argv[1]
        module_path, func_name, extra = _HYDRA_COMMANDS[command]
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # strip subcommand for Hydra
        sys.argv = _inject_external_configs(sys.argv)
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            print(f"Missing dependencies for 'dynacell {command}': {e}\nInstall with: pip install 'dynacell[{extra}]'")
            raise SystemExit(1) from e
        getattr(module, func_name)()
    else:
        from viscy_utils.cli import main

        main()


if __name__ == "__main__":
    main_cli()
