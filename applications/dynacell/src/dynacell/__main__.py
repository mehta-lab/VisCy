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

# HPC-specific config groups (target, feature_extractor/dynaclr, benchmark eval
# leaves) live outside the Python package so the wheel ships only schema + path-
# free references. Editable installs / repo checkouts expose these through
# hydra.searchpath; wheel installs without the repo simply don't see them, and
# external users provide their own groups via --config-dir.
_EXTERNAL_SEARCHPATHS: tuple[str, ...] = (
    "configs/benchmarks/virtual_staining/_internal",
    "configs/benchmarks/virtual_staining/_internal/shared/eval",
)


def _external_configs_dirs() -> list[Path]:
    """Return existing repo-checkout searchpath roots for Hydra eval groups.

    Walks up from this module until it finds the nearest ``pyproject.toml``
    (the application root in editable installs), then returns every
    configured subpath that exists on disk. Missing paths are silently
    skipped so wheel installs behave the same as repo checkouts where the
    dirs were removed.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return [p for sub in _EXTERNAL_SEARCHPATHS if (p := parent / sub).is_dir()]
    return []


def _inject_external_configs(argv: list[str]) -> list[str]:
    """Inject a hydra.searchpath override so external configs are discoverable.

    Hydra's argparse uses a single ``overrides`` positional with
    ``nargs="*"``, which means the first contiguous run of positional args
    is greedily consumed and any later positional (after a flag like
    ``-c job``) is reported as an unrecognized argument. To keep both
    ``dynacell evaluate -c job leaf=x`` and
    ``dynacell evaluate leaf=x -c job`` working, insert the token
    adjacent to an existing positional override when one is present;
    otherwise append.
    """
    dirs = _external_configs_dirs()
    if not dirs:
        return argv
    token = f"hydra.searchpath=[{','.join(f'file://{d}' for d in dirs)}]"
    for i, arg in enumerate(argv[1:], start=1):
        if not arg.startswith("-") and "=" in arg:
            return argv[:i] + [token] + argv[i:]
    return argv + [token]


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
