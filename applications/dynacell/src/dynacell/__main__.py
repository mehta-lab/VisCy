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

_HYDRA_COMMANDS: dict[str, tuple[str, str, str]] = {
    "evaluate": ("dynacell.evaluation.pipeline", "evaluate_model", "eval"),
    "report": ("dynacell.reporting.cli", "generate_report", "report"),
}


def main_cli():
    """Console script entry point for ``dynacell`` command."""
    if len(sys.argv) >= 2 and sys.argv[1] in _HYDRA_COMMANDS:
        module_path, func_name, extra = _HYDRA_COMMANDS[sys.argv[1]]
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # strip subcommand for Hydra
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            print(
                f"Missing dependencies for 'dynacell {sys.argv[0]}': {e}\nInstall with: pip install 'dynacell[{extra}]'"
            )
            raise SystemExit(1) from e
        getattr(module, func_name)()
    else:
        from viscy_utils.cli import main

        main()


if __name__ == "__main__":
    main_cli()
