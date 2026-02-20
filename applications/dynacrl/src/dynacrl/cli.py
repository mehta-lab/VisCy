"""Click-based CLI for DynaCLR evaluation and analysis tools."""

import importlib
import sys
from pathlib import Path

import click


class LazyCommand(click.Command):
    """Lazy-load command to improve startup time.

    Defers module import until invocation. If the import fails (e.g. missing
    optional dependencies), ``--help`` still works but shows only the
    short_help description.
    """

    def __init__(self, name, import_path, help=None, short_help=None):
        self.import_path = import_path
        self._real_command = None
        super().__init__(name=name, help=help, short_help=short_help, callback=self._callback)

    def _load_real_command(self):
        if self._real_command is None:
            module_path, attr_name = self.import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self._real_command = getattr(module, attr_name)
        return self._real_command

    def _callback(self, *args, **kwargs):
        _ensure_evaluation_importable()
        real_cmd = self._load_real_command()
        return real_cmd.callback(*args, **kwargs)

    def get_params(self, ctx):  # noqa: D102
        try:
            _ensure_evaluation_importable()
            real_cmd = self._load_real_command()
            return real_cmd.get_params(ctx)
        except (ImportError, ModuleNotFoundError):
            return super().get_params(ctx)


def _ensure_evaluation_importable():
    """Add the evaluation directory to sys.path if not already present."""
    eval_dir = Path(__file__).resolve().parents[2] / "evaluation"
    eval_dir_str = str(eval_dir)
    if eval_dir_str not in sys.path:
        sys.path.insert(0, eval_dir_str)


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
def dynaclr():
    """DynaCLR evaluation and analysis tools."""
    pass


dynaclr.add_command(
    LazyCommand(
        name="train-linear-classifier",
        import_path="linear_classifiers.train_linear_classifier.main",
        short_help="Train a linear classifier on cell embeddings",
    )
)

dynaclr.add_command(
    LazyCommand(
        name="apply-linear-classifier",
        import_path="linear_classifiers.apply_linear_classifier.main",
        short_help="Apply a trained linear classifier to new embeddings",
    )
)


def main():
    """Run the DynaCLR CLI."""
    dynaclr()


if __name__ == "__main__":
    main()
