"""Click-based CLI for DynaCLR evaluation and analysis tools."""

import importlib

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
    """No-op: evaluation is now part of the dynaclr package."""
    pass


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
def dynaclr():
    """DynaCLR evaluation and analysis tools."""
    pass


dynaclr.add_command(
    LazyCommand(
        name="train-linear-classifier",
        import_path="dynaclr.evaluation.linear_classifiers.scripts.train_linear_classifier.main",
        short_help="Train a linear classifier on cell embeddings",
    )
)

dynaclr.add_command(
    LazyCommand(
        name="apply-linear-classifier",
        import_path="dynaclr.evaluation.linear_classifiers.scripts.apply_linear_classifier.main",
        short_help="Apply a trained linear classifier to new embeddings",
    )
)

dynaclr.add_command(
    LazyCommand(
        name="evaluate-smoothness",
        import_path="dynaclr.evaluation.benchmarking.smoothness.evaluate_smoothness.main",
        short_help="Evaluate temporal smoothness of embedding models",
    )
)

dynaclr.add_command(
    LazyCommand(
        name="compare-models",
        import_path="dynaclr.evaluation.benchmarking.smoothness.compare_models.main",
        short_help="Compare previously saved smoothness results",
    )
)

dynaclr.add_command(
    LazyCommand(
        name="reduce-dimensionality",
        import_path="dynaclr.evaluation.dimensionality_reduction.reduce_dimensionality.main",
        short_help="Compute PCA, UMAP, and/or PHATE on saved embeddings",
    )
)


def main():
    """Run the DynaCLR CLI."""
    dynaclr()


if __name__ == "__main__":
    main()
