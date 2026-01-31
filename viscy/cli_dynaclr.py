"""Click-based CLI for DynaCLR evaluation and analysis tools."""

import importlib

import click


class LazyCommand(click.Command):
    """Lazy-load command to improve startup time."""

    def __init__(self, name, import_path, help=None, short_help=None):
        self.import_path = import_path
        self._real_command = None
        super().__init__(
            name=name,
            help=help,
            short_help=short_help,
            callback=self._callback,
            context_settings={"help_option_names": ["-h", "--help"]},
            add_help_option=True,
        )

    def _load_real_command(self):
        """Load the actual command function on first access."""
        if self._real_command is None:
            module_path, attr_name = self.import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self._real_command = getattr(module, attr_name)
        return self._real_command

    def _callback(self, *args, **kwargs):
        """Invoke the real command's callback."""
        real_cmd = self._load_real_command()
        return real_cmd.callback(*args, **kwargs)

    def get_params(self, ctx):
        """Get parameters from the real command, plus help option."""
        real_cmd = self._load_real_command()
        # Get base params (includes help option)
        base_params = super().get_params(ctx)
        # Get real command's params (excluding help)
        real_params = [p for p in real_cmd.params if p.name != "help"]
        # Combine: help from base, others from real
        help_param = [p for p in base_params if p.name == "help"]
        return help_param + real_params


# Command registry
COMMANDS = [
    {
        "name": "evaluate-smoothness",
        "import_path": "applications.DynaCLR.evaluation.evaluate_smoothness.main",
        "help": "Evaluate temporal smoothness of representation learning models",
        "short_help": "Evaluate temporal smoothness",
    },
    {
        "name": "compare-models",
        "import_path": "applications.DynaCLR.evaluation.compare_models.main",
        "help": "Compare previously saved evaluation results",
        "short_help": "Compare saved results",
    },
    {
        "name": "tracking-stats",
        "import_path": "applications.DynaCLR.evaluation.tracking_stats.main",
        "help": "Compute tracking statistics from zarr or CSV files",
        "short_help": "Compute tracking stats",
    },
]


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
def dynaclr():
    """DynaCLR evaluation and analysis tools."""
    pass


# Register commands
for cmd in COMMANDS:
    dynaclr.add_command(
        LazyCommand(
            name=cmd["name"],
            import_path=cmd["import_path"],
            help=cmd["help"],
            short_help=cmd.get("short_help"),
        )
    )


def main():
    """Main entry point for DynaCLR CLI."""
    dynaclr()


if __name__ == "__main__":
    main()
