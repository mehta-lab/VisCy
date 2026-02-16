"""Click-based CLI for DynaCLR evaluation and analysis tools."""

import importlib

import click


class LazyCommand(click.Command):
    """Lazy-load command to improve startup time."""

    def __init__(self, name, import_path, help=None, short_help=None):
        self.import_path = import_path
        self._real_command = None
        super().__init__(
            name=name, help=help, short_help=short_help, callback=self._callback
        )

    def _load_real_command(self):
        if self._real_command is None:
            module_path, attr_name = self.import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self._real_command = getattr(module, attr_name)
        return self._real_command

    def _callback(self, *args, **kwargs):
        real_cmd = self._load_real_command()
        return real_cmd.callback(*args, **kwargs)

    def get_params(self, ctx):
        real_cmd = self._load_real_command()
        return real_cmd.get_params(ctx)


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
def dynaclr():
    """DynaCLR evaluation and analysis tools."""
    pass


def main():
    """Main entry point for DynaCLR CLI."""
    dynaclr()


if __name__ == "__main__":
    main()
