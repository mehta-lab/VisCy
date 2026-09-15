"""The ``dynacell`` console script routes every subcommand family it declares.

Phase 13 moved the croissant / distribution CLIs into this package but left them
unreachable: the ``dynacell-paper`` dispatcher their docstrings named was retired
in the same migration, and nothing here replaced it. These tests drive the real
``main_cli`` so a future entry-point edit cannot silently strand them again.
"""

import sys

import pytest

from dynacell.__main__ import _ARGPARSE_COMMANDS, _HYDRA_COMMANDS, main_cli


@pytest.mark.parametrize("command", sorted(_ARGPARSE_COMMANDS))
def test_argparse_subcommand_is_reachable(command, monkeypatch, capsys):
    """``dynacell <command> --help`` reaches the module's own parser."""
    monkeypatch.setattr(sys, "argv", ["dynacell", command, "--help"])
    with pytest.raises(SystemExit) as excinfo:
        main_cli()
    assert excinfo.value.code == 0
    assert f"usage: dynacell {command}" in capsys.readouterr().out


def test_argparse_and_hydra_command_names_do_not_collide():
    """A name in both tables would make routing order silently decide behavior."""
    assert not set(_ARGPARSE_COMMANDS) & set(_HYDRA_COMMANDS)


def test_argparse_targets_expose_a_main():
    """Every routed module must actually define the ``main`` this dispatcher calls."""
    import importlib

    for module_path, _extra in _ARGPARSE_COMMANDS.values():
        assert callable(importlib.import_module(module_path).main)
