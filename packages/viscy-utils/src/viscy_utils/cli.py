"""VisCy Lightning CLI with custom defaults."""

import atexit
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import yaml
from jsonargparse import lazy_instance
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from viscy_utils.compose import load_composed_config
from viscy_utils.trainer import VisCyTrainer


class VisCyCLI(LightningCLI):
    """Extending lightning CLI arguments and defaults."""

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Define custom subcommands."""
        subcommands = LightningCLI.subcommands()
        subcommand_base_args = {"model"}
        subcommands["preprocess"] = subcommand_base_args
        subcommands["export"] = subcommand_base_args
        subcommands["precompute"] = subcommand_base_args
        subcommands["convert_to_anndata"] = subcommand_base_args
        return subcommands

    def add_arguments_to_parser(self, parser) -> None:
        """Set default logger and progress bar."""
        defaults = {
            "trainer.logger": lazy_instance(
                TensorBoardLogger,
                save_dir="",
                version=datetime.now().strftime(r"%Y%m%d-%H%M%S"),
                log_graph=True,
            ),
        }
        if not sys.stdout.isatty():
            defaults["trainer.callbacks"] = [lazy_instance(TQDMProgressBar, refresh_rate=10, leave=True)]
        parser.set_defaults(defaults)

    def _parse_ckpt_path(self) -> None:
        try:
            return super()._parse_ckpt_path()
        except SystemExit:
            # FIXME: https://github.com/Lightning-AI/pytorch-lightning/issues/21255
            return None


def _setup_environment() -> None:
    """Set log level and TF32 precision."""
    log_level = os.getenv("VISCY_LOG_LEVEL", logging.INFO)
    logging.getLogger("lightning.pytorch").setLevel(log_level)
    torch.set_float32_matmul_precision("high")


def _maybe_compose_config() -> None:
    """Compose config from ``base:`` references if present.

    Scans ``sys.argv`` for ``--config`` or ``-c``, loads the YAML file,
    and if it contains a ``base:`` key, recursively merges the referenced
    recipe fragments via :func:`viscy_utils.compose.load_composed_config`.
    The composed config is written to a temp file and ``sys.argv`` is
    updated in place.  Configs without ``base:`` pass through unchanged.
    """
    # Match "--config path", "-c path", "--config=path", or "-c=path".
    config_idx: int | None = None
    config_path_str: str | None = None
    for i, a in enumerate(sys.argv):
        if a in ("--config", "-c"):
            if i + 1 < len(sys.argv):
                config_idx = i
                config_path_str = sys.argv[i + 1]
            break
        for prefix in ("--config=", "-c="):
            if a.startswith(prefix):
                config_idx = i
                config_path_str = a[len(prefix) :]
                break
        if config_idx is not None:
            break
    if config_idx is None or config_path_str is None:
        return
    config_path = Path(config_path_str)
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return  # let LightningCLI give its own diagnostic
    if not isinstance(raw, dict) or "base" not in raw:
        return
    composed = load_composed_config(config_path)
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as tmp:
        yaml.dump(composed, tmp, default_flow_style=False)
    atexit.register(lambda p=tmp.name: Path(p).unlink(missing_ok=True))
    # Replace the path in argv, handling both "--config path" and "--config=path".
    if "=" in sys.argv[config_idx]:
        prefix = sys.argv[config_idx].split("=", 1)[0]
        sys.argv[config_idx] = f"{prefix}={tmp.name}"
    else:
        sys.argv[config_idx + 1] = tmp.name


def main() -> None:
    """Run the Lightning CLI with VisCy defaults.

    Set log level, TF32 precision, and default random seed to 42.
    Compose config from ``base:`` references if present.
    """
    _setup_environment()
    _maybe_compose_config()
    require_model = {
        "preprocess",
        "precompute",
        "convert_to_anndata",
    }.isdisjoint(sys.argv)
    require_data = {
        "preprocess",
        "precompute",
        "export",
        "convert_to_anndata",
    }.isdisjoint(sys.argv)
    _ = VisCyCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule if require_data else None,
        trainer_class=VisCyTrainer,
        seed_everything_default=42,
        subclass_mode_model=require_model,
        subclass_mode_data=require_data,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"description": "Computer vision models for single-cell phenotyping."},
    )


if __name__ == "__main__":
    main()
