"""VisCy Lightning CLI with custom defaults."""

import atexit
import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import yaml
from jsonargparse import Namespace, lazy_instance
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from viscy_utils.compose import load_composed_config
from viscy_utils.trainer import VisCyTrainer

_WANDB_LOGGER_CLASS_PATH = "lightning.pytorch.loggers.WandbLogger"
_WANDB_RUN_NAME_PREFIX = re.compile(r"^\d{8}-\d{6}_")
_WANDB_RUN_TIMESTAMP_FORMAT = r"%Y%m%d-%H%M%S"


def _prefix_wandb_run_name(base_name: str, run_timestamp: str) -> str:
    """Return a timestamped W&B run name unless already stamped."""
    if _WANDB_RUN_NAME_PREFIX.match(base_name):
        return base_name
    return f"{run_timestamp}_{base_name}"


def _configure_wandb_logger(
    config: Namespace,
    subcommand: str | None,
    now: datetime | None = None,
) -> None:
    """Apply a consistent W&B naming and grouping convention."""
    root = config[subcommand] if subcommand is not None else config
    if not isinstance(root, Namespace):
        return
    trainer = root.get("trainer")
    if not isinstance(trainer, Namespace):
        return
    logger = trainer.get("logger")
    if not isinstance(logger, Namespace):
        return
    if logger.get("class_path") != _WANDB_LOGGER_CLASS_PATH:
        return

    init_args = logger.get("init_args")
    if not isinstance(init_args, Namespace):
        init_args = Namespace()
        logger["init_args"] = init_args

    base_name = init_args.get("name") or subcommand or "run"
    run_timestamp = (now or datetime.now()).strftime(_WANDB_RUN_TIMESTAMP_FORMAT)
    init_args["name"] = _prefix_wandb_run_name(base_name, run_timestamp)

    if init_args.get("job_type") is None and subcommand is not None:
        init_args["job_type"] = subcommand

    group_override = os.getenv("VISCY_WANDB_GROUP") or os.getenv("VISCY_WANDB_LAUNCH")
    if group_override:
        init_args["group"] = group_override
    elif init_args.get("group") is None:
        init_args["group"] = base_name


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
        # Snapshot model init_args from the user config before checkpoint hparams
        # overwrite them. LightningCLI applies checkpoint hyper_parameters as the
        # highest-priority layer, but the correct hierarchy is:
        #   base-class defaults → checkpoint hparams → user config
        # Restoring the snapshot after the merge enforces that hierarchy.
        subcommand = self.config.get("subcommand")
        saved_init_args: dict = {}
        if subcommand:
            sc = self.config.get(subcommand)
            if isinstance(sc, Namespace):
                model = sc.get("model")
                if isinstance(model, Namespace):
                    init_args = model.get("init_args")
                    if isinstance(init_args, Namespace):
                        saved_init_args = vars(init_args).copy()
        try:
            super()._parse_ckpt_path()
        except SystemExit:
            # FIXME: https://github.com/Lightning-AI/pytorch-lightning/issues/21255
            return None
        if subcommand and saved_init_args:
            sc = self.config.get(subcommand)
            if isinstance(sc, Namespace):
                model = sc.get("model")
                if isinstance(model, Namespace):
                    init_args = model.get("init_args")
                    if isinstance(init_args, Namespace):
                        for key, val in saved_init_args.items():
                            init_args[key] = val

    def before_instantiate_classes(self) -> None:
        """Apply shared config rewrites before Lightning object creation."""
        _configure_wandb_logger(self.config, self.subcommand)


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
