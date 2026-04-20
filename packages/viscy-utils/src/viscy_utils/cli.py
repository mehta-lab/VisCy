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
        # For predict/test/validate: snapshot model init_args before checkpoint
        # hparams overwrite them, then restore after.  This lets the user config
        # win over stale checkpoint values (e.g. predict_method, predict_overlap).
        #
        # For fit: skip the snapshot so checkpoint hparams correctly override
        # parser defaults (important for training resumption — lr, architecture,
        # model_config, etc. must come from the checkpoint, not defaults).
        subcommand = self.config.get("subcommand")
        saved_init_args: dict = {}
        if subcommand and subcommand != "fit":
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
        if saved_init_args:
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


_RESERVED_TOP_LEVEL_KEYS = ("launcher", "benchmark")


def _find_config_arg() -> tuple[int | None, str | None]:
    """Scan sys.argv for --config/-c and return (index, path)."""
    for i, a in enumerate(sys.argv):
        if a in ("--config", "-c"):
            if i + 1 < len(sys.argv):
                return i, sys.argv[i + 1]
            return None, None
        for prefix in ("--config=", "-c="):
            if a.startswith(prefix):
                return i, a[len(prefix) :]
    return None, None


def _replace_config_path_in_argv(config_idx: int, new_path: str) -> None:
    """Rewrite sys.argv so --config/-c points at *new_path*."""
    if "=" in sys.argv[config_idx]:
        prefix = sys.argv[config_idx].split("=", 1)[0]
        sys.argv[config_idx] = f"{prefix}={new_path}"
    else:
        sys.argv[config_idx + 1] = new_path


def _maybe_compose_config() -> None:
    """Compose config from ``base:`` references and strip reserved keys.

    Scans ``sys.argv`` for ``--config`` or ``-c`` and loads the YAML.
    If the file has a ``base:`` key, the referenced recipe fragments are
    merged via :func:`viscy_utils.compose.load_composed_config`. In all
    cases, top-level ``launcher:`` and ``benchmark:`` keys (dynacell's
    reserved benchmark metadata) are dropped before the composed YAML is
    written to a temp file, since LightningCLI rejects unknown top-level
    keys. Configs without either ``base:`` or reserved keys pass through
    unchanged.
    """
    config_idx, config_path_str = _find_config_arg()
    if config_idx is None or config_path_str is None:
        return
    config_path = Path(config_path_str)
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        return
    has_base = "base" in raw
    has_reserved = any(k in raw for k in _RESERVED_TOP_LEVEL_KEYS)
    if not (has_base or has_reserved):
        return
    composed = load_composed_config(config_path) if has_base else dict(raw)
    for k in _RESERVED_TOP_LEVEL_KEYS:
        composed.pop(k, None)
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as tmp:
        yaml.dump(composed, tmp, default_flow_style=False)
    atexit.register(lambda p=tmp.name: Path(p).unlink(missing_ok=True))
    _replace_config_path_in_argv(config_idx, tmp.name)


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
