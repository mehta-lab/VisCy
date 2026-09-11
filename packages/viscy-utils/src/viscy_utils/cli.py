"""VisCy Lightning CLI with custom defaults."""

import atexit
import logging
import os
import re
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch
import yaml
from jsonargparse import Namespace, lazy_instance
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger

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


_MODEL_CHECKPOINT_CLASS_PATH = "lightning.pytorch.callbacks.ModelCheckpoint"
_MONITOR_HEALTH_CLASS_PATH = "viscy_utils.callbacks.MonitorHealthCheck"
_OPTIMIZER_HEALTH_CLASS_PATH = "viscy_utils.callbacks.OptimizerHealthCheck"
_LATEST_CKPT_FILENAME = "latest-epoch={epoch}-step={step}"


def _callback_field(callback, key: str):
    """Return ``key`` from a callback config entry (Namespace or dict), else None."""
    if isinstance(callback, Namespace):
        return callback.get(key)
    if isinstance(callback, dict):
        return callback.get(key)
    return None


def _init_args_of(callback) -> Namespace | dict | None:
    """Return the mutable ``init_args`` mapping of a callback config entry."""
    init_args = _callback_field(callback, "init_args")
    return init_args if isinstance(init_args, (Namespace, dict)) else None


def _inject_checkpoint_guardrails(config: Namespace, subcommand: str | None) -> None:
    """Guarantee latest-weights checkpoints and fail loud on a dead monitor.

    A monitored ``ModelCheckpoint`` writes nothing once ``save_top_k`` is filled
    and the metric stops improving — ``last.ckpt`` included, because Lightning
    writes it on the same cadence. A frozen or non-finite monitor therefore ends
    a run with no weights newer than the first few epochs.

    For ``fit`` only, this appends two callbacks to the resolved config:

    1. an **unmonitored** ``ModelCheckpoint`` sharing the monitored callback's
       ``dirpath`` and **inheriting its ``every_n_epochs``**, which saves on that
       cadence unconditionally and owns ``save_last``; and
    2. a :class:`~viscy_utils.callbacks.MonitorHealthCheck` on the same metric.

    ``save_last`` is turned off on the monitored callback so exactly one callback
    writes ``last.ckpt`` and that file means "latest weights" rather than "weights
    at the last improvement".

    The monitored callback's ``monitor``, ``mode``, ``save_top_k`` and ``filename``
    are deliberately left untouched: those determine which checkpoint a run calls
    "best", so changing them would alter model selection and invalidate
    comparisons against already-published results.

    This runs on the resolved config rather than the shared trainer recipe because
    leaves override ``trainer.callbacks`` wholesale to set their own ``dirpath`` —
    a recipe-level default would be silently dropped by every real leaf.
    """
    if subcommand != "fit":
        return
    root = config.get(subcommand) if subcommand is not None else config
    if not isinstance(root, Namespace):
        return
    trainer = root.get("trainer")
    if not isinstance(trainer, Namespace):
        return
    callbacks = trainer.get("callbacks")
    if not isinstance(callbacks, list):
        return

    # Idempotent: a config that already carries the guardrails is left alone.
    for callback in callbacks:
        if _callback_field(callback, "class_path") == _MONITOR_HEALTH_CLASS_PATH:
            return

    monitored = None
    for callback in callbacks:
        if _callback_field(callback, "class_path") != _MODEL_CHECKPOINT_CLASS_PATH:
            continue
        init_args = _init_args_of(callback)
        if init_args is not None and init_args.get("monitor"):
            monitored = callback
            break
    if monitored is None:
        return

    monitored_args = _init_args_of(monitored)
    monitor = monitored_args.get("monitor")
    dirpath = monitored_args.get("dirpath")
    # Hand `last.ckpt` to the unconditional callback so it tracks the newest epoch.
    monitored_args["save_last"] = False

    # Inherit the leaf's write cadence instead of forcing every epoch. A leaf that
    # sets `every_n_epochs` is making a deliberate write-amplification decision:
    # the Phase 17 arms run ~1470 epochs, so `every_n_epochs: 10` is the difference
    # between 147 and 1470 write cycles of a 424 MB checkpoint, and this callback
    # writes *two* files (`latest-*` and `last.ckpt`) each time -- ~1.25 TB of NFS
    # traffic per arm at cadence 1. Forcing 1 here silently overrode that choice.
    #
    # The guarantee is unaffected: an *unmonitored* callback writes on its cadence
    # unconditionally, so the newest weights are always within `every_n_epochs`
    # epochs of where the run stopped, however dead the monitor is. Defaults to 1
    # when the leaf is silent, preserving the original behaviour.
    every_n_epochs = monitored_args.get("every_n_epochs") or 1

    latest_args = {
        "monitor": None,
        "filename": _LATEST_CKPT_FILENAME,
        "auto_insert_metric_name": False,
        "every_n_epochs": every_n_epochs,
        "save_top_k": 1,
        "save_last": True,
    }
    if dirpath is not None:
        latest_args["dirpath"] = dirpath
    callbacks.append(Namespace(class_path=_MODEL_CHECKPOINT_CLASS_PATH, init_args=Namespace(**latest_args)))
    callbacks.append(
        Namespace(class_path=_MONITOR_HEALTH_CLASS_PATH, init_args=Namespace(monitor=monitor, patience=10))
    )


def _inject_optimizer_health_guard(config: Namespace, subcommand: str | None) -> None:
    """Attach the step-level optimizer guard to every ``fit`` run.

    A NaN *gradient* is invisible to loss-based monitoring. Under ``16-mixed``,
    ``GradScaler`` skips the step and halves its scale while the forward — and the
    logged loss — stays finite; once the scale reaches ``0.0`` every step is
    skipped for the rest of the run and the weights never move again. Under
    ``bf16-mixed`` there is no scaler, so the NaN lands in the weights instead.
    :class:`~viscy_utils.callbacks.OptimizerHealthCheck` covers both.

    Injected here rather than in the shared trainer recipe for the same reason as
    the checkpoint guardrails: leaves override ``trainer.callbacks`` wholesale to
    set their own ``dirpath``, so a recipe-level default is silently dropped by
    every real leaf. Injecting on the resolved config makes it unforgettable.

    Unlike :func:`_inject_checkpoint_guardrails` this does not depend on the leaf
    declaring a monitored ``ModelCheckpoint`` — a run with no checkpointing at all
    can still stall — so it only requires a ``trainer.callbacks`` list to append to.
    """
    if subcommand != "fit":
        return
    root = config.get(subcommand) if subcommand is not None else config
    if not isinstance(root, Namespace):
        return
    trainer = root.get("trainer")
    if not isinstance(trainer, Namespace):
        return
    callbacks = trainer.get("callbacks")
    if not isinstance(callbacks, list):
        return
    for callback in callbacks:
        if _callback_field(callback, "class_path") == _OPTIMIZER_HEALTH_CLASS_PATH:
            return
    callbacks.append(Namespace(class_path=_OPTIMIZER_HEALTH_CLASS_PATH))


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
        """Set default logger."""
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(WandbLogger),
            }
        )

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
        _inject_checkpoint_guardrails(self.config, self.subcommand)
        _inject_optimizer_health_guard(self.config, self.subcommand)


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


def _maybe_compose_config(resolver: Callable[[dict], dict] | None = None) -> None:
    """Compose config from ``base:`` references and strip reserved keys.

    Scans ``sys.argv`` for ``--config`` or ``-c`` and loads the YAML.
    When the file has a ``base:`` key or a reserved top-level key
    (``launcher`` / ``benchmark``), it is passed through
    :func:`viscy_utils.compose.load_composed_config` with the optional
    ``resolver`` — applications (e.g. dynacell) inject a callable here
    to transform the composed dict before LightningCLI consumes it.
    Reserved top-level keys are then stripped because LightningCLI
    rejects unknown top-level keys. Configs without either ``base:`` or
    reserved keys pass through unchanged.
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
    composed = load_composed_config(config_path, resolver=resolver)
    for k in _RESERVED_TOP_LEVEL_KEYS:
        composed.pop(k, None)
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as tmp:
        yaml.dump(composed, tmp, default_flow_style=False)
    atexit.register(lambda p=tmp.name: Path(p).unlink(missing_ok=True))
    _replace_config_path_in_argv(config_idx, tmp.name)


def main(*, resolver: Callable[[dict], dict] | None = None) -> None:
    """Run the Lightning CLI with VisCy defaults.

    Set log level, TF32 precision, and default random seed to 42.
    Compose config from ``base:`` references if present. The optional
    ``resolver`` is threaded into
    :func:`viscy_utils.compose.load_composed_config` so callers can
    transform the composed dict before LightningCLI parses it.
    """
    _setup_environment()
    _maybe_compose_config(resolver=resolver)
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
