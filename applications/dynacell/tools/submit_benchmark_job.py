r"""Submit a dynacell benchmark leaf via sbatch.

Composes the leaf via :func:`viscy_utils.compose.load_composed_config`,
extracts the top-level ``launcher:`` block, strips reserved keys from the
resolved config, renders an sbatch script from
``tools/sbatch_template.sbatch``, writes both to ``{run_root}/resolved/``
and ``{run_root}/slurm/``, and submits via ``sbatch`` (unless
``--dry-run``).

Usage::

    uv run python applications/dynacell/tools/submit_benchmark_job.py \
        applications/dynacell/configs/benchmarks/virtual_staining/er/celldiff/ipsc_confocal/train.yml \
        --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import string
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from iohub.ngff import open_ome_zarr

from dynacell._compose_hook import _dynacell_ref_resolver
from viscy_utils.compose import deep_merge, load_composed_config
from viscy_utils.prediction_metadata import (
    PREDICTION_COMPLETE_KEY,
    completion_marker,
    outruns,
    prediction_complete,
    prediction_run,
    same_marker,
    started_marker,
    tzyx_shape,
)

_VALID_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SBATCH_DIRECTIVE_ORDER = (
    ("job_name", "--job-name"),
    ("time", "--time"),
    ("nodes", "--nodes"),
    ("ntasks_per_node", "--ntasks-per-node"),
    ("partition", "--partition"),
    ("cpus_per_task", "--cpus-per-task"),
    ("gpus", "--gpus"),
    ("mem", "--mem"),
    ("constraint", "--constraint"),
    ("exclude", "--exclude"),
)


class SbatchTemplate(string.Template):
    """Template using ``@@`` as delimiter to pass shell ``$VAR`` through verbatim."""

    delimiter = "@@"


def _parse_override(token: str) -> tuple[list[str], Any]:
    """Parse ``key.path=value`` into (path-segments, parsed-value).

    ``${...}`` interpolation is rejected outright (load_composed_config is
    pure stdlib — allowing OmegaConf-style interpolation here would create
    a semantic gap between the compose path and the override path).
    """
    if "=" not in token:
        raise SystemExit(f"--override {token!r}: missing '=' (expected key.path=value)")
    key, value = token.split("=", 1)
    if value.startswith("${"):
        raise SystemExit(f"--override {token!r}: ${{...}} interpolation is not supported")
    parsed = yaml.safe_load(value)
    return key.split("."), parsed


def _apply_override(composed: dict, path: list[str], value: Any) -> dict:
    """Deep-merge a single dotlist override and return the new config."""
    nested: Any = value
    for seg in reversed(path):
        nested = {seg: nested}
    return deep_merge(composed, nested)


_PRED_WRITER_CLASS_SUFFIX = "HCSPredictionWriter"


def _writer_callbacks(composed: dict) -> list[dict]:
    """Return the ``HCSPredictionWriter`` entries of ``trainer.callbacks``.

    Matches by ``class_path`` ending in ``HCSPredictionWriter`` -- robust
    against re-ordered callback lists and additional callbacks. Callers
    mutate the returned dicts in place; ``--override
    "trainer.callbacks[0].init_args.x=..."`` cannot do that because
    :func:`deep_merge` is dict-key-only and silently no-ops on
    ``[0]``-style segments.
    """
    callbacks = composed.get("trainer", {}).get("callbacks", [])
    if not isinstance(callbacks, list):
        raise SystemExit(
            f"trainer.callbacks must be a list to address the prediction writer (got {type(callbacks).__name__})"
        )
    return [
        cb
        for cb in callbacks
        if isinstance(cb, dict) and str(cb.get("class_path", "")).endswith(_PRED_WRITER_CLASS_SUFFIX)
    ]


def _apply_overwrite_alias(composed: dict, leaf_path: Path) -> None:
    """Set ``init_args.overwrite=True`` on every ``HCSPredictionWriter`` callback.

    Mutates ``composed`` in place. Raises if no writer is found, since the
    alias cannot do what the user asked.
    """
    writers = _writer_callbacks(composed)
    if not writers:
        callbacks = composed.get("trainer", {}).get("callbacks", [])
        class_paths = [cb.get("class_path") for cb in callbacks if isinstance(cb, dict)]
        raise SystemExit(
            f"{leaf_path}: --overwrite requested but no HCSPredictionWriter callback "
            f"found under trainer.callbacks (got class_paths={class_paths!r})"
        )
    for cb in writers:
        cb.setdefault("init_args", {})["overwrite"] = True


# ``data.init_args`` that steer loading, FOV selection and training, not the
# predicted voxels. ``data_path`` is among them: the marker records the source
# shape, and a moved input must still resume.
_LOADING_ONLY_DATA_ARGS = frozenset(
    {
        "data_path",
        "batch_size",
        "num_workers",
        "persistent_workers",
        "prefetch_factor",
        "pin_memory",
        "mmap_preload",
        "scratch_dir",
        "include_fov_names",
        "exclude_fov_names",
        "split_ratio",
        "ground_truth_masks",
        "augmentations",
        "gpu_augmentations",
        "val_augmentations",
        "val_gpu_augmentations",
        "min_nonzero_fraction",
        "nonzero_threshold",
        "nonzero_channel",
        "max_nonzero_retries",
        "fg_mask_key",
    }
)


def prediction_settings_sha256_12(composed: dict) -> str:
    """Hash the settings besides the checkpoint that shape a predict run's voxels.

    Covers the model class and its init args except ``ckpt_path`` (inference
    settings such as ``predict_method`` or ``num_generate_steps`` live there),
    the data module class and its init args except the loading-only ones in
    ``_LOADING_ONLY_DATA_ARGS`` (so ``normalizations``, channels, patch size
    and depth window count), and ``trainer.precision``. Two runs with the same
    checkpoint but different settings produce different voxels and must not
    complete one store together; a resume that changes only paths, batch
    size, workers or FOV selection still matches.
    """
    model = composed.get("model", {})
    data = composed.get("data", {})
    settings = {
        "model": {
            "class_path": model.get("class_path"),
            "init_args": {k: v for k, v in model.get("init_args", {}).items() if k != "ckpt_path"},
        },
        "data": {
            "class_path": data.get("class_path"),
            "init_args": {k: v for k, v in data.get("init_args", {}).items() if k not in _LOADING_ONLY_DATA_ARGS},
        },
        "precision": composed.get("trainer", {}).get("precision"),
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]


def bind_prediction_run(composed: dict) -> None:
    """Bind the run identity to every ``HCSPredictionWriter`` of a composed predict config.

    Mutates ``composed`` in place so each FOV's completion marker records the
    checkpoint (``model.init_args.ckpt_path``, when the model has one) and the
    hash of the other prediction settings that produced it (see
    :mod:`viscy_utils.prediction_metadata`). Shared by the single-job and batch
    launchers: a submission path that skipped it would leave the writer
    recording a null identity, and a resume through the other path would then
    reject the store as another run's.
    """
    ckpt_path = composed.get("model", {}).get("init_args", {}).get("ckpt_path")
    settings = prediction_settings_sha256_12(composed)
    for cb in _writer_callbacks(composed):
        init_args = cb.setdefault("init_args", {})
        if ckpt_path:
            init_args["checkpoint_path"] = str(ckpt_path)
        init_args["settings_sha256_12"] = settings


def _as_channel_list(target_channel: Any) -> list[str]:
    """Normalize a ``target_channel`` config value to a list of channel names."""
    if target_channel is None:
        raise SystemExit("--resume-predict requires data.init_args.target_channel")
    if isinstance(target_channel, str):
        return [target_channel]
    return list(target_channel)


def _prediction_output_store(composed: dict, leaf_path: Path) -> str:
    """Return the ``HCSPredictionWriter.output_store`` path from the composed config."""
    for cb in _writer_callbacks(composed):
        store = cb.get("init_args", {}).get("output_store")
        if store:
            return str(store)
    raise SystemExit(f"{leaf_path}: --resume-predict requires an HCSPredictionWriter callback with output_store")


@dataclass(frozen=True)
class _StoreSurvey:
    """Per-FOV resume verdicts for an existing prediction store."""

    total: int
    completed: set[str]
    conflicting: set[str]
    unverifiable: set[str]
    oversized: set[str]


def _survey_prediction_store(
    output_store: str, data_path: str, prediction_channels: list[str], run: dict[str, Any]
) -> _StoreSurvey:
    """Classify the output store's FOVs for predict resume.

    A FOV is complete when every prediction channel carries the marker this
    run would write: the input's TZYX shape plus the run identity (checkpoint
    content hash, array level, depth window and reduction), stamped by the
    writer only after all of the FOV's (T, Z-window) writes succeed. A FOV
    whose present channels carry only this run's started marker is its own
    unfinished work and is recomputed. A FOV that holds a prediction channel
    with no marker of its own predates the markers (the writer stamps a
    started marker before its first write), so nothing can vouch for it even
    when other channels of the FOV are marked: it is unverifiable rather than
    incomplete. Any other marker is conflicting: the channel was predicted
    from another source shape or with other weights or settings, and finishing
    the store would mix them. Input shapes are read
    from the array level ``run["array_key"]``, which every input must have. A
    FOV whose output array at that level outruns its source in T or Z is
    oversized whatever its markers say: another channel's run can grow the
    shared array after this channel completed, and the writer would refuse to
    rewrite it, so it is reported before a job is submitted. Metadata-only:
    reads markers and shapes, never voxel data.

    Parameters
    ----------
    output_store : str
        Path to the (possibly partial) prediction OME-Zarr store.
    data_path : str
        Path to the input HCS OME-Zarr the predict run reads.
    prediction_channels : list of str
        Channel names the writer emits (``<target>_prediction``).
    run : dict
        Run identity from :func:`viscy_utils.prediction_metadata.prediction_run`.

    Returns
    -------
    _StoreSurvey
        Total input FOV count plus plate-relative names (e.g. ``"0/0/fov0000"``)
        of complete, conflicting, unverifiable and oversized FOVs.
    """
    array_key = run["array_key"]
    input_shapes: dict[str, list[int]] = {}
    with open_ome_zarr(data_path, mode="r") as plate:
        for name, pos in plate.positions():
            input_shapes[name] = tzyx_shape(pos[array_key])
    completed: set[str] = set()
    conflicting: set[str] = set()
    unverifiable: set[str] = set()
    oversized: set[str] = set()
    if not os.path.exists(output_store):
        return _StoreSurvey(len(input_shapes), completed, conflicting, unverifiable, oversized)
    with open_ome_zarr(output_store, mode="r") as plate:
        for name, pos in plate.positions():
            if name not in input_shapes:
                continue
            try:
                output = pos[array_key]
            except KeyError:
                output = None
            present = [ch for ch in prediction_channels if ch in pos.channel_names]
            markers = pos.zattrs.get(PREDICTION_COMPLETE_KEY, {})
            if output is not None and outruns(output, input_shapes[name]):
                oversized.add(name)
            elif any(ch not in markers for ch in present):
                unverifiable.add(name)
            elif prediction_complete(pos, prediction_channels, completion_marker(input_shapes[name], run)):
                completed.add(name)
            elif not all(same_marker(markers[ch], started_marker(run)) for ch in present):
                conflicting.add(name)
    return _StoreSurvey(len(input_shapes), completed, conflicting, unverifiable, oversized)


_OPTIONAL_SBATCH_DIRECTIVES = frozenset({"constraint", "exclude"})


def _render_sbatch_directives(job_name: str, run_root: str, sbatch: dict) -> str:
    """Render ordered ``#SBATCH`` lines. Order is pinned; output/error appended last.

    Optional directives (``constraint``, ``exclude``) are skipped when the
    value is missing or null — profiles can set ``constraint: null`` to
    express "run on any GPU", and ``exclude`` can be set via ``--override
    launcher.sbatch.exclude=<hostlist>`` to steer around bad nodes.
    """
    values = dict(sbatch)
    values.setdefault("job_name", job_name)
    lines = []
    for key, flag in _SBATCH_DIRECTIVE_ORDER:
        if key not in values:
            if key in _OPTIONAL_SBATCH_DIRECTIVES:
                continue
            raise SystemExit(f"hardware profile missing sbatch.{key}")
        raw = values[key]
        if raw is None and key in _OPTIONAL_SBATCH_DIRECTIVES:
            continue
        rendered = f'"{raw}"' if flag == "--constraint" else str(raw)
        lines.append(f"#SBATCH {flag}={rendered}")
    lines.append(f"#SBATCH --output={run_root}/slurm/%j.out")
    lines.append(f"#SBATCH --error={run_root}/slurm/%j.err")
    return "\n".join(lines)


def _render_env_block(env: dict | None) -> str:
    """Render ``export KEY=VALUE`` lines, shlex-quoting values and validating keys."""
    if not env:
        return ""
    lines = []
    for k, v in env.items():
        if not _VALID_ENV_NAME.match(str(k)):
            raise SystemExit(f"launcher.env key {k!r} is not a valid shell identifier")
        lines.append(f"export {k}={shlex.quote(str(v))}")
    return "\n".join(lines)


def resolve_newest_last_ckpt(ckpt_dir: Path) -> Path | None:
    """Most recently modified ``last*.ckpt`` in ``ckpt_dir``, or None if there is none.

    Lightning writes ``last-v1.ckpt`` (then ``-v2``, ...) whenever ``last.ckpt``
    already exists, so on any resumed run the newest training state is NOT
    ``last.ckpt``. Measured across seven joint FCMAE arms at completion,
    ``last.ckpt`` lagged the true final state by 14/23/26/50/81/85/91 epochs.
    Every consumer of "the last checkpoint" — ``--resume`` and
    :func:`resolve_best_ckpt` — must therefore key on mtime, not on the name.

    Parameters
    ----------
    ckpt_dir : Path
        Directory holding a run's checkpoints.

    Returns
    -------
    Path | None
        Newest ``last*.ckpt`` by mtime, or None if ``ckpt_dir`` holds none.
    """
    lasts = sorted(ckpt_dir.glob("last*.ckpt"), key=lambda p: p.stat().st_mtime)
    return lasts[-1] if lasts else None


def resolve_best_ckpt(ckpt_dir: Path) -> Path | None:
    """Best-by-monitor checkpoint in ``ckpt_dir``, or None if none is resolvable.

    Reads ``best_model_path`` from the ``ModelCheckpoint`` callback state in the most
    recently modified ``last*.ckpt``. A resumed run leaves ``last.ckpt`` alongside
    ``last-vN.ckpt``; the newest is the authoritative training state — keying on
    ``last.ckpt`` alone mis-resolves a resumed run to its FIRST segment's best (e.g.
    the messy legacy iPSC dirs, where ``last.ckpt`` is epoch 1 but ``last-v5.ckpt`` is
    epoch 183). The training recipe uses ``monitor: loss/validate``, ``save_top_k: 5``
    with the default ``epoch=N-step=M`` filename, so the loss is not in the filename and
    the checkpoint state is the authoritative source.

    ``best_model_path`` is an ABSOLUTE path into the directory where training ran, so
    after a dir move/rename (e.g. the canonical-path migration) it is stale; the best
    file travels with the dir, so its basename under ``ckpt_dir`` is authoritative —
    prefer that, then the literal stored path, then the highest-epoch ``epoch=*.ckpt``.
    """
    newest_last = resolve_newest_last_ckpt(ckpt_dir)
    if newest_last is not None:
        import torch  # lazy: only imported for best-ckpt resolution

        state = torch.load(newest_last, map_location="cpu", weights_only=False)
        for key, val in state.get("callbacks", {}).items():
            if "ModelCheckpoint" in str(key) and isinstance(val, dict):
                best = val.get("best_model_path")
                if best:
                    rebased = ckpt_dir / Path(best).name
                    if rebased.is_file():
                        return rebased
                    if Path(best).is_file():
                        return Path(best)
    # Highest-epoch fallback. Skip any nonconforming ``epoch=*.ckpt`` (e.g.
    # ``epoch=final.ckpt``) rather than crashing mid-submit on a None re.match.
    epoch_ckpts: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("epoch=*.ckpt"):
        m = re.match(r"epoch=(\d+)", p.name)
        if m is not None:
            epoch_ckpts.append((int(m.group(1)), p))
    return max(epoch_ckpts, key=lambda t: t[0])[1] if epoch_ckpts else None


def _resolve_best_ckpt(ckpt_dir: Path) -> Path:
    """Best-by-monitor checkpoint in ``ckpt_dir``; raise if none is resolvable."""
    best = resolve_best_ckpt(ckpt_dir)
    if best is None:
        raise SystemExit(
            f"--ckpt best: no resolvable checkpoint in {ckpt_dir} (no last*.ckpt best_model_path, no epoch=*.ckpt)"
        )
    return best


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("leaf", type=Path, help="path to a benchmark leaf YAML")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write resolved config + sbatch to launcher.run_root but skip submission "
        "(requires write permission). Combine with --print-* to suppress writes.",
    )
    ap.add_argument(
        "--print-script",
        action="store_true",
        help="preview rendered sbatch to stdout. No disk writes, no submission, "
        "safe on any run_root (overrides --dry-run's disk write).",
    )
    ap.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="preview resolved YAML (launcher+benchmark stripped) to stdout. "
        "No disk writes, no submission (overrides --dry-run's disk write).",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="key.path=value",
        help="dotlist override, deep-merged after compose (repeatable). "
        "Note: list-index syntax (callbacks[0]) is NOT honored — deep_merge "
        "operates on dict keys. Use --overwrite for the prediction writer.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="set init_args.overwrite=True on every HCSPredictionWriter "
        "callback after compose. Required to re-run a leaf whose output "
        "store already contains the prediction channel. Off by default.",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        metavar="{best,last,PATH}",
        help="predict mode: override model.init_args.ckpt_path. 'best' resolves "
        "the best-by-monitor checkpoint (from last.ckpt's ModelCheckpoint state) "
        "in the leaf's checkpoint dir; 'last' uses the newest last*.ckpt by mtime "
        "(Lightning renames to last-vN.ckpt on resume); a PATH is "
        "used verbatim. Use 'best' for Phase-9 re-predict so a retrained model "
        "predicts from its new best checkpoint instead of the leaf's hardcoded "
        "(possibly stale) epoch. predict mode only (fit uses --resume).",
    )
    resume = ap.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        action="store_true",
        help="resume a fit job from the newest <run_root>/checkpoints/last*.ckpt "
        "by mtime, resolved at submit time (appends --ckpt_path to the training "
        "command). The standing policy for re-training resubmits: never restart "
        "from scratch. Lightning writes last-v1.ckpt, last-v2.ckpt, ... on each "
        "resume, so last.ckpt is the first segment's state, not the newest — "
        "mtime is what identifies the true training state. Use --resume-from to "
        "anchor on a different checkpoint. fit mode only.",
    )
    resume.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        metavar="CKPT",
        help="resume a fit job from an explicit checkpoint path (appends "
        "--ckpt_path=CKPT). Use when the resume source is not the newest "
        "last*.ckpt. fit mode only.",
    )
    ap.add_argument(
        "--resume-predict",
        action="store_true",
        help="predict mode: resume a partially-written prediction store after a "
        "wall-time kill. Skips FOVs already fully written (via exclude_fov_names) "
        "and sets the writer overwrite=True, so the resubmit continues instead of "
        "crashing on the existing prediction channel (overwrite=False) or recomputing "
        "every FOV (--overwrite alone). Requires completion markers for all Z windows; "
        "the markers record the checkpoint's content hash, depth handling and a hash of the "
        "prediction settings, so a store "
        "predicted with other weights or settings is refused rather than mixed, and a store "
        "written before markers existed cannot be verified and is refused too. "
        "Reuses the leaf's checkpoint; cannot combine with --ckpt.",
    )
    ap.add_argument(
        "--dependency",
        default=None,
        metavar="afterok:<job_id>",
        help="SLURM dependency expression for the rendered sbatch job. "
        "When set, sbatch is invoked with --dependency=<value>. Default off; "
        "manual invocations behave as before.",
    )
    ap.add_argument(
        "--parsable",
        action="store_true",
        help="Invoke sbatch with --parsable and forward sbatch's parsable "
        "stdout (typically the bare job id, or 'job_id;cluster' on "
        "multi-cluster setups) to this process's stdout, in place of "
        "sbatch's default 'Submitted batch job <id>' prose. Default off; "
        "manual invocations see the existing prose. Useful for "
        "orchestration that needs to capture sbatch's machine-readable "
        "output.",
    )
    return ap.parse_args(argv)


def submit(argv: list[str] | None = None) -> int:
    """Render and submit the leaf; return process exit code."""
    # Shared-group writes: resolved/ and slurm/ artifacts land on a shared
    # project path (`launcher.run_root`), so guarantee g+w regardless of the
    # caller's login umask. The sbatch template re-asserts umask 0002 on the
    # compute node for wandb/checkpoint/prediction outputs.
    os.umask(0o002)
    args = _parse_args(argv)

    if args.resume_predict and args.ckpt is not None:
        raise SystemExit(
            "--resume-predict cannot be combined with --ckpt: a resumed predict must reuse "
            "the original checkpoint, or the store would mix predictions from two models"
        )

    composed = load_composed_config(args.leaf, resolver=_dynacell_ref_resolver)
    for token in args.override:
        path, value = _parse_override(token)
        composed = _apply_override(composed, path, value)
    if args.overwrite:
        _apply_overwrite_alias(composed, args.leaf)

    if "launcher" not in composed:
        raise SystemExit("leaf is missing required 'launcher:' block")
    launcher = composed.pop("launcher")
    composed.pop("benchmark", None)

    mode = launcher.get("mode")
    job_name = launcher.get("job_name")
    run_root = launcher.get("run_root")
    sbatch = launcher.get("sbatch", {})
    env = launcher.get("env", {})
    if mode not in ("fit", "predict"):
        raise SystemExit(f"launcher.mode must be 'fit' or 'predict' (got {mode!r})")
    if not job_name:
        raise SystemExit("launcher.job_name must be non-empty")
    if not run_root or not str(run_root).startswith("/"):
        raise SystemExit(f"launcher.run_root must be an absolute path (got {run_root!r})")

    # Consistency: under SLURM, Lightning's SLURMEnvironment derives
    # world_size from SLURM_NTASKS — not from trainer.devices — and
    # rejects bare `--ntasks`, demanding `--ntasks-per-node` (see
    # SLURMEnvironment._validate_srun_variables). If ntasks_per_node
    # ≠ devices, DDP silently runs with the wrong world_size and only
    # some GPUs train. Invariant: trainer.devices == sbatch.ntasks_per_node,
    # and sbatch.gpus == sbatch.nodes × trainer.devices.
    trainer = composed.get("trainer", {})
    devices = trainer.get("devices")
    nodes = sbatch.get("nodes", 1)
    ntasks_per_node = sbatch.get("ntasks_per_node")
    gpus = sbatch.get("gpus")
    if not isinstance(devices, int) or ntasks_per_node != devices or gpus != nodes * devices:
        raise SystemExit(
            f"topology mismatch: trainer.devices={devices!r}, sbatch.nodes={nodes!r}, "
            f"sbatch.ntasks_per_node={ntasks_per_node!r}, sbatch.gpus={gpus!r}. "
            f"Must satisfy devices == ntasks_per_node and gpus == nodes × devices. "
            f"Check --override values or hardware profile."
        )

    # Resume support: append ``--ckpt_path`` to the fit command so a resubmit
    # continues from a checkpoint instead of restarting from scratch. Standing
    # policy for re-training resubmits (verified via LightningCLI fit, which
    # restores model+optimizer+loop state). fit mode only; predict has its own
    # checkpoint handling.
    #
    # The anchor is the NEWEST ``last*.ckpt`` by mtime, never the fixed
    # ``last.ckpt``: Lightning renames to ``last-vN.ckpt`` on every resume, so on
    # a resumed run ``last.ckpt`` is the FIRST segment's state (measured 14-91
    # epochs stale). Job 35154718 rewound 83 epochs on 2026-08-07 replaying it.
    resume_arg = ""
    if args.resume or args.resume_from is not None:
        if mode != "fit":
            raise SystemExit(f"--resume/--resume-from is only valid for fit mode (got {mode!r})")
        if args.resume_from is not None:
            ckpt = args.resume_from
            if not ckpt.is_file():
                raise SystemExit(f"resume checkpoint not found: {ckpt}")
        else:
            ckpt_dir = Path(run_root) / "checkpoints"
            newest = resolve_newest_last_ckpt(ckpt_dir)
            if newest is None:
                raise SystemExit(
                    f"resume checkpoint not found: no last*.ckpt in {ckpt_dir}. "
                    f"Pass --resume-from CKPT to name the resume source explicitly."
                )
            ckpt = newest
        resume_arg = f" --ckpt_path={shlex.quote(str(ckpt))}"

    # Predict-mode checkpoint override: repoint model.init_args.ckpt_path so a
    # re-predict uses the retrained model's best checkpoint rather than the leaf's
    # hardcoded (possibly stale/deconv) epoch. predict mode only.
    if args.ckpt is not None:
        if mode != "predict":
            raise SystemExit(f"--ckpt is only valid for predict mode (got {mode!r}); fit uses --resume")
        model_init = composed.get("model", {}).get("init_args", {})
        current_ckpt = model_init.get("ckpt_path")
        if not current_ckpt:
            raise SystemExit("--ckpt: leaf has no model.init_args.ckpt_path to resolve the checkpoint dir from")
        ckpt_dir = Path(current_ckpt).parent
        if args.ckpt == "best":
            resolved_ckpt = _resolve_best_ckpt(ckpt_dir)
        elif args.ckpt == "last":
            # Lightning renames to last-vN.ckpt whenever last.ckpt already exists, so
            # the name is not the newest file -- key on mtime, per the invariant stated
            # at resolve_newest_last_ckpt. 2066caf8 fixed --resume and _resolve_best_ckpt
            # and missed this third caller.
            resolved_ckpt = resolve_newest_last_ckpt(ckpt_dir)
            if resolved_ckpt is None:
                raise SystemExit(f"--ckpt last: no last*.ckpt in {ckpt_dir}")
        else:
            resolved_ckpt = Path(args.ckpt)
        if not resolved_ckpt.is_file():
            raise SystemExit(f"--ckpt resolved to a missing checkpoint: {resolved_ckpt}")
        model_init["ckpt_path"] = str(resolved_ckpt)

    model_init = composed.get("model", {}).get("init_args", {})
    if mode == "predict":
        bind_prediction_run(composed)

    # Predict-mode resume: continue a partially-written prediction store instead of
    # crashing or recomputing. The writer raises FileExistsError on an existing
    # prediction channel when overwrite=False, and --overwrite alone re-runs every
    # FOV from scratch (fatal for long diffusion predicts that can't fit one wall
    # window). Skip fully-written FOVs via exclude_fov_names and set overwrite=True
    # so the kept partial FOVs (and any re-done ones) can be rewritten.
    if args.resume_predict:
        if mode != "predict":
            raise SystemExit(f"--resume-predict is only valid for predict mode (got {mode!r})")
        output_store = _prediction_output_store(composed, args.leaf)
        if os.path.exists(output_store):
            data_init = composed.setdefault("data", {}).setdefault("init_args", {})
            data_path = data_init.get("data_path")
            if not data_path:
                raise SystemExit(f"{args.leaf}: --resume-predict requires data.init_args.data_path")
            pred_channels = [ch + "_prediction" for ch in _as_channel_list(data_init.get("target_channel"))]
            if not model_init.get("ckpt_path"):
                raise SystemExit(
                    f"{args.leaf}: --resume-predict requires model.init_args.ckpt_path; the completion "
                    "markers record the checkpoint so a resume cannot mix weights"
                )
            if data_init.get("z_window_size") is None:
                raise SystemExit(f"{args.leaf}: --resume-predict requires data.init_args.z_window_size")
            writer_init = _writer_callbacks(composed)[0]["init_args"]
            # Defaults mirror HCSDataModule.array_key and HCSPredictionWriter.z_reduction;
            # bind_prediction_run() set the settings hash above.
            run = prediction_run(
                array_key=str(data_init.get("array_key", "0")),
                z_window_size=int(data_init["z_window_size"]),
                z_reduction=str(writer_init.get("z_reduction", "blend")),
                checkpoint_path=model_init["ckpt_path"],
                settings_sha256_12=writer_init["settings_sha256_12"],
            )
            survey = _survey_prediction_store(output_store, data_path, pred_channels, run)
            if survey.oversized:
                examples = ", ".join(sorted(survey.oversized)[:5])
                raise SystemExit(
                    f"--resume-predict: {len(survey.oversized)} FOVs in {output_store} hold more timepoints or "
                    f"depth slices than their source ({examples}); arrays only grow, so the stale planes would "
                    "survive every rewrite. Predict into a new output store"
                )
            if survey.unverifiable:
                examples = ", ".join(sorted(survey.unverifiable)[:5])
                raise SystemExit(
                    f"--resume-predict: {len(survey.unverifiable)} FOVs in {output_store} hold "
                    f"{pred_channels} without a completion marker ({examples}), so their completion "
                    "cannot be verified and a resume would overwrite them; predict into a new output "
                    "store, or delete this one to recompute everything"
                )
            if survey.conflicting:
                examples = ", ".join(sorted(survey.conflicting)[:5])
                raise SystemExit(
                    f"--resume-predict: {len(survey.conflicting)} FOVs in {output_store} were predicted "
                    f"from another source shape or with another checkpoint or settings ({examples}); "
                    "predict into a new output store instead of mixing them"
                )
            completed, total = survey.completed, survey.total
            if total and len(completed) == total:
                print(f"--resume-predict: all {total} FOVs already complete in {output_store}; nothing to submit")
                return 0
            if completed:
                existing = data_init.get("exclude_fov_names") or []
                data_init["exclude_fov_names"] = sorted(set(existing) | completed)
            _apply_overwrite_alias(composed, args.leaf)
            print(
                f"--resume-predict: {len(completed)}/{total} FOVs complete; "
                f"predicting {total - len(completed)} remaining (overwrite=True)"
            )
        else:
            print(f"--resume-predict: no store at {output_store}; running full predict")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S_%f")
    run_root_path = Path(run_root)
    resolved_dir = run_root_path / "resolved"
    slurm_dir = run_root_path / "slurm"
    resolved_path = resolved_dir / f"{mode}_{job_name}_{timestamp}.yml"
    sbatch_path = slurm_dir / f"{timestamp}_{job_name}.sbatch"

    template_text = (Path(__file__).parent / "sbatch_template.sbatch").read_text()
    # ``repo_root`` is substituted so the template invokes the NCCL preflight
    # script by absolute path — the rendered sbatch does not ``cd`` and is
    # submitted from arbitrary CWDs, so a relative path would break.
    rendered = SbatchTemplate(template_text).substitute(
        sbatch_directives=_render_sbatch_directives(job_name, str(run_root), sbatch),
        run_root=str(run_root),
        env_block=_render_env_block(env),
        mode=mode,
        resolved_config=str(resolved_path),
        resume_arg=resume_arg,
        repo_root=str(_REPO_ROOT),
    )

    if args.print_resolved_config:
        sys.stdout.write(yaml.safe_dump(composed, default_flow_style=False))
    if args.print_script:
        sys.stdout.write(rendered)

    # Preview contract:
    # - --print-* (either) = pure preview: no disk writes, no submission.
    #   Safe against run_roots the caller can't write to.
    # - --dry-run alone = write artifacts to run_root but don't submit.
    #   Requires write permission on launcher.run_root. Use --print-script
    #   to also see the rendered sbatch on stdout.
    # - --dry-run combined with --print-* = --print-* wins (preview).
    # - Bare invocation = write + submit.
    preview_only = args.print_script or args.print_resolved_config
    skip_submit = preview_only or args.dry_run
    if not preview_only:
        resolved_dir.mkdir(parents=True, exist_ok=True)
        slurm_dir.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(yaml.safe_dump(composed, default_flow_style=False))
        sbatch_path.write_text(rendered)
    if not skip_submit:
        sbatch_cmd = ["sbatch"]
        if args.parsable:
            sbatch_cmd.append("--parsable")
        if args.dependency:
            sbatch_cmd.append(f"--dependency={args.dependency}")
        sbatch_cmd.append(str(sbatch_path))
        # --parsable mode: capture only sbatch's stdout (its parsable
        # output) and forward to our caller, so an orchestrator can chain
        # submissions. Stderr stays attached to the parent so any sbatch
        # warnings or diagnostics remain visible. Without --parsable,
        # sbatch's prose ("Submitted batch job <id>") flows through to the
        # parent's stdout untouched — backward-compatible with every
        # existing manual workflow.
        if args.parsable:
            result = subprocess.run(sbatch_cmd, check=True, stdout=subprocess.PIPE, text=True)
            print(result.stdout.strip())
        else:
            subprocess.run(sbatch_cmd, check=True)

    return 0


if __name__ == "__main__":
    sys.exit(submit())
