#!/usr/bin/env python3
"""Generate grouped eval leaf configs for the re-eval campaign.

Walks ``{ipsc,a549}/{predictions,joint_predictions}`` under the dynacell
training tree, parses prediction-zarr filenames into canonical identities
``(organelle, model, variant, train_set, test_set, condition)``, dedupes
across the two directories preferring ``joint_predictions/``, and emits
12 production grouped-eval leaves plus a 13th sanity-probe leaf under
``applications/dynacell/configs/benchmarks/virtual_staining/_internal/leaf/grouped/``.

The leaf shape mirrors existing single-condition leaves: ``# @package
_global_`` header + overlay body (no ``defaults:`` block — that would
create a composition cycle since ``evaluate-grouped`` already starts
from ``config_name="eval_grouped"``). Each condition carries a
``benchmark.dataset_ref`` dict ``{dataset, target}`` validated against
the bundled manifests; ``apply_dataset_ref`` (Hydra-side resolver hook)
splices ``io.gt_path``, ``io.cell_segmentation_path``,
``io.gt_channel_name``, ``io.gt_cache_dir``, ``io.pred_channel_name``,
and ``pixel_metrics.spacing`` from the manifest at eval time.

See :doc:`/hpc/mydata/alex.kalinin/.claude/plans/vectorized-sleeping-clock.md`
for the campaign plan.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DYNACELL_ROOT = Path("/hpc/projects/virtual_staining/training/dynacell")

# Code → paper name (per dynacell/CLAUDE.md table).
_CODE_TO_PAPER: dict[str, str] = {
    "fcmae_vscyto3d_scratch": "unext2",
    "fcmae_vscyto3d_pretrained": "vscyto3d",
    "fnet3d_paper": "fnet3d",
    "unetvit3d": "unetvit3d",
    "celldiff": "celldiff",
    "celldiff_r2": "celldiff_r2",
}

# File-prefix → logical organelle key.
_ORG_PREFIX_TO_ORGANELLE: dict[str, str] = {
    "sec61b": "er",
    "tomm20": "mitochondria",
    "nucl": "nucleus",
    "nucleus": "nucleus",
    "memb": "membrane",
    "membrane": "membrane",
}

# iPSC: target key in aics-hipsc manifest.
_IPSC_TARGET_KEY: dict[str, str] = {
    "er": "sec61b",
    "mitochondria": "tomm20",
    "nucleus": "nucleus",
    "membrane": "membrane",
}

# A549: target key in a549-mantis-<gene>-<cond> manifest (gene marker, not logical organelle).
_A549_GENE: dict[str, str] = {
    "er": "sec61b",
    "mitochondria": "tomm20",
    "nucleus": "h2b",
    "membrane": "caax",
}

# A549 manifest slug template (per organelle, per condition).
_A549_SLUG_TEMPLATE: dict[str, str] = {
    "er": "a549-mantis-sec61b-{cond}",
    "mitochondria": "a549-mantis-tomm20-{cond}",
    "nucleus": "a549-mantis-h2b-{cond}",
    "membrane": "a549-mantis-caax-{cond}",
}

_IPSC_SLUG = "aics-hipsc"

_A549_CONDITIONS: tuple[str, ...] = ("mock", "denv", "zikv")
_CELLDIFF_VARIANTS: tuple[str, ...] = ("iterative", "sliding_window", "denoise")
_DETERMINISTIC_MODELS: tuple[str, ...] = (
    "fcmae_vscyto3d_scratch",
    "fcmae_vscyto3d_pretrained",
    "fnet3d_paper",
    "unetvit3d",
)
_CELLDIFF_MODELS: tuple[str, ...] = ("celldiff_r2", "celldiff")  # r2 first so longest match wins
_TRAIN_SETS: tuple[str, ...] = ("ipsc_trained", "joint", "a549_trained")
_ORGANELLES: tuple[str, ...] = ("er", "mitochondria", "nucleus", "membrane")

# Known stale or duplicate-named zarrs to skip entirely.
_SKIP_FILENAMES: frozenset[str] = frozenset(
    {
        # May-5 Memb CellDiff joint zarrs (ep~13 stale; superseded by *_celldiff_r2_*).
        "memb_celldiff_mock.zarr",
        "memb_celldiff_denv.zarr",
        "memb_celldiff_zikv.zarr",
        # Legacy aliases (same content under canonical fnet3d_paper / fcmae_vscyto3d_scratch names).
        "sec61b_fnet3d.zarr",
        "sec61b_unext2.zarr",
    }
)

# Non-zarr scaffolding to ignore when walking prediction dirs.
_IGNORE_NAMES: frozenset[str] = frozenset(
    {
        "checkpoints",
        "slurm",
        "resolved",
        "_stale",
        "_smoke_gate11",
    }
)

# Required GT-cache backbone shas (post-refactor).
_REQUIRED_GT_CACHE_BACKBONES: frozenset[str] = frozenset(
    {
        "dynaclr-e409a5a079aa",
        "celldino-ef7c17ffb0aa",
    }
)

# Repo paths.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # applications/dynacell/tools → repo root
_MANIFEST_ROOT = _REPO_ROOT / "applications/dynacell/src/dynacell/_manifests"
_LEAF_OUT_ROOT = _REPO_ROOT / "applications/dynacell/configs/benchmarks/virtual_staining/_internal/leaf/grouped"
_PROBE_TMP_ROOT = Path("/tmp/reeval_probe")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedZarr:
    """Canonical identity for one prediction zarr."""

    pred_path: Path
    organelle: str
    model: str
    variant: str | None
    train_set: str
    test_set: str
    condition: str | None  # mock | denv | zikv for A549, None for iPSC
    is_legacy_form: bool  # True if filename used `__<gene>_<cond>` form

    @property
    def model_variant(self) -> str:
        """Model code-name joined with its variant (or just the code-name)."""
        return f"{self.model}_{self.variant}" if self.variant else self.model

    @property
    def paper_name(self) -> str:
        """Paper-side display name for the model (per dynacell/CLAUDE.md)."""
        return _CODE_TO_PAPER[self.model]

    @property
    def paper_variant(self) -> str:
        """Paper name joined with the variant (used in canonical save_dir names)."""
        return f"{self.paper_name}_{self.variant}" if self.variant else self.paper_name

    @property
    def canonical_identity(self) -> tuple:
        """Hashable dedupe key across ``predictions/`` and ``joint_predictions/``."""
        return (
            self.organelle,
            self.model,
            self.variant,
            self.train_set,
            self.test_set,
            self.condition,
        )


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def _strip_known_suffix(stem: str, candidates: tuple[str, ...]) -> tuple[str, str | None]:
    """Return ``(prefix, matched)`` if any of ``candidates`` is a trailing ``_<x>`` suffix.

    Longest match wins so e.g. ``celldiff_r2_iterative`` beats ``celldiff_r2``.
    """
    for cand in sorted(candidates, key=len, reverse=True):
        token = f"_{cand}"
        if stem.endswith(token):
            return stem[: -len(token)], cand
    return stem, None


def _starts_with_prefix(stem: str, prefixes: tuple[str, ...]) -> tuple[str | None, str]:
    """Return ``(matched_prefix, remainder)`` if stem begins with one of ``prefixes``."""
    for cand in sorted(prefixes, key=len, reverse=True):
        token = f"{cand}_"
        if stem.startswith(token):
            return cand, stem[len(token) :]
    return None, stem


def parse_zarr_name(zarr_path: Path, dynacell_root: Path = _DYNACELL_ROOT) -> ParsedZarr:
    """Parse a prediction zarr path into a :class:`ParsedZarr`.

    Raises ``ValueError`` on unrecognized grammar.
    """
    name = zarr_path.name
    if not name.endswith(".zarr"):
        raise ValueError(f"not a .zarr: {zarr_path}")
    stem = name[: -len(".zarr")]

    # Decide test_set from directory tree.
    rel = zarr_path.relative_to(dynacell_root)
    parts = rel.parts
    if parts[0] == "ipsc":
        test_set = "ipsc"
    elif parts[0] == "a549":
        test_set = "a549"
    else:
        raise ValueError(f"unknown dataset root for {zarr_path}: parts={parts}")

    # Identify whether this is a joint_predictions/ home.
    if parts[1] in ("joint_predictions",):
        in_joint_dir = True
    elif parts[1] in ("predictions",):
        in_joint_dir = False
    else:
        raise ValueError(f"unexpected predictions subdir for {zarr_path}: {parts[1]!r}")

    # Identify the organelle prefix.
    organelle_prefix, after_org = _starts_with_prefix(stem, tuple(_ORG_PREFIX_TO_ORGANELLE))
    if organelle_prefix is None:
        raise ValueError(f"unknown prediction zarr grammar (no organelle prefix): {name}")
    organelle = _ORG_PREFIX_TO_ORGANELLE[organelle_prefix]

    is_legacy_form = False
    condition: str | None = None
    train_set_filename: str | None = None

    if test_set == "ipsc":
        # iPSC test grammar (no condition token).
        # Modern: <org>_<model>[_<variant>][_jointtrained|_a549trained]
        body, train_set_filename = _strip_known_suffix(after_org, ("jointtrained", "a549trained"))
        # Now ``body`` is <model>[_<variant>].
    else:
        # A549 test grammar — has trailing condition token.
        body = after_org
        # First check legacy form: <model_variant>__<gene>_<cond>
        # Find the `__` if present.
        if "__" in body:
            # Split once on `__`.
            left, _, right = body.partition("__")
            if right.count("_") != 1:
                raise ValueError(
                    f"legacy `__` form expects exactly `<gene>_<cond>` after `__`, got {right!r} in {name}"
                )
            gene_token, cond_token = right.split("_", 1)
            if cond_token not in _A549_CONDITIONS:
                raise ValueError(f"unknown A549 condition {cond_token!r} in {name}")
            if gene_token != organelle_prefix:
                raise ValueError(
                    f"legacy `__` form gene mismatch: prefix={organelle_prefix!r}, suffix gene={gene_token!r} in {name}"
                )
            body = left
            condition = cond_token
            is_legacy_form = True
        else:
            # Modern A549 form: <model_variant>[_jointtrained|_a549trained]_<cond>
            # Trailing _<cond> first.
            body, cond_token = _strip_known_suffix(body, _A549_CONDITIONS)
            if cond_token is None:
                raise ValueError(f"unknown A549 condition (trailing token) in {name}")
            condition = cond_token
            body, train_set_filename = _strip_known_suffix(body, ("jointtrained", "a549trained"))

    # Resolve train_set from filename infix + directory.
    if in_joint_dir:
        train_set = "joint"
    elif train_set_filename == "jointtrained":
        train_set = "joint"
    elif train_set_filename == "a549trained":
        train_set = "a549_trained"
    elif train_set_filename is None:
        train_set = "ipsc_trained"
    else:
        raise ValueError(f"unhandled train_set filename infix {train_set_filename!r} in {name}")

    # Now ``body`` is the model_variant. Split into (model, variant).
    model, variant = _split_model_variant(body, name)

    return ParsedZarr(
        pred_path=zarr_path,
        organelle=organelle,
        model=model,
        variant=variant,
        train_set=train_set,
        test_set=test_set,
        condition=condition,
        is_legacy_form=is_legacy_form,
    )


def _split_model_variant(body: str, full_name: str) -> tuple[str, str | None]:
    """Resolve ``body`` into ``(model, variant)``.

    ``body`` is the substring between the organelle prefix and any
    train_set/condition tokens — i.e. the model + optional variant.

    Variants only exist for CellDiff models. Deterministic models have
    ``variant=None`` and ``body`` is the full model code-name.
    """
    # Try CellDiff first: prefix match then variant suffix.
    for celldiff_model in _CELLDIFF_MODELS:
        if body == celldiff_model:
            return celldiff_model, None
        token = f"{celldiff_model}_"
        if body.startswith(token):
            variant = body[len(token) :]
            if variant not in _CELLDIFF_VARIANTS:
                raise ValueError(f"unknown CellDiff variant {variant!r} after model {celldiff_model!r} in {full_name}")
            return celldiff_model, variant

    # Fall through to deterministic models.
    if body in _DETERMINISTIC_MODELS:
        return body, None

    raise ValueError(f"unknown model code-name {body!r} in {full_name}")


# ---------------------------------------------------------------------------
# Walk + dedupe
# ---------------------------------------------------------------------------


_SKIP_MODELS: frozenset[str] = frozenset(
    {
        # Original CellDiff (pre-r2) — user instruction 2026-05-21: only the
        # ``celldiff_r2`` family is in scope for the re-eval campaign.
        "celldiff",
    }
)

# Prediction families owned by the separate vscyto3d-ablations eval track, not
# this grouped campaign. They have their own single-condition eval__*.yaml
# leaves under leaf/<org>/fcmae_vscyto3d_pretrained_{randinit,cytoland,
# infectionft}/, leaf/<org>/vscyto3d_{cytolandft,infectionft_dynacellft}/, and
# the dual nucleus+membrane predicts under _dual_nucl_memb/ — all driven by
# tools/run_eval_direct.slurm. Skip them during the walk rather than crash on
# grammar this parser does not model (the ``dual_`` prefix and ablation infixes).
# Substring match is sufficient: ``_cytoland`` also covers ``_cytolandft`` and
# ``_infectionft`` also covers ``_infectionft_dynacellft``.
_ABLATION_NAME_TOKENS: tuple[str, ...] = ("_randinit", "_cytoland", "_infectionft")


def _is_ablation_track_zarr(name: str) -> bool:
    """Return True if ``name`` is a vscyto3d-ablations / dual prediction zarr.

    These belong to the standalone ablation eval leaves, not the grouped
    re-eval campaign, so :func:`walk_predictions` skips them instead of
    passing them to :func:`parse_zarr_name` (which would raise).
    """
    if name.startswith("dual_"):
        return True
    return any(token in name for token in _ABLATION_NAME_TOKENS)


def walk_predictions(dynacell_root: Path = _DYNACELL_ROOT) -> list[ParsedZarr]:
    """Walk ``{ipsc,a549}/{predictions,joint_predictions}/`` and parse all zarrs.

    Dedupes by canonical identity, preferring ``joint_predictions/`` when
    the same identity appears in both directories. Drops zarrs whose parsed
    ``model`` matches :data:`_SKIP_MODELS`.
    """
    by_identity: dict[tuple, ParsedZarr] = {}
    for dataset in ("ipsc", "a549"):
        for subdir in ("predictions", "joint_predictions"):
            root = dynacell_root / dataset / subdir
            if not root.is_dir():
                continue
            for entry in sorted(root.iterdir()):
                if entry.name in _IGNORE_NAMES:
                    continue
                if entry.name.startswith("_") or entry.name.startswith("."):
                    # Scaffolding / smoke fixtures (e.g. _smoke_gate11, _stale).
                    continue
                if not entry.name.endswith(".zarr"):
                    continue
                if entry.name in _SKIP_FILENAMES:
                    continue
                if _is_ablation_track_zarr(entry.name):
                    continue
                if not entry.is_dir():
                    continue
                parsed = parse_zarr_name(entry, dynacell_root=dynacell_root)
                if parsed.model in _SKIP_MODELS:
                    continue
                existing = by_identity.get(parsed.canonical_identity)
                if existing is None:
                    by_identity[parsed.canonical_identity] = parsed
                else:
                    # Prefer joint_predictions/ path on collision.
                    if "joint_predictions" in parsed.pred_path.parts:
                        by_identity[parsed.canonical_identity] = parsed
    return list(by_identity.values())


# ---------------------------------------------------------------------------
# Save_dir + pred_cache_dir derivation
# ---------------------------------------------------------------------------

# Parent dir per (test_set, train_set) — per plan Decision #1.
_PARENT_DIR: dict[tuple[str, str], str] = {
    ("ipsc", "ipsc_trained"): "evaluations_with_embeddings",
    ("ipsc", "a549_trained"): "evaluations_a549trained_with_embeddings",
    ("ipsc", "joint"): "evaluations_jointtrained_with_embeddings",
    ("a549", "ipsc_trained"): "evaluations_with_embeddings",
    ("a549", "a549_trained"): "evaluations_a549trained_with_embeddings",
    ("a549", "joint"): "evaluations_jointtrained_with_embeddings",
}

# Eval dir name infix per train_set.
_DIR_INFIX: dict[str, str] = {
    "ipsc_trained": "",
    "a549_trained": "_a549trained",
    "joint": "_jointtrained",
}


def save_dir_for(parsed: ParsedZarr, dynacell_root: Path = _DYNACELL_ROOT) -> Path:
    """Return the canonical campaign save_dir for ``parsed``."""
    parent = _PARENT_DIR[(parsed.test_set, parsed.train_set)]
    infix = _DIR_INFIX[parsed.train_set]
    if parsed.test_set == "ipsc":
        eval_name = f"eval_{parsed.paper_variant}{infix}_{parsed.organelle}"
    else:
        eval_name = f"eval_{parsed.paper_variant}{infix}_{parsed.organelle}_{parsed.condition}"
    return dynacell_root / parsed.test_set / parent / eval_name


def pred_cache_dir_for(parsed: ParsedZarr, dynacell_root: Path = _DYNACELL_ROOT) -> Path:
    """Return canonical pred_cache_dir (see plan "Pred cache layout").

    The trailing segment is organelle-namespaced. A given
    ``(train_set, model_variant)`` is evaluated once per organelle, and the four
    organelles' prediction zarrs differ, so they must not share a cache dir.
    A549 namespaces via the gene marker (``sec61b``/``tomm20``/``h2b``/``caax``)
    plus the plate condition. iPSC has no plate condition, so it namespaces by
    the logical organelle. A bare ``ipsc`` segment collapses all four organelles
    onto one dir: the first to run wins the manifest's ``pred.plate_path`` and
    every other organelle then raises StaleCacheError.
    """
    if parsed.test_set == "ipsc":
        cond_seg = f"{parsed.organelle}_ipsc"
    else:
        gene = _A549_GENE[parsed.organelle]
        cond_seg = f"{gene}_{parsed.condition}"
    return dynacell_root / parsed.test_set / "eval_cache_pred" / parsed.train_set / parsed.model_variant / cond_seg


def benchmark_dataset_ref(parsed: ParsedZarr) -> dict[str, str]:
    """Return the ``benchmark.dataset_ref`` dict for ``parsed``."""
    if parsed.test_set == "ipsc":
        return {"dataset": _IPSC_SLUG, "target": _IPSC_TARGET_KEY[parsed.organelle]}
    return {
        "dataset": _A549_SLUG_TEMPLATE[parsed.organelle].format(cond=parsed.condition),
        "target": _A549_GENE[parsed.organelle],
    }


def condition_name(parsed: ParsedZarr) -> str:
    """Stable per-condition label."""
    if parsed.test_set == "ipsc":
        return f"{parsed.paper_variant}__{parsed.train_set}__ipsc"
    return f"{parsed.paper_variant}__{parsed.train_set}__a549_{parsed.condition}"


# ---------------------------------------------------------------------------
# Pre-checks
# ---------------------------------------------------------------------------


def _read_manifest_targets(slug: str) -> set[str]:
    """Return the set of target keys in ``<slug>/manifest.yaml``."""
    manifest = _MANIFEST_ROOT / slug / "manifest.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    with manifest.open() as f:
        data = yaml.safe_load(f)
    return set(data.get("targets", {}))


def _gt_cache_dir_for(parsed: ParsedZarr, dynacell_root: Path = _DYNACELL_ROOT) -> Path:
    """Resolve the GT-cache directory via the manifest."""
    if parsed.test_set == "ipsc":
        target_key = _IPSC_TARGET_KEY[parsed.organelle]
        manifest_slug = _IPSC_SLUG
    else:
        target_key = _A549_GENE[parsed.organelle]
        manifest_slug = _A549_SLUG_TEMPLATE[parsed.organelle].format(cond=parsed.condition)
    manifest = _MANIFEST_ROOT / manifest_slug / "manifest.yaml"
    with manifest.open() as f:
        data = yaml.safe_load(f)
    gt_cache = data["targets"][target_key]["stores"]["gt_cache_dir"]
    return Path(gt_cache)


def _gt_cache_has_required_backbones(gt_cache_dir: Path) -> bool:
    """Verify the GT cache exists and carries new DynaCLR + CellDINO features."""
    if not gt_cache_dir.is_dir():
        return False
    # Layout is under features/dynaclr/<sha12>.zarr and features/celldino/<sha12>.zarr
    dynaclr_dir = gt_cache_dir / "features" / "dynaclr"
    celldino_dir = gt_cache_dir / "features" / "celldino"
    if not dynaclr_dir.is_dir() or not celldino_dir.is_dir():
        return False
    dynaclr_shas = {p.name for p in dynaclr_dir.iterdir() if p.name.endswith(".zarr")}
    celldino_shas = {p.name for p in celldino_dir.iterdir() if p.name.endswith(".zarr")}
    # Require the post-refactor shas referenced in the plan.
    return any("e409a5a079aa" in s for s in dynaclr_shas) and any("ef7c17ffb0aa" in s for s in celldino_shas)


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------


_HYDRA_HEADER = "# @package _global_\n"
_BASE_OVERLAY: dict = {
    "compute_feature_metrics": True,
    "use_gpu": True,
    "io": {"require_complete_cache": False},
    "runtime": {
        "executor": "serial",
        "fov_workers": 1,
        "threads_per_worker": "auto",
    },
    "force_recompute": {"final_metrics": True},
}


def build_leaf_yaml(
    organelle: str,
    train_set: str,
    conditions: list[ParsedZarr],
    dynacell_root: Path = _DYNACELL_ROOT,
) -> dict:
    """Return the OmegaConf-compatible dict for one grouped leaf."""
    body: dict = {
        "target_name": organelle,
        **{k: v for k, v in _BASE_OVERLAY.items()},
    }
    condition_blocks: list[dict] = []
    for parsed in conditions:
        block = {
            "name": condition_name(parsed),
            "benchmark": {"dataset_ref": benchmark_dataset_ref(parsed)},
            "io": {
                "pred_path": str(parsed.pred_path),
                "pred_cache_dir": str(pred_cache_dir_for(parsed, dynacell_root)),
            },
            "save": {"save_dir": str(save_dir_for(parsed, dynacell_root))},
        }
        condition_blocks.append(block)
    body["conditions"] = condition_blocks
    return body


def emit_leaf_file(out_path: Path, body: dict, organelle: str, train_set: str, condition_count: int) -> None:
    """Write a grouped leaf YAML with the Hydra header + comment block."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comment = (
        f"# Grouped leaf: {organelle} bucket, {train_set} models "
        f"({condition_count} conditions). Auto-generated by "
        f"tools/generate_grouped_eval_configs.py.\n"
    )
    text = _HYDRA_HEADER + comment + yaml.safe_dump(body, default_flow_style=False, sort_keys=False)
    out_path.write_text(text)


def emit_probe_leaf(out_path: Path, parsed_pool: list[ParsedZarr], dynacell_root: Path = _DYNACELL_ROOT) -> int:
    """Pick a small subset covering every code path, redirect all save_dirs to /tmp.

    ER is chosen as the target organelle because it has both modern and
    legacy ``__<gene>_<cond>`` filename forms on A549 — so the probe
    exercises every code path the production leaves will hit. Up to 6
    conditions are picked covering distinct ``(train_set, test_set,
    is_legacy_form)`` keys.
    """
    target_org = "er"
    candidates = [p for p in parsed_pool if p.organelle == target_org]
    seen_patterns: set[tuple[str, str, bool]] = set()
    deduped: list[ParsedZarr] = []
    for p in candidates:
        key = (p.train_set, p.test_set, p.is_legacy_form)
        if key in seen_patterns:
            continue
        if not p.pred_path.is_dir():
            continue
        seen_patterns.add(key)
        deduped.append(p)
    if not deduped:
        raise RuntimeError(f"no probe candidates available for organelle {target_org!r}")

    # Override save_dirs to /tmp/reeval_probe/<name>.
    condition_blocks: list[dict] = []
    for parsed in deduped:
        block = {
            "name": condition_name(parsed),
            "benchmark": {"dataset_ref": benchmark_dataset_ref(parsed)},
            "io": {
                "pred_path": str(parsed.pred_path),
                "pred_cache_dir": str(_PROBE_TMP_ROOT / "cache" / condition_name(parsed)),
            },
            "save": {"save_dir": str(_PROBE_TMP_ROOT / "out" / condition_name(parsed))},
        }
        condition_blocks.append(block)
    body = {
        "target_name": target_org,
        **{k: v for k, v in _BASE_OVERLAY.items()},
        "conditions": condition_blocks,
    }
    comment = (
        f"# Grouped probe leaf: {target_org} bucket, {len(deduped)} conditions covering "
        f"every code path. All save_dirs/pred_cache_dirs under /tmp/reeval_probe/ "
        f"so partial runs cannot corrupt production data. Used by Step 5 of the campaign.\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_HYDRA_HEADER + comment + yaml.safe_dump(body, default_flow_style=False, sort_keys=False))
    return len(deduped)


def emit_readme(out_path: Path, buckets: dict[tuple[str, str], list[ParsedZarr]], probe_count: int) -> None:
    """Write a README summarizing per-bucket condition counts."""
    lines = ["# Grouped re-eval leaves\n"]
    lines.append("Auto-generated by `applications/dynacell/tools/generate_grouped_eval_configs.py`.\n")
    lines.append("## Bucket summary\n")
    lines.append("| Organelle | Train set | Conditions |\n|---|---|---|\n")
    for org in _ORGANELLES:
        for train_set in _TRAIN_SETS:
            n = len(buckets.get((org, train_set), []))
            lines.append(f"| {org} | {train_set} | {n} |\n")
    lines.append(f"\n## Probe leaf\n\n{probe_count} conditions under `_probe/`.\n")
    out_path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Generate the 12 grouped leaves + probe leaf + README."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dynacell-root",
        type=Path,
        default=_DYNACELL_ROOT,
        help=f"override dynacell training root (default: {_DYNACELL_ROOT})",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=_LEAF_OUT_ROOT,
        help=f"override leaf output root (default: {_LEAF_OUT_ROOT})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the bucket summary, run pre-checks, but skip writing YAMLs",
    )
    ap.add_argument(
        "--skip-fresh",
        action="store_true",
        help="suppress emission of conditions whose canonical save_dir already has "
        "a 74-col feature_metrics.csv (saves wall but leaves canonical paths empty)",
    )
    args = ap.parse_args(argv)

    parsed_pool = walk_predictions(args.dynacell_root)
    print(f"[gen] parsed {len(parsed_pool)} prediction zarrs after dedupe")

    # Group by (organelle, train_set).
    buckets: dict[tuple[str, str], list[ParsedZarr]] = {}
    for parsed in parsed_pool:
        buckets.setdefault((parsed.organelle, parsed.train_set), []).append(parsed)

    # Sort each bucket for stable output.
    for k in buckets:
        buckets[k].sort(
            key=lambda p: (
                p.test_set,
                p.condition or "",
                p.model,
                p.variant or "",
                p.is_legacy_form,
            )
        )

    # ----- Pre-checks ------------------------------------------------------

    errors: list[str] = []

    # 1. pred_path existence.
    for parsed in parsed_pool:
        if not parsed.pred_path.is_dir():
            errors.append(f"pred_path missing: {parsed.pred_path}")

    # 2. dataset_ref target exists in the manifest.
    manifest_targets_cache: dict[str, set[str]] = {}
    for parsed in parsed_pool:
        ref = benchmark_dataset_ref(parsed)
        slug = ref["dataset"]
        if slug not in manifest_targets_cache:
            try:
                manifest_targets_cache[slug] = _read_manifest_targets(slug)
            except FileNotFoundError as exc:
                errors.append(str(exc))
                manifest_targets_cache[slug] = set()
        if ref["target"] not in manifest_targets_cache[slug]:
            errors.append(f"manifest {slug!r} missing target {ref['target']!r} for {parsed.canonical_identity}")

    # 3. gt_cache_dir exists with new DynaCLR + CellDINO backbones.
    gt_cache_seen: set[Path] = set()
    for parsed in parsed_pool:
        gt_cache = _gt_cache_dir_for(parsed, args.dynacell_root)
        if gt_cache in gt_cache_seen:
            continue
        gt_cache_seen.add(gt_cache)
        if not _gt_cache_has_required_backbones(gt_cache):
            errors.append(f"gt_cache_dir missing or lacks new DynaCLR+CellDINO entries: {gt_cache}")

    # 4. Distinct save_dirs per bucket; no collision against pre-existing FRESH dirs.
    for (org, train_set), members in buckets.items():
        save_dirs_in_bucket: dict[Path, str] = {}
        for parsed in members:
            sd = save_dir_for(parsed, args.dynacell_root)
            if sd in save_dirs_in_bucket:
                errors.append(
                    f"save_dir collision in bucket ({org}, {train_set}): {sd} "
                    f"used by both {save_dirs_in_bucket[sd]!r} and {condition_name(parsed)!r}"
                )
            save_dirs_in_bucket[sd] = condition_name(parsed)

    if errors:
        print("[gen] PRE-CHECK FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"[gen] pre-checks PASSED ({len(parsed_pool)} conditions, {len(buckets)} buckets)")

    if args.skip_fresh:
        filtered_buckets: dict[tuple[str, str], list[ParsedZarr]] = {}
        skipped_count = 0
        for key, members in buckets.items():
            keep: list[ParsedZarr] = []
            for parsed in members:
                csv = save_dir_for(parsed, args.dynacell_root) / "feature_metrics.csv"
                if csv.is_file():
                    with csv.open() as f:
                        first = f.readline()
                    if len(first.split(",")) >= 70:
                        skipped_count += 1
                        continue
                keep.append(parsed)
            filtered_buckets[key] = keep
        if skipped_count:
            print(f"[gen] --skip-fresh: dropped {skipped_count} already-fresh conditions")
        buckets = filtered_buckets

    # ----- Emit ------------------------------------------------------------

    summary_lines: list[str] = []
    for org in _ORGANELLES:
        for train_set in _TRAIN_SETS:
            members = buckets.get((org, train_set), [])
            summary_lines.append(f"  {org}/{train_set}: {len(members)} conditions")
    print("[gen] bucket summary:")
    print("\n".join(summary_lines))

    if args.dry_run:
        print("[gen] --dry-run: no files written")
        return 0

    written = 0
    for org in _ORGANELLES:
        for train_set in _TRAIN_SETS:
            members = buckets.get((org, train_set), [])
            if not members:
                print(f"[gen] skipping empty bucket {org}/{train_set}")
                continue
            out_path = args.out_root / f"{org}_{train_set}" / "eval_grouped.yaml"
            body = build_leaf_yaml(org, train_set, members, args.dynacell_root)
            emit_leaf_file(out_path, body, org, train_set, len(members))
            written += 1
            print(f"[gen] wrote {out_path} ({len(members)} conditions)")

    probe_path = args.out_root / "_probe" / "eval_grouped.yaml"
    probe_count = emit_probe_leaf(probe_path, parsed_pool, args.dynacell_root)
    print(f"[gen] wrote {probe_path} ({probe_count} conditions)")

    readme_path = args.out_root / "README.md"
    emit_readme(readme_path, buckets, probe_count)
    print(f"[gen] wrote {readme_path}")

    print(f"[gen] wrote {written} production leaves + probe + README")
    return 0


if __name__ == "__main__":
    sys.exit(main())
