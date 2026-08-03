#!/usr/bin/env python3
"""Generate grouped eval leaf configs for the re-eval campaign.

Walks the canonical prediction layout
``<organelle>/<model>/<train_set>/<test>[__<cond>]/prediction.zarr`` under the
dynacell training tree, recovers each canonical identity
``(organelle, model, variant, train_set, test_set, condition)`` via
``paths.key_from_prediction_store`` (the inverse of ``paths.prediction_store``),
and emits 12 production grouped-eval leaves plus a 13th sanity-probe leaf under
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

from dynacell.evaluation import paths
from dynacell.evaluation.paths import PAPER_KEY, eval_leaf, pred_cache_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DYNACELL_ROOT = Path("/hpc/projects/virtual_staining/training/dynacell")

# Campaign model registry (membership gate for the instance-AP opt-out + the
# drift-guard tests against paths.PAPER_KEY). Paper display names come from the
# authoritative paths.PAPER_KEY; this map's values are kept for the drift check.
_CODE_TO_PAPER: dict[str, str] = {
    "fcmae_vscyto3d_scratch": "unext2",
    "fcmae_vscyto3d_pretrained": "vscyto3d",
    "fnet3d_paper": "fnet3d",
    "fnet3d_bigpatch": "fnet3d_bigpatch",
    "fnet3d_vscyto3daug": "fnet3d_vscyto3daug",
    # FNet3D temporal-sampling ablation (Phase 17): equal frame budgets, early
    # window vs spread over the whole time course.
    "fnet3d_t01": "fnet3d_t01",
    "fnet3d_tspread": "fnet3d_tspread",
    "unetvit3d": "unetvit3d",
    "pix2pix3d_unetvit": "pix2pix3d",
    "celldiff": "celldiff",
    "celldiff_r2": "celldiff_r2",
    # In-focus 2D track (2D-vs-3D benchmark).
    "fcmae_vscyto2d_scratch": "unext2_2d",
    "fcmae_vscyto2d_pretrained": "vscyto2d",
    "fnet2d": "fnet2d",
    "celldiff_2d": "celldiff_2d",
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

_CELLDIFF_VARIANTS: tuple[str, ...] = ("iterative", "sliding_window", "denoise")
_DETERMINISTIC_MODELS: tuple[str, ...] = (
    "fcmae_vscyto3d_scratch",
    "fcmae_vscyto3d_pretrained",
    "fnet3d_paper",
    "fnet3d_bigpatch",
    "fnet3d_vscyto3daug",
    "fnet3d_t01",
    "fnet3d_tspread",
    "unetvit3d",
    "pix2pix3d_unetvit",
    # In-focus 2D track — deterministic like their 3D counterparts (no diffusion
    # sampling), so the same single-pass zarr-name parser handles them.
    "fcmae_vscyto2d_scratch",
    "fcmae_vscyto2d_pretrained",
    "fnet2d",
)
_CELLDIFF_MODELS: tuple[str, ...] = ("celldiff_r2", "celldiff_2d", "celldiff")
"""CellDiff-family model tokens, longest first so prefix matching does not
truncate ``celldiff_r2``/``celldiff_2d`` down to bare ``celldiff``."""
_TRAIN_SETS: tuple[str, ...] = ("ipsc_trained", "joint", "a549_trained")
_ORGANELLES: tuple[str, ...] = ("er", "mitochondria", "nucleus", "membrane")

# Canonical on-disk organelle roots to walk. The generator keeps ``mitochondria``
# as its internal spelling (so _A549_GENE / _IPSC_TARGET_KEY / _ORGANELLES / bucket
# dir names are unchanged); only the walk boundary reverse-maps ``mito`` -> it.
_CANONICAL_ORGANELLE_ROOTS: tuple[str, ...] = ("er", "mito", "nucleus", "membrane")
_CANONICAL_ORG_TO_INTERNAL: dict[str, str] = {"mito": "mitochondria"}

# Test sets this campaign buckets. `paths.py` knows more of them (the `hek`
# third-cell-type probe), and the canonical prediction tree is shared across
# branches, so the walk filters on this rather than taking whatever appears on
# disk. Everything downstream — bucket keys, benchmark_dataset_ref,
# _gt_cache_dir_for — assumes ipsc/a549, so widening it needs those too.
_DEFAULT_TEST_SETS: frozenset[str] = frozenset({"ipsc", "a549"})

# Canonical train_set token -> generator bucket label. The FULL canonical token
# (carrying deconv provenance: a549__deconv, joint__legacy_deconvgt) drives the
# on-disk save_dir/pred_cache paths; the bucket label groups leaves and builds
# condition_name.
# The ``__bf`` tokens get their OWN buckets, unlike ``a549__deconv`` above. That
# entry is a misleading precedent: deconv provenance marks how the *target* was
# produced from the same ``Phase3D`` input, so folding it into ``a549_trained``
# groups like with like. ``__bf`` is a different *input channel* (raw Brightfield
# instead of the waveorder Phase3D reconstruction it is derived from), so folding
# it in would collide a brightfield and a phase prediction on the same
# ``canonical_identity`` and make one silently shadow the other.
_CANONICAL_TRAIN_SET_TO_BUCKET: dict[str, str] = {
    "ipsc": "ipsc_trained",
    "a549": "a549_trained",
    "a549__deconv": "a549_trained",
    "ipsc__bf": "ipsc_bf_trained",
    "a549__bf": "a549_bf_trained",
    "joint": "joint",
    "joint__legacy_deconvgt": "joint",
}

# Instance average-precision (AP_0.50..0.95 / mAP / instance_dice) is defined only
# for the two organelles with a cell-instance interpretation, and it is computed in
# the SAME pass as the pixel/feature/semantic metrics — the instance masks feed both
# the instance-AP columns and the semantic Dice/IoU rows. nucleus -> per-side
# Cellpose nucleus instances; membrane -> GT-nuclei-seeded watershed whole-cell
# instances. ER/mito have no cell instances, so they keep the semantic (supermodel)
# mask path with no instance metrics.
_INSTANCE_ORGANELLES: frozenset[str] = frozenset({"nucleus", "membrane"})
_INSTANCE_BACKEND: dict[str, str] = {"nucleus": "cpdino", "membrane": "cpdino"}

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
    train_set: str  # bucket label: ipsc_trained | a549_trained | joint
    train_set_canonical: str  # full canonical token (carries deconv provenance)
    test_set: str
    condition: str | None  # mock | denv | zikv for A549, None for iPSC

    @property
    def model_variant(self) -> str:
        """Model code-name joined with its variant (or just the code-name)."""
        return f"{self.model}_{self.variant}" if self.variant else self.model

    @property
    def paper_name(self) -> str:
        """Paper-side display name for the model (authoritative ``paths.PAPER_KEY``)."""
        return PAPER_KEY[self.model]

    @property
    def paper_variant(self) -> str:
        """Paper name joined with the variant (used in canonical save_dir names)."""
        return f"{self.paper_name}_{self.variant}" if self.variant else self.paper_name

    @property
    def canonical_identity(self) -> tuple:
        """Hashable identity per canonical prediction-store path.

        Uses ``train_set_canonical`` (not the bucket label) so raw ``a549`` and
        ``a549__deconv`` ER/mito predictions stay distinct — the on-disk layout
        guarantees exactly one prediction.zarr per identity.
        """
        return (
            self.organelle,
            self.model,
            self.variant,
            self.train_set_canonical,
            self.test_set,
            self.condition,
        )


# ---------------------------------------------------------------------------
# Prediction-store path parsing
# ---------------------------------------------------------------------------


def parse_zarr_name(zarr_path: Path, dynacell_root: Path = _DYNACELL_ROOT) -> ParsedZarr:
    """Parse a canonical prediction-store path into a :class:`ParsedZarr`.

    The identity is recovered from the directory grammar
    ``<organelle>/<model>/<train_set>/<test>[__<cond>]/prediction.zarr`` via
    :func:`paths.key_from_prediction_store` (the single source of the path
    grammar) — never from a zarr filename. Raises ``ValueError`` on a
    non-canonical/legacy path, an unknown model dir, an unknown CellDiff
    variant, or a train_set that is valid in ``paths.py`` but out of scope for
    the grouped campaign (e.g. the ablation tokens or ``a549__bf``).
    """
    key = paths.key_from_prediction_store(zarr_path, dynacell_root)
    organelle = _CANONICAL_ORG_TO_INTERNAL.get(key.organelle, key.organelle)
    model, variant = _split_model_variant(key.model, str(zarr_path))
    # ``key.train_set`` is already paths-valid; a token that is not a campaign
    # bucket (ablations, a549__bf, ...) is out of scope, not a crash. Raise the
    # documented ValueError so the instance-AP coverage audit can catch it and
    # report it alongside the other out-of-scope predictions instead of dying on
    # a bare KeyError.
    bucket = _CANONICAL_TRAIN_SET_TO_BUCKET.get(key.train_set)
    if bucket is None:
        raise ValueError(
            f"train_set {key.train_set!r} in {zarr_path} is valid in paths.py but is not a "
            f"grouped-campaign bucket (in scope: {sorted(_CANONICAL_TRAIN_SET_TO_BUCKET)})"
        )
    return ParsedZarr(
        pred_path=zarr_path,
        organelle=organelle,
        model=model,
        variant=variant,
        train_set=bucket,
        train_set_canonical=key.train_set,
        test_set=key.test_set,
        condition=key.condition,
    )


def _split_model_variant(body: str, full_name: str) -> tuple[str, str | None]:
    """Resolve a model-dir name ``body`` into ``(model, variant)``.

    ``body`` is the canonical model directory name (``key.model`` from the
    prediction-store path) — i.e. the model + optional variant. R1 CellDiff
    variant dirs (``celldiff_iterative`` / ``celldiff_denoise`` /
    ``celldiff_sliding_window`` / bare ``celldiff``) collapse to model
    ``celldiff`` here, then get dropped by :data:`_SKIP_MODELS`.

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


def leaf_test_set(leaf: str) -> str:
    """Return the test-set token of a ``<test>[__<cond>]`` leaf segment.

    A deliberate cheap prefix read rather than ``paths._parse_leaf_suffix``: this
    runs BEFORE :func:`parse_zarr_name`, so an out-of-scope test set must be
    *filterable* rather than raise. An unrecognized segment returns itself, which
    no test-set filter accepts, so it is skipped like any other out-of-scope leaf.
    """
    return leaf.split("__", 1)[0]


def walk_predictions(
    dynacell_root: Path = _DYNACELL_ROOT,
    test_sets: frozenset[str] = _DEFAULT_TEST_SETS,
) -> list[ParsedZarr]:
    """Discover every canonical prediction zarr under the 4 organelle roots.

    Iterates ``<organelle>/<model>/<train_set>/<test>[__<cond>]/prediction.zarr``
    for ``er, mito, nucleus, membrane`` and parses each into a :class:`ParsedZarr`.
    Drops R1 CellDiff (``model`` in :data:`_SKIP_MODELS`). The canonical layout
    guarantees exactly one prediction.zarr per identity, so a duplicate
    ``canonical_identity`` is a layout violation and raises (never silently
    prefers one). The legacy ``{predictions,joint_predictions}`` dirs are NOT
    read.

    ``test_sets`` gates which leaves are considered, defaulting to the campaign's
    :data:`_DEFAULT_TEST_SETS`. This is a load-bearing guard, not a convenience:
    the canonical tree is SHARED across branches and campaigns, so a prediction
    written for a probe on another test set (the ``hek__<arm>`` third-cell-type
    leaves) would otherwise be picked up here, bucketed by ``(organelle,
    train_set)`` into the 12 committed campaign leaves, and fed to
    ``benchmark_dataset_ref`` / ``_gt_cache_dir_for``, which key off the A549
    condition vocabulary and would fail the whole generation run. Opt such a probe
    in explicitly instead.
    """
    by_identity: dict[tuple, ParsedZarr] = {}
    for organelle_root in _CANONICAL_ORGANELLE_ROOTS:
        root = dynacell_root / organelle_root
        if not root.is_dir():
            continue
        # Bounded 3-level glob, NOT rglob("prediction.zarr"): rglob descends into
        # every zarr's chunk tree (minutes per organelle); the canonical layout is
        # exactly <model>/<train_set>/<test>/prediction.zarr.
        for zarr_path in sorted(root.glob("*/*/*/prediction.zarr")):
            if leaf_test_set(zarr_path.parent.name) not in test_sets:
                continue
            parsed = parse_zarr_name(zarr_path, dynacell_root=dynacell_root)
            if parsed.model in _SKIP_MODELS:
                continue
            existing = by_identity.get(parsed.canonical_identity)
            if existing is not None:
                raise ValueError(
                    f"duplicate canonical identity {parsed.canonical_identity} from "
                    f"{existing.pred_path} and {parsed.pred_path}"
                )
            by_identity[parsed.canonical_identity] = parsed
    return list(by_identity.values())


# ---------------------------------------------------------------------------
# Save_dir + pred_cache_dir derivation
# ---------------------------------------------------------------------------


def save_dir_for(parsed: ParsedZarr, dynacell_root: Path = _DYNACELL_ROOT) -> Path:
    """Return the canonical eval leaf dir for ``parsed`` (see ``paths.eval_leaf``).

    Uses the FULL canonical train_set token (``train_set_canonical``) so ER/mito
    deconv provenance (``a549__deconv`` / ``joint__legacy_deconvgt``) is preserved
    in the on-disk path rather than collapsed to the lossy bucket label.
    """
    return eval_leaf(
        organelle=parsed.organelle,
        model=parsed.model_variant,
        train_set=parsed.train_set_canonical,
        test_set=parsed.test_set,
        condition=parsed.condition,
        data_root=dynacell_root,
    )


def pred_cache_dir_for(parsed: ParsedZarr, dynacell_root: Path = _DYNACELL_ROOT) -> Path:
    """Return the canonical pred-side feature cache dir (see ``paths.pred_cache_dir``)."""
    return pred_cache_dir(
        organelle=parsed.organelle,
        model=parsed.model_variant,
        train_set=parsed.train_set_canonical,
        test_set=parsed.test_set,
        condition=parsed.condition,
        data_root=dynacell_root,
    )


def benchmark_dataset_ref(parsed: ParsedZarr) -> dict[str, str]:
    """Return the ``benchmark.dataset_ref`` dict for ``parsed``."""
    if parsed.test_set == "ipsc":
        return {"dataset": _IPSC_SLUG, "target": _IPSC_TARGET_KEY[parsed.organelle]}
    return {
        "dataset": _A549_SLUG_TEMPLATE[parsed.organelle].format(cond=parsed.condition),
        "target": _A549_GENE[parsed.organelle],
    }


def a549_nuclei_store(condition: str) -> str:
    """Test-store path of the A549 H2B (nuclei) manifest for ``condition``.

    Whole-cell (membrane) instance segmentation seeds its watershed from the GT
    nuclei, which on A549 live in a SEPARATE ``H2B_<cond>.ozx`` store (the membrane
    GT is ``CAAX_<cond>.ozx``), positions matched 1:1 by name. iPSC nuclei live in
    the same ``cell.zarr`` as the membrane GT, so no separate path is needed there.
    """
    manifest = _MANIFEST_ROOT / f"a549-mantis-h2b-{condition}" / "manifest.yaml"
    with manifest.open() as f:
        data = yaml.safe_load(f)
    return data["targets"]["h2b"]["stores"]["test"]


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
    # nucleus & membrane compute instance AP in the same pass as pixel/feature/
    # semantic metrics. The instance masks (Cellpose nucleus / GT-nuclei-seeded
    # watershed whole-cell) feed both the AP_*/mAP/instance_dice columns AND the
    # semantic Dice/IoU rows, so the semantic seg is instance-derived (matching how
    # the paper tables are built) rather than the supermodel binary path. ER/mito
    # have no cell instances and keep the default semantic path.
    if organelle in _INSTANCE_ORGANELLES:
        body["compute_instance_ap"] = True
        seg: dict = {"backend": _INSTANCE_BACKEND[organelle]}
        if organelle == "membrane":
            # Carved cytoplasm-shape metrics are canonical (6aedf52f): inherit
            # the eval.yaml subtract_nuclei=true default — do NOT re-add a
            # subtract_nuclei=false override here (that restores whole-cell).
            seg["nuclei_channel_name"] = "Nuclei"
        body["segmentation"] = seg
    condition_blocks: list[dict] = []
    # Collision guard: raw ``a549`` and ``a549__deconv`` ER/mito preds both fall in
    # the ``a549_trained`` bucket and produce the SAME condition_name (built from the
    # bucket label), yet carry distinct save_dir / train_set_canonical. Fail loud on
    # such a pair rather than silently overwrite one leaf condition with the other.
    seen_condition_names: dict[str, str] = {}
    for parsed in conditions:
        cname = condition_name(parsed)
        prior_canonical = seen_condition_names.get(cname)
        if prior_canonical is not None and prior_canonical != parsed.train_set_canonical:
            raise ValueError(
                f"condition_name collision in bucket ({organelle}, {train_set}): {cname!r} maps "
                f"to distinct train_set_canonical {prior_canonical!r} and {parsed.train_set_canonical!r} "
                f"(raw a549 vs a549__deconv provenance) — distinct save_dirs, same leaf condition"
            )
        seen_condition_names[cname] = parsed.train_set_canonical
        io_block: dict = {
            "pred_path": str(parsed.pred_path),
            "pred_cache_dir": str(pred_cache_dir_for(parsed, dynacell_root)),
        }
        # Whole-cell watershed on A549 seeds from the separate H2B nuclei store
        # (per-condition io override; iPSC reads nuclei from the membrane gt_path).
        if organelle == "membrane" and parsed.test_set == "a549":
            io_block["nuclei_gt_path"] = a549_nuclei_store(parsed.condition)
        block = {
            "name": cname,
            "benchmark": {"dataset_ref": benchmark_dataset_ref(parsed)},
            "io": io_block,
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

    ER is chosen as the target organelle because its train_sets carry deconv
    provenance (``a549__deconv`` + ``joint__legacy_deconvgt``) alongside the raw
    ``ipsc`` pool — the widest span of canonical train_set tokens — so the probe
    exercises every code path the production leaves will hit. One condition per
    distinct ``(train_set_canonical, test_set)`` key is picked.
    """
    target_org = "er"
    candidates = [p for p in parsed_pool if p.organelle == target_org]
    seen_patterns: set[tuple[str, str]] = set()
    deduped: list[ParsedZarr] = []
    for p in candidates:
        key = (p.train_set_canonical, p.test_set)
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
    ap.add_argument(
        "--test-sets",
        default=",".join(sorted(_DEFAULT_TEST_SETS)),
        help="comma-separated test sets to bucket (default: %(default)s). Predictions on "
        "any other test set in the shared canonical tree are skipped; widening this "
        "also needs benchmark_dataset_ref and _gt_cache_dir_for to handle them",
    )
    args = ap.parse_args(argv)

    test_sets = frozenset(t.strip() for t in args.test_sets.split(",") if t.strip())
    parsed_pool = walk_predictions(args.dynacell_root, test_sets=test_sets)
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
                p.train_set_canonical,
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
