"""Canonical artifact-path grammar for dynacell virtual-staining benchmarks.

Single source of truth for the on-disk locations of trained-model checkpoints,
prediction zarrs, and evaluation outputs. Every VisCy-internal consumer imports
from here; the paper repo vendor-copies this logic and asserts parity against a
snapshot (see :doc:`radiant-skipping-corbato`).

The grammar keys every artifact on the same conceptual tuple::

    (organelle, model, train_set, test_set, condition[, component])

with a small structured vocabulary and three registries:

1. **config-token -> canonical alias** (:data:`_ORG_ALIAS`, :data:`_TRAIN_ALIAS`,
   :data:`_MODEL_PAPER_TO_CODE`) normalizes the many on-disk config spellings to
   canonical tokens before validation.
2. **canonical -> paper display registry** (:data:`PAPER_KEY`,
   :data:`ORGANELLE_PAPER`) renders code tokens for figures/tables.
3. :func:`resolve_model` recovers the true model identity from a checkpoint path
   (never trusting a bare ``model_name: celldiff``).

Grammar functions (:func:`checkpoint_dir`, :func:`prediction_store`,
:func:`eval_leaf`, :func:`gt_cache_dir`, :func:`pred_cache_dir`,
:func:`metrics_repo_dir`, :func:`iter_organelle_evals`) build canonical paths;
:func:`normalize_legacy` recognizes every pre-canonical on-disk form and maps it
to a :class:`CanonicalKey` (or ``None`` -> UNMAPPED, never a guess).

Historical note
---------------
This module replaces ``save_paths.py``. The legacy ``eval_save_dir`` /
``PAPER_KEY`` (paper-key + ``*_with_embeddings`` scheme) is retained here only via
the display registry and :func:`normalize_legacy`; the forward grammar emits the
new model-centric layout ``<organelle>/<model>/<train_set>/<test>[__cond]/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ===========================================================================
# Roots (two filesystems — checkpoints on Lustre, predictions/evals on VAST)
# ===========================================================================

MODELS_ROOT = Path("/hpc/projects/comp.micro/virtual_staining/models/dynacell")
DATA_ROOT = Path("/hpc/projects/virtual_staining/training/dynacell")

# Legacy default retained for callers still passing the old constant name.
_DEFAULT_DATA_ROOT = DATA_ROOT
DEFAULT_EVAL_RUN_ROOT = DATA_ROOT / "eval_runs"

# ===========================================================================
# Canonical vocabulary
# ===========================================================================

# Single-target organelle tokens.
_SINGLE_ORGANELLES: frozenset[str] = frozenset({"nucleus", "membrane", "er", "mito"})
# Multi-target combined tokens (canonical sorted underscore-join of components).
_MULTI_ORGANELLES: frozenset[str] = frozenset({"dual_nucleus_membrane"})
# Component sets for each multi-target token (canonical component order).
_MULTI_COMPONENTS: dict[str, tuple[str, ...]] = {
    "dual_nucleus_membrane": ("nucleus", "membrane"),
}
_ALL_ORGANELLES: frozenset[str] = _SINGLE_ORGANELLES | _MULTI_ORGANELLES

# Test sets and A549 conditions.
_TEST_SETS: frozenset[str] = frozenset({"ipsc", "a549"})
_CONDITIONS: frozenset[str] = frozenset({"mock", "denv", "zikv"})

# Forward-emittable structured train_set tokens. ``__deconv`` is valid only for
# ER/mito (see _tuple_is_valid); ``__bf`` / ``__bf__deconv`` are grammar-ready
# but data-side follow-ups. ``joint__deconv`` and ``ipsc__deconv`` are NOT valid.
_FORWARD_TRAIN_SETS: frozenset[str] = frozenset(
    {
        "ipsc",
        "a549",
        "a549__deconv",
        "a549__bf",
        "a549__bf__deconv",
        "joint",
    }
)
# Closed legacy train-set markers — never emitted forward, only recognized.
# ``joint__legacy_deconvgt``: pre-existing mixed-joint ER/mito artifacts trained
# against the legacy deconv A549 GT (joint is a mixture, so NOT ``joint__deconv``).
_LEGACY_JOINT_DECONV = "joint__legacy_deconvgt"
# Closed ablation train_set tokens: the ``_no_train_*`` ablations of
# ``fcmae_vscyto3d_pretrained``. Valid ONLY paired with their ablation model keys.
_ABLATION_TRAIN_SETS: frozenset[str] = frozenset({"randinit", "cytoland", "infectionft"})
_LEGACY_TRAIN_SETS: frozenset[str] = _ABLATION_TRAIN_SETS | {_LEGACY_JOINT_DECONV}

# Model code keys (open set — the display registry below is the closed subset with
# known paper names; unknown-but-parseable keys can still form valid paths, but
# paper_key() raises on them).
_ABLATION_MODELS: frozenset[str] = frozenset(
    {
        "fcmae_vscyto3d_pretrained_randinit",
        "fcmae_vscyto3d_pretrained_cytoland",
        "fcmae_vscyto3d_pretrained_infectionft",
    }
)

# ===========================================================================
# Map (b): canonical -> paper display registry
# ===========================================================================

# Model code key -> paper display name. Source of truth: applications/dynacell/CLAUDE.md
# plus the grouped generator's _CODE_TO_PAPER. The paper renders the CELL-Diff
# family as one "CELL-Diff" row (display collapse) — but the PATH always uses the
# true model key (celldiff_r2, celldiff_r2_iterative, ...), so R1/R2 stay distinct
# on disk. celldiff_r2 and unext2_timm_scratch are live PATH tokens now (the new
# grammar puts the true model key in the path), so both MUST be present here or
# paper_key() raises.
PAPER_KEY: dict[str, str] = {
    "fcmae_vscyto3d_scratch": "unext2",
    "fcmae_vscyto3d_pretrained": "vscyto3d",
    "fnet3d_paper": "fnet3d",
    "unetvit3d": "unetvit3d",
    # pix2pix3d GAN (UNetViT3D generator, DynacellGAN engine). Distinct paper key
    # from the deterministic `unetvit3d` so eval dirs don't collide.
    "pix2pix3d_unetvit": "pix2pix3d",
    # unext2_timm_scratch is a DISTINCT model from fcmae_vscyto3d_scratch; it is a
    # live path token (the `er/unext2/` config dir carries model_name:
    # unext2_timm_scratch). Do NOT alias it to fcmae_vscyto3d_scratch.
    "unext2_timm_scratch": "unext2_timm_scratch",
    # CELL-Diff variants. celldiff (R1) collapses to the iterative R2 paper key for
    # iPSC-trained displays; celldiff_r2 is now a first-class path token.
    "celldiff": "celldiff_r2_iterative",
    "celldiff_r2": "celldiff_r2",
    "celldiff_iterative": "celldiff_r2_iterative",
    "celldiff_r2_iterative": "celldiff_r2_iterative",
    "celldiff_sliding_window": "celldiff_r2_iterative",
    "celldiff_r2_sliding_window": "celldiff_r2_iterative",
    "celldiff_denoise": "celldiff_r2_iterative",
    "celldiff_r2_denoise": "celldiff_r2_iterative",
    # VSCyto3D ablations: random-init, external-ckpt (no-FT), and dynacell-FT from
    # cytoland / infection-FT sources. Launched outside the standard submitter.
    "fcmae_vscyto3d_pretrained_randinit": "vscyto3d_randinit",
    "fcmae_vscyto3d_pretrained_cytoland": "vscyto3d_cytoland",
    "fcmae_vscyto3d_pretrained_infectionft": "vscyto3d_infectionft",
    "vscyto3d_cytolandft": "vscyto3d_cytolandft",
    "vscyto3d_infectionft_dynacellft": "vscyto3d_infectionft_dynacellft",
}

# Organelle code token -> paper-script organelle key. Mito uses the long form
# `mitochondria` in paper outputs.
ORGANELLE_PAPER: dict[str, str] = {
    "nucleus": "nucleus",
    "membrane": "membrane",
    "er": "er",
    "mito": "mitochondria",
    "dual_nucleus_membrane": "dual_nucleus_membrane",
}

# Organelle -> LaTeX display (paper figures/tables).
ORGANELLE_LATEX: dict[str, str] = {
    "nucleus": "Nucleus",
    "membrane": "Membrane",
    "er": "ER\\ (SEC61B)",
    "mito": "Mito.\\ (TOMM20)",
}

# Train-domain -> paper pool display.
POOL_DISPLAY: dict[str, str] = {"ipsc": "iPSC", "a549": "A549", "joint": "Joint"}
POOLS: tuple[str, ...] = ("iPSC", "A549", "Joint")

# condition <-> cell_type (paper display).
CONDITION_DISPLAY: dict[str, str] = {"mock": "Mock", "denv": "DENV", "zikv": "ZIKV"}

# ===========================================================================
# Map (a): config-token -> canonical alias
# ===========================================================================

# Organelle config spellings -> canonical organelle token. Covers the three
# on-disk dual spellings (config peer dir `_dual_nucl_memb`, overlay file
# `dual_nucl_memb`, zarr prefix `dual_nucl_memb`) and the zarr prefixes nucl/memb.
_ORG_ALIAS: dict[str, str] = {
    "nucleus": "nucleus",
    "nucl": "nucleus",
    "membrane": "membrane",
    "memb": "membrane",
    "er": "er",
    "sec61b": "er",
    "mito": "mito",
    "mitochondria": "mito",
    "tomm20": "mito",
    "dual_nucleus_membrane": "dual_nucleus_membrane",
    "dual_nucl_memb": "dual_nucleus_membrane",
    "_dual_nucl_memb": "dual_nucleus_membrane",
}

# Train-set config spellings -> canonical structured train_set token.
_TRAIN_ALIAS: dict[str, str] = {
    "ipsc_confocal": "ipsc",
    "ipsc": "ipsc",
    "a549_mantis": "a549",
    "a549": "a549",
    "a549_mantis_deconv": "a549__deconv",
    "a549__deconv": "a549__deconv",
    "a549_mantis_bf": "a549__bf",
    "a549_mantis_brightfield": "a549__bf",
    "a549__bf": "a549__bf",
    "joint_ipsc_confocal_a549_mantis": "joint",
    "joint": "joint",
}

# Model paper key -> code key (the reverse of the display registry, for tokens the
# paper writes into config leaves). NOTE: NO blanket `unext2 -> ...` alias — the
# bare `unext2` token is ambiguous (legacy paper key for fcmae_vscyto3d_scratch in
# old eval_unext2_* dirs vs the code model unext2_timm_scratch). Resolve per-artifact
# via resolve_model.
_MODEL_PAPER_TO_CODE: dict[str, str] = {
    "vscyto3d": "fcmae_vscyto3d_pretrained",
    "fnet3d": "fnet3d_paper",
    "pix2pix3d": "pix2pix3d_unetvit",
}

# ===========================================================================
# GT-cache target-keying (frozen, model-independent; case-inconsistent on disk)
# ===========================================================================

# iPSC GT caches are keyed by an UPPERCASE gene for ER/mito, lowercase logical
# organelle for nucleus/membrane (verified on disk: SEC61B/TOMM20/nucleus/membrane).
_IPSC_GT_CACHE_KEY: dict[str, str] = {
    "er": "SEC61B",
    "mito": "TOMM20",
    "nucleus": "nucleus",
    "membrane": "membrane",
}
# A549 GT caches are keyed by lowercase gene marker + condition (sec61b_mock, ...).
_A549_GENE: dict[str, str] = {
    "er": "sec61b",
    "mito": "tomm20",
    "nucleus": "h2b",
    "membrane": "caax",
}

# ===========================================================================
# Map (c): canonical eval `target` group name (GT-repr-aware)
# ===========================================================================

# Base organelle -> eval-side Hydra `target` group name (raw GT). ER/Mito
# disambiguate by gene to match the target YAMLs under
# _internal/shared/eval/target/. The deconv variants live in the SAME eval target
# tree and are gated on gt_repr="deconv".
_ORGANELLE_EVAL_TARGET_RAW: dict[str, str] = {
    "nucleus": "nucleus",
    "membrane": "membrane",
    "er": "er_sec61b",
    "mito": "mito_tomm20",
}
_ORGANELLE_EVAL_TARGET_DECONV: dict[str, str] = {
    "er": "er_sec61b_deconvolved",
    "mito": "mito_tomm20_deconvolved",
}


class _OrganelleEvalTargetMap:
    """GT-repr-aware organelle -> eval `target` group name.

    Behaves like the old ``ORGANELLE_EVAL_TARGET`` dict for the raw default
    (``ORGANELLE_EVAL_TARGET[organelle]``), and adds
    ``ORGANELLE_EVAL_TARGET.for_repr(organelle, gt_repr)`` for the deconv-GT track.
    """

    def __getitem__(self, organelle: str) -> str:
        return self.for_repr(organelle, "raw")

    def __contains__(self, organelle: str) -> bool:
        return organelle in _ORGANELLE_EVAL_TARGET_RAW

    def for_repr(self, organelle: str, gt_repr: str = "raw") -> str:
        """Return the eval `target` group for ``organelle`` under ``gt_repr``.

        Parameters
        ----------
        organelle : str
            Canonical single-target organelle token (nucleus/membrane/er/mito).
        gt_repr : {"raw", "deconv"}, optional
            GT representation. ``deconv`` is valid only for ER/mito.

        Raises
        ------
        ValueError
            If organelle is unknown, or gt_repr is deconv for a non-ER/mito
            organelle, or gt_repr is not one of {raw, deconv}.
        """
        if gt_repr == "raw":
            if organelle not in _ORGANELLE_EVAL_TARGET_RAW:
                raise ValueError(
                    f"unknown organelle {organelle!r}; expected one of {sorted(_ORGANELLE_EVAL_TARGET_RAW)}"
                )
            return _ORGANELLE_EVAL_TARGET_RAW[organelle]
        if gt_repr == "deconv":
            if organelle not in _ORGANELLE_EVAL_TARGET_DECONV:
                raise ValueError(f"deconv GT target is valid only for er|mito, got organelle={organelle!r}")
            return _ORGANELLE_EVAL_TARGET_DECONV[organelle]
        raise ValueError(f"unknown gt_repr {gt_repr!r}; expected 'raw' or 'deconv'")


ORGANELLE_EVAL_TARGET = _OrganelleEvalTargetMap()

# ===========================================================================
# CanonicalKey
# ===========================================================================


@dataclass(frozen=True)
class CanonicalKey:
    """Canonical identity of one artifact tuple.

    Attributes
    ----------
    organelle : str
        Canonical organelle token (single-target or multi-target combined).
    model : str
        Model code key.
    train_set : str
        Structured train_set token (or a closed legacy marker).
    test_set : str
        ``ipsc`` | ``a549``.
    condition : str | None
        ``mock`` | ``denv`` | ``zikv`` for A549; ``None`` for iPSC.
    component : str | None
        Per-component organelle for a multi-target eval; ``None`` for a
        single-target leaf or a shared multi-target prediction.
    track : str
        Eval track: ``default`` | ``instance_ap``.
    gt_repr : str
        GT representation the eval was scored against: ``raw`` | ``deconv``.
    """

    organelle: str
    model: str
    train_set: str
    test_set: str
    condition: str | None = None
    component: str | None = None
    track: str = "default"
    gt_repr: str = "raw"


# ===========================================================================
# Validation
# ===========================================================================


def _tuple_is_valid(organelle: str, train_set: str, model: str | None = None) -> bool:
    """Return True if the (organelle, train_set[, model]) combination is data-valid."""
    if organelle not in _ALL_ORGANELLES:
        return False
    # deconv (forward or legacy-joint) is valid only for ER/mito.
    is_deconv = train_set == "a549__deconv" or train_set.endswith("__deconv") or train_set == _LEGACY_JOINT_DECONV
    if is_deconv and organelle not in {"er", "mito"}:
        return False
    if train_set in _FORWARD_TRAIN_SETS:
        return True
    if train_set == _LEGACY_JOINT_DECONV:
        return True
    if train_set in _ABLATION_TRAIN_SETS:
        # Ablation train_set is valid only paired with its ablation model.
        if model is None:
            return True
        return model in _ABLATION_MODELS
    return False


def _validate(organelle: str, train_set: str, model: str | None = None) -> None:
    if organelle not in _ALL_ORGANELLES:
        raise ValueError(f"unknown organelle {organelle!r}; expected one of {sorted(_ALL_ORGANELLES)}")
    if not _tuple_is_valid(organelle, train_set, model):
        raise ValueError(
            f"invalid (organelle={organelle!r}, train_set={train_set!r}, model={model!r}) tuple: "
            f"__deconv is er/mito-only; ipsc__deconv/joint__deconv are not valid forward tokens; "
            f"ablation train_sets {sorted(_ABLATION_TRAIN_SETS)} pair only with ablation models"
        )


def _norm_organelle(token: str) -> str:
    """Normalize a config-side organelle token to canonical. Raises on unknown."""
    if token in _ALL_ORGANELLES:
        return token
    if token in _ORG_ALIAS:
        return _ORG_ALIAS[token]
    raise ValueError(f"unknown organelle token {token!r}; expected one of {sorted(set(_ORG_ALIAS) | _ALL_ORGANELLES)}")


def _norm_train_set(token: str) -> str:
    """Normalize a config-side train_set token to canonical. Raises on unknown."""
    if token in _FORWARD_TRAIN_SETS or token in _LEGACY_TRAIN_SETS:
        return token
    if token in _TRAIN_ALIAS:
        return _TRAIN_ALIAS[token]
    raise ValueError(f"unknown train_set token {token!r}")


def _leaf_suffix(test_set: str, condition: str | None) -> str:
    """Return the ``<test>[__<condition>]`` leaf segment."""
    if test_set == "a549":
        if condition is None:
            raise ValueError("A549 test set requires a condition (mock|denv|zikv)")
        if condition not in _CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; expected one of {sorted(_CONDITIONS)}")
        return f"a549__{condition}"
    if test_set == "ipsc":
        if condition is not None:
            raise ValueError("iPSC test set takes no condition")
        return "ipsc"
    raise ValueError(f"unknown test_set {test_set!r}; expected one of {sorted(_TEST_SETS)}")


# ===========================================================================
# Display helpers (retained public API)
# ===========================================================================


def paper_key(code_model: str) -> str:
    """Translate the code-side model key to its paper display name.

    Raises ``ValueError`` for a model not in the display registry.
    """
    if code_model not in PAPER_KEY:
        raise ValueError(f"unknown model key {code_model!r}; expected one of {sorted(PAPER_KEY)}")
    return PAPER_KEY[code_model]


def eval_predict_set_group(dataset_name: str) -> str:
    """Return the eval-side Hydra ``predict_set`` group name for one leaf.

    iPSC composes back to itself; A549 leaves carry the per-condition dataset slug
    ``a549-mantis-<marker>-<cond>`` and the group name uses underscores.
    """
    if dataset_name == "aics-hipsc":
        return "ipsc_confocal"
    if dataset_name.startswith("a549-mantis-"):
        return "a549_mantis_" + dataset_name.removeprefix("a549-mantis-").replace("-", "_")
    raise ValueError(
        f"cannot map dataset {dataset_name!r} to a predict_set group; "
        f"expected 'aics-hipsc' or 'a549-mantis-<marker>-<cond>'"
    )


def extract_predict_output_store(composed: dict, leaf_path: Path) -> Path:
    """Pull ``HCSPredictionWriter.init_args.output_store`` from a composed predict config."""
    callbacks = composed.get("trainer", {}).get("callbacks", [])
    if not isinstance(callbacks, list):
        raise ValueError(f"{leaf_path}: trainer.callbacks must be a list (got {type(callbacks).__name__})")
    for cb in callbacks:
        if not isinstance(cb, dict):
            continue
        if str(cb.get("class_path", "")).endswith("HCSPredictionWriter"):
            init_args = cb.get("init_args", {}) or {}
            store = init_args.get("output_store")
            if not store:
                raise ValueError(f"{leaf_path}: HCSPredictionWriter has no init_args.output_store")
            return Path(store)
    raise ValueError(f"{leaf_path}: no HCSPredictionWriter callback found under trainer.callbacks")


# ===========================================================================
# Map (c): resolve_model
# ===========================================================================


def resolve_model(benchmark: dict | None, ckpt_path: str | Path | None, leaf_path: str | Path | None = None) -> str:
    """Recover the true model code key from a checkpoint path.

    The bare ``model_name: celldiff`` in a config is NOT authoritative — the real
    variant lives in ``ckpt_path`` (``.../celldiff_r2/...``), and the documented
    rule "joint celldiff = R2" applies. Resolution order:

    1. If ``ckpt_path`` contains a ``celldiff_r2`` segment -> ``celldiff_r2``.
    2. If ``ckpt_path`` contains a bare ``celldiff`` segment and the leaf/benchmark
       is joint-trained -> ``celldiff_r2`` (joint celldiff = R2).
    3. Otherwise, if the config ``model_name`` is a recognizable code key, use it.
    4. Otherwise raise.

    Parameters
    ----------
    benchmark : dict | None
        Composed benchmark block (may carry ``model_name`` / ``train_set`` /
        ``trained_on``). May be ``None``.
    ckpt_path : str | Path | None
        Trained-checkpoint path (predict leaves).
    leaf_path : str | Path | None, optional
        The config leaf path, used as a secondary source for the joint hint.

    Returns
    -------
    str
        Model code key.
    """
    benchmark = benchmark or {}
    ckpt_parts = Path(ckpt_path).parts if ckpt_path is not None else ()
    ckpt_str = str(ckpt_path) if ckpt_path is not None else ""
    leaf_str = str(leaf_path) if leaf_path is not None else ""

    model_name = str(benchmark.get("model_name", "") or "")
    train_hint = str(benchmark.get("train_set", benchmark.get("trained_on", "")) or "")
    is_joint = (
        "joint" in train_hint
        or "joint_predictions" in ckpt_str
        or "joint_ipsc_confocal_a549_mantis" in ckpt_str
        or "joint" in leaf_str.split("/")
    )

    # CELL-Diff identity comes from the ckpt path, never the bare model_name.
    if model_name.startswith("celldiff") or "celldiff" in ckpt_str:
        if "celldiff_r2" in ckpt_parts or "celldiff_r2" in ckpt_str:
            return "celldiff_r2"
        # Bare `celldiff` segment in the ckpt path.
        if "celldiff" in ckpt_parts or model_name == "celldiff":
            if is_joint:
                return "celldiff_r2"  # joint celldiff = R2 (documented rule)
            # A bare non-joint celldiff ckpt is R1 (celldiff); keep the variant if named.
            if model_name in PAPER_KEY:
                return model_name
            return "celldiff"
        if model_name in PAPER_KEY:
            return model_name

    # Non-celldiff: prefer a recognizable ckpt-path segment, else the config name.
    for part in reversed(ckpt_parts):
        if part in PAPER_KEY:
            return part
        if part == "unext2":  # dir spelling of the distinct code model
            return "unext2_timm_scratch"
    if model_name in PAPER_KEY:
        return model_name
    if model_name == "unext2":
        # Ambiguous bare token — only safe when config carries the true key.
        raise ValueError(
            "bare model_name='unext2' is ambiguous (fcmae_vscyto3d_scratch legacy paper key vs "
            "unext2_timm_scratch code model); resolve via ckpt_path or the explicit code key"
        )
    if model_name in _MODEL_PAPER_TO_CODE:
        return _MODEL_PAPER_TO_CODE[model_name]
    raise ValueError(
        f"cannot resolve model from benchmark model_name={model_name!r}, ckpt_path={ckpt_path!r}, leaf={leaf_path!r}"
    )


# ===========================================================================
# Grammar functions (forward)
# ===========================================================================


def checkpoint_dir(organelle: str, model: str, train_set: str, models_root: str | Path = MODELS_ROOT) -> Path:
    """Return the canonical checkpoint directory.

    ``MODELS_ROOT/<train_set>/<organelle>/<model>/checkpoints``.
    """
    organelle = _norm_organelle(organelle)
    train_set = _norm_train_set(train_set)
    _validate(organelle, train_set, model)
    return Path(models_root) / train_set / organelle / model / "checkpoints"


def prediction_store(
    organelle: str,
    model: str,
    train_set: str,
    test_set: str,
    condition: str | None = None,
    data_root: str | Path = DATA_ROOT,
) -> Path:
    """Return the canonical single prediction zarr for one tuple.

    ``DATA_ROOT/<organelle>/<model>/<train_set>/<test>[__<cond>]/prediction.zarr``.
    One zarr per tuple, shared by all eval tracks and (for multi-target models)
    all components.
    """
    organelle = _norm_organelle(organelle)
    train_set = _norm_train_set(train_set)
    _validate(organelle, train_set, model)
    leaf = _leaf_suffix(test_set, condition)
    return Path(data_root) / organelle / model / train_set / leaf / "prediction.zarr"


def eval_leaf(
    organelle: str,
    model: str,
    train_set: str,
    test_set: str,
    condition: str | None = None,
    component: str | None = None,
    track: str = "default",
    gt_repr: str = "raw",
    data_root: str | Path = DATA_ROOT,
) -> Path:
    """Return the canonical eval output dir (== ``save.save_dir``).

    Layout::

        DATA_ROOT/<organelle>/<model>/<train_set>/<test>[__<cond>]/
            [<component>/]              # multi-target combined-token models only
            [instance_ap/]             # track="instance_ap"
            [deconv_gt/]               # gt_repr="deconv" (er/mito legacy comparison)

    The default track's metrics live at the leaf top (or component top);
    ``instance_ap`` and ``deconv_gt`` are subtracks below it.

    Parameters
    ----------
    organelle : str
        Canonical organelle (single-target or multi-target combined token).
    model, train_set, test_set, condition
        Tuple axes (see module docstring).
    component : str | None
        Per-component organelle for a multi-target combined-token model; REQUIRED
        for such models, and must be one of the token's components. Must be ``None``
        for single-target organelles.
    track : {"default", "instance_ap"}
        Eval track.
    gt_repr : {"raw", "deconv"}
        GT representation. ``deconv`` -> a ``deconv_gt/`` subdir (er/mito only).
    """
    organelle = _norm_organelle(organelle)
    train_set = _norm_train_set(train_set)
    _validate(organelle, train_set, model)
    if track not in {"default", "instance_ap"}:
        raise ValueError(f"unknown track {track!r}; expected 'default' or 'instance_ap'")
    if gt_repr not in {"raw", "deconv"}:
        raise ValueError(f"unknown gt_repr {gt_repr!r}; expected 'raw' or 'deconv'")
    if gt_repr == "deconv" and organelle not in {"er", "mito"}:
        raise ValueError(f"deconv GT eval is valid only for er|mito, got organelle={organelle!r}")

    leaf = _leaf_suffix(test_set, condition)
    path = Path(data_root) / organelle / model / train_set / leaf

    if organelle in _MULTI_ORGANELLES:
        if component is None:
            raise ValueError(f"multi-target organelle {organelle!r} requires a component")
        if component not in _MULTI_COMPONENTS[organelle]:
            raise ValueError(f"component {component!r} not in {organelle!r} components {_MULTI_COMPONENTS[organelle]}")
        path = path / component
    elif component is not None:
        raise ValueError(f"single-target organelle {organelle!r} takes no component (got {component!r})")

    if gt_repr == "deconv":
        path = path / "deconv_gt"
    if track == "instance_ap":
        path = path / "instance_ap"
    return path


def gt_cache_dir(
    organelle: str,
    test_set: str,
    condition: str | None = None,
    gt_repr: str = "raw",
    data_root: str | Path = DATA_ROOT,
) -> Path:
    """Return the FROZEN, target-keyed GT-feature cache dir (model-independent).

    Reproduces the on-disk case-inconsistent convention verbatim:

    - iPSC: ``ipsc/eval_cache/<SEC61B|TOMM20|nucleus|membrane>``
    - A549: ``a549/eval_cache/<gene>_<cond>[_deconv]`` (gene lowercase).

    ``gt_repr="deconv"`` emits the ``_deconv`` A549 variant (er/mito only).
    Single- and multi-target models share these caches (target-keyed).
    """
    organelle = _norm_organelle(organelle)
    if organelle in _MULTI_ORGANELLES:
        raise ValueError(f"gt_cache_dir is target-keyed; pass a component organelle, not {organelle!r}")
    if gt_repr not in {"raw", "deconv"}:
        raise ValueError(f"unknown gt_repr {gt_repr!r}; expected 'raw' or 'deconv'")
    if gt_repr == "deconv" and organelle not in {"er", "mito"}:
        raise ValueError(f"deconv GT cache is valid only for er|mito, got organelle={organelle!r}")
    root = Path(data_root)
    if test_set == "ipsc":
        if condition is not None:
            raise ValueError("iPSC GT cache takes no condition")
        if gt_repr == "deconv":
            raise ValueError("iPSC has no deconv GT cache")
        return root / "ipsc" / "eval_cache" / _IPSC_GT_CACHE_KEY[organelle]
    if test_set == "a549":
        if condition is None or condition not in _CONDITIONS:
            raise ValueError(f"A549 GT cache requires a condition in {sorted(_CONDITIONS)}, got {condition!r}")
        gene = _A549_GENE[organelle]
        name = f"{gene}_{condition}"
        if gt_repr == "deconv":
            name = f"{name}_deconv"
        return root / "a549" / "eval_cache" / name
    raise ValueError(f"unknown test_set {test_set!r}; expected one of {sorted(_TEST_SETS)}")


def pred_cache_dir(
    organelle: str,
    model: str,
    train_set: str,
    test_set: str,
    condition: str | None = None,
    track: str = "default",
    data_root: str | Path = DATA_ROOT,
) -> Path:
    """Return the canonical (regenerable) pred-side feature cache dir.

    Model-centric mirror of the eval leaf under ``eval_cache_pred``; ``instance_ap``
    features live in a distinct tree. Unlike the frozen GT cache, this is fully
    regenerable and keyed on the full tuple.
    """
    organelle = _norm_organelle(organelle)
    train_set = _norm_train_set(train_set)
    _validate(organelle, train_set, model)
    if track not in {"default", "instance_ap"}:
        raise ValueError(f"unknown track {track!r}; expected 'default' or 'instance_ap'")
    subroot = "eval_cache_pred_instance_ap" if track == "instance_ap" else "eval_cache_pred"
    leaf = _leaf_suffix(test_set, condition)
    return Path(data_root) / test_set / subroot / organelle / model / train_set / leaf


def metrics_repo_dir(
    organelle: str,
    model: str,
    train_set: str,
    test_set: str,
    condition: str | None = None,
    component: str | None = None,
    track: str = "default",
    repo_root: str | Path | None = None,
) -> Path:
    """Return the git-tracked metrics mirror dir for one leaf.

    ``applications/dynacell/results/metrics/<organelle>/<model>/<train_set>/
    <test>[__cond][/<component>][/instance_ap]``. Mirrors the DATA_ROOT eval leaf
    structure (component present only for multi-target combined tokens).
    """
    organelle = _norm_organelle(organelle)
    train_set = _norm_train_set(train_set)
    _validate(organelle, train_set, model)
    if track not in {"default", "instance_ap"}:
        raise ValueError(f"unknown track {track!r}; expected 'default' or 'instance_ap'")
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]  # .../applications/dynacell/src/dynacell/evaluation
    base = Path(repo_root) / "applications" / "dynacell" / "results" / "metrics"
    leaf = _leaf_suffix(test_set, condition)
    path = base / organelle / model / train_set / leaf
    if organelle in _MULTI_ORGANELLES:
        if component is None:
            raise ValueError(f"multi-target organelle {organelle!r} requires a component")
        if component not in _MULTI_COMPONENTS[organelle]:
            raise ValueError(f"component {component!r} not in {organelle!r} components")
        path = path / component
    elif component is not None:
        raise ValueError(f"single-target organelle {organelle!r} takes no component")
    if track == "instance_ap":
        path = path / "instance_ap"
    return path


def iter_organelle_evals(
    organelle: str,
    models: list[str],
    train_sets: list[str],
    test_set: str,
    condition: str | None = None,
    track: str = "default",
    gt_repr: str = "raw",
    data_root: str | Path = DATA_ROOT,
) -> list[Path]:
    """Yield every eval dir for one component ``organelle`` across model arities.

    Spans both single-target leaves (``<organelle>/<model>/...``) AND the matching
    component subdir of every multi-target combined-token leaf that contains this
    organelle (``dual_nucleus_membrane/<model>/.../<organelle>/``) — so the paper's
    "all <organelle> models" query stays one call.

    Parameters
    ----------
    organelle : str
        Canonical single-target organelle to collect (nucleus/membrane/er/mito).
    models : list[str]
        Model code keys to enumerate.
    train_sets : list[str]
        Train-set tokens to enumerate.
    test_set, condition, track, gt_repr, data_root
        Passed through to :func:`eval_leaf`.

    Returns
    -------
    list[Path]
        Eval dirs for valid (model, train_set) combinations; single-target first,
        then each multi-target combined token that includes this organelle.
    """
    organelle = _norm_organelle(organelle)
    if organelle not in _SINGLE_ORGANELLES:
        raise ValueError(f"iter_organelle_evals expects a single-target organelle, got {organelle!r}")
    out: list[Path] = []
    for model in models:
        for train_set in train_sets:
            train_norm = _norm_train_set(train_set)
            # Single-target leaf.
            if _tuple_is_valid(organelle, train_norm, model):
                out.append(
                    eval_leaf(
                        organelle,
                        model,
                        train_norm,
                        test_set,
                        condition,
                        track=track,
                        gt_repr=gt_repr,
                        data_root=data_root,
                    )
                )
            # Multi-target combined tokens that include this organelle, as a component.
            for combined, components in _MULTI_COMPONENTS.items():
                if organelle not in components:
                    continue
                if not _tuple_is_valid(combined, train_norm, model):
                    continue
                out.append(
                    eval_leaf(
                        combined,
                        model,
                        train_norm,
                        test_set,
                        condition,
                        component=organelle,
                        track=track,
                        gt_repr=gt_repr,
                        data_root=data_root,
                    )
                )
    return out


# ===========================================================================
# normalize_legacy — recognize every pre-canonical on-disk form
# ===========================================================================

# Zarr organelle prefixes -> canonical (ordered longest-first for _starts_with).
_ZARR_ORG_PREFIX: dict[str, str] = {
    "sec61b": "er",
    "tomm20": "mito",
    "nucl": "nucleus",
    "memb": "membrane",
    "dual_nucl_memb": "dual_nucleus_membrane",
}

# Legacy eval-parent dir -> (train_set, gt_repr default handled per-parent).
# For the *_with_embeddings families the train_set derives from the infix, not
# the parent alone; these map (test_set, parent) -> train_set for the
# ipsc-trained / a549trained / jointtrained triad.
_EVAL_PARENT_TRAIN_SET: dict[str, str] = {
    "evaluations_with_embeddings": "ipsc",
    "evaluations_a549trained_with_embeddings": "a549",
    "evaluations_jointtrained_with_embeddings": "joint",
}
# Ablation eval parents -> (model-suffix-ablation-token). These carry
# fcmae_vscyto3d_pretrained_<token> models with the closed legacy train_set token.
_ABLATION_EVAL_PARENTS: dict[str, str] = {
    "evaluations_randinit": "randinit",
    "evaluations_cytoland": "cytoland",
    "evaluations_infectionft": "infectionft",
}

# Legacy eval-dir-name model infixes (paper display keys) -> code model. These are
# the tokens that appear in `eval_<key>[_<infix>]_<organelle>[_<cond>]` dir names.
_EVAL_NAME_MODEL_TO_CODE: dict[str, str] = {
    "unext2": "fcmae_vscyto3d_scratch",  # legacy paper key in eval dir names = UNeXt2
    "vscyto3d": "fcmae_vscyto3d_pretrained",
    "fnet3d": "fnet3d_paper",
    "unetvit3d": "unetvit3d",
    "pix2pix3d": "pix2pix3d_unetvit",
    "celldiff_r2": "celldiff_r2",
    "celldiff_r2_iterative": "celldiff_r2_iterative",
    "celldiff_r2_sliding_window": "celldiff_r2_sliding_window",
    "celldiff_r2_denoise": "celldiff_r2_denoise",
    "vscyto3d_randinit": "fcmae_vscyto3d_pretrained_randinit",
    "vscyto3d_cytoland": "fcmae_vscyto3d_pretrained_cytoland",
    "vscyto3d_infectionft": "fcmae_vscyto3d_pretrained_infectionft",
    "vscyto3d_cytolandft": "vscyto3d_cytolandft",
    "vscyto3d_infectionft_dynacellft": "vscyto3d_infectionft_dynacellft",
}

# Paper organelle token -> canonical (in eval dir names).
_PAPER_ORG_TO_CANONICAL: dict[str, str] = {
    "nucleus": "nucleus",
    "membrane": "membrane",
    "er": "er",
    "mitochondria": "mito",
    "dual_nucleus_membrane": "dual_nucleus_membrane",
}

# Deliberately-skipped legacy filenames (stale / alias-dup / probe fixtures) — these
# are recognized as SKIP (return None -> UNMAPPED) so callers surface them as
# archive decision rows rather than guessing a canonical home.
_SKIP_ZARR_FILENAMES: frozenset[str] = frozenset(
    {
        "memb_celldiff_mock.zarr",
        "memb_celldiff_denv.zarr",
        "memb_celldiff_zikv.zarr",
        "sec61b_fnet3d.zarr",  # alias-dup of sec61b_fnet3d_paper
        "sec61b_unext2.zarr",  # alias-dup of sec61b_fcmae_vscyto3d_scratch
    }
)


def _starts_with_longest(stem: str, prefixes: dict[str, str]) -> tuple[str | None, str, str | None]:
    """Return ``(matched_prefix, remainder, canonical)`` for the longest matching prefix."""
    for prefix in sorted(prefixes, key=len, reverse=True):
        token = f"{prefix}_"
        if stem.startswith(token):
            return prefix, stem[len(token) :], prefixes[prefix]
    return None, stem, None


def _split_model_variant_legacy(body: str) -> str | None:
    """Resolve a legacy zarr ``body`` (model[_variant]) into a canonical model key.

    Handles celldiff (R1/R2) + variants and the deterministic model set. Returns
    ``None`` if unrecognized.
    """
    celldiff_models = ("celldiff_r2", "celldiff")  # r2 first so longest wins
    variants = ("iterative", "sliding_window", "denoise")
    for cd in celldiff_models:
        if body == cd:
            return cd
        token = f"{cd}_"
        if body.startswith(token):
            variant = body[len(token) :]
            if variant in variants:
                return f"{cd}_{variant}"
            return None
    deterministic = (
        "fcmae_vscyto3d_scratch",
        "fcmae_vscyto3d_pretrained",
        "fnet3d_paper",
        "unetvit3d",
        "pix2pix3d_unetvit",
        "unext2_timm_scratch",
    )
    if body in deterministic:
        return body
    return None


def _normalize_prediction_zarr(path: Path, data_root: Path) -> CanonicalKey | None:
    """normalize_legacy for a legacy prediction zarr path. Returns None on SKIP/UNMAPPED."""
    name = path.name
    if not name.endswith(".zarr"):
        return None
    if name in _SKIP_ZARR_FILENAMES:
        return None
    stem = name[: -len(".zarr")]

    rel = path.relative_to(data_root)
    parts = rel.parts
    if parts[0] == "ipsc":
        test_set = "ipsc"
    elif parts[0] == "a549":
        test_set = "a549"
    else:
        return None
    subdir = parts[1]
    if subdir in ("joint_predictions", "joint_predictions_v2"):
        in_joint_dir = True
    elif subdir == "predictions":
        in_joint_dir = False
    else:
        return None

    # Ablation / dual-track skip families (own eval track, not the standard grammar).
    # dual_* IS modeled here (multi-target), so only ablation infixes skip.
    org_prefix, after_org, organelle = _starts_with_longest(stem, _ZARR_ORG_PREFIX)
    if org_prefix is None or organelle is None:
        return None

    condition: str | None = None
    train_set_infix: str | None = None
    gt_repr = "raw"

    if test_set == "ipsc":
        body, train_set_infix = _strip_suffix(after_org, ("jointtrained", "a549trained"))
    else:
        body = after_org
        if "__" in body:
            # Legacy ER/mito iPSC-trained double-underscore form:
            # <org>_<model>[_a549trained|_jointtrained]__<gene>_<cond>
            left, _, right = body.partition("__")
            if right.count("_") != 1:
                return None
            gene_token, cond_token = right.split("_", 1)
            if cond_token not in _CONDITIONS:
                return None
            condition = cond_token
            body, train_set_infix = _strip_suffix(left, ("jointtrained", "a549trained"))
        else:
            body, cond_token = _strip_suffix(body, tuple(_CONDITIONS))
            if cond_token is None:
                return None
            condition = cond_token
            body, train_set_infix = _strip_suffix(body, ("jointtrained", "a549trained"))

    # Ablation model infix in the body -> skip (own track).
    if any(tok in body for tok in ("_randinit", "_cytoland", "_infectionft")):
        return None

    model = _split_model_variant_legacy(body)
    if model is None:
        return None

    # Resolve train_set from directory + infix.
    if in_joint_dir or train_set_infix == "jointtrained":
        train_set = "joint"
    elif train_set_infix == "a549trained":
        train_set = "a549"
    elif train_set_infix is None:
        # Bare (no infix) = iPSC-trained (bare-a549 rule: a bare a549-dir zarr with
        # no infix is an iPSC-trained model tested on A549).
        train_set = "ipsc"
    else:
        return None

    # Deconv-provenance rule: pre-flip ER/mito artifacts trained on A549/joint carry
    # deconv provenance (their training GT was deconv). Relabel train_set.
    if organelle in ("er", "mito"):
        if train_set == "a549":
            train_set = "a549__deconv"
        elif train_set == "joint":
            train_set = _LEGACY_JOINT_DECONV

    if not _tuple_is_valid(organelle, train_set, model):
        return None
    return CanonicalKey(
        organelle=organelle,
        model=model,
        train_set=train_set,
        test_set=test_set,
        condition=condition,
        component=None,
        track="default",
        gt_repr=gt_repr,
    )


def _strip_suffix(stem: str, candidates: tuple[str, ...]) -> tuple[str, str | None]:
    """Return ``(prefix, matched)`` if any candidate is a trailing ``_<x>`` (longest wins)."""
    for cand in sorted(candidates, key=len, reverse=True):
        token = f"_{cand}"
        if stem.endswith(token):
            return stem[: -len(token)], cand
    return stem, None


def _normalize_eval_dir(path: Path, data_root: Path) -> CanonicalKey | None:
    """normalize_legacy for a legacy eval dir (eval_<key>[_<infix>]_<organelle>[_<cond>])."""
    rel = path.relative_to(data_root)
    parts = rel.parts
    if parts[0] not in ("ipsc", "a549"):
        return None
    test_set = parts[0]
    parent = parts[1]
    name = parts[2] if len(parts) >= 3 else None
    if name is None or not name.startswith("eval_"):
        return None
    # Track: a trailing /instance_ap or the evaluations_instance_ap parent.
    track = "default"
    if parent == "evaluations_instance_ap" or (len(parts) >= 4 and parts[3] == "instance_ap"):
        track = "instance_ap"

    # Determine train_set + gt_repr from the parent family.
    if parent in _EVAL_PARENT_TRAIN_SET:
        base_train = _EVAL_PARENT_TRAIN_SET[parent]
    elif parent == "evaluations_instance_ap":
        base_train = None  # derive from the name infix below
    elif parent in _ABLATION_EVAL_PARENTS:
        base_train = "ablation"
    else:
        # Stale pre-2D parents (evaluations, joint_evaluations, *_temp, *_v2, ...) -> UNMAPPED.
        return None

    body = name[len("eval_") :]

    # Ablation eval dir: eval_vscyto3d_<abltoken>_<organelle> -> ablation model + legacy train_set.
    if parent in _ABLATION_EVAL_PARENTS:
        abltoken = _ABLATION_EVAL_PARENTS[parent]
        organelle = _last_organelle_token(body)
        if organelle is None:
            return None
        model = f"fcmae_vscyto3d_pretrained_{abltoken}"
        return CanonicalKey(
            organelle=organelle,
            model=model,
            train_set=abltoken,
            test_set=test_set,
            condition=None,
            component=None,
            track=track,
            gt_repr="raw",
        )

    # Strip trailing condition for A549.
    condition: str | None = None
    if test_set == "a549":
        body, cond = _strip_suffix(body, tuple(_CONDITIONS))
        if cond is None:
            return None
        condition = cond

    # Strip trailing organelle token.
    organelle, body_wo_org = _strip_last_organelle(body)
    if organelle is None:
        return None

    # Now body_wo_org = <model_key>[_<infix>]. Strip train-set infix.
    body_wo_infix, infix = _strip_suffix(body_wo_org, ("jointtrained", "a549trained"))
    if base_train is None:
        # instance_ap parent: derive train_set from infix.
        if infix == "jointtrained":
            train_set = "joint"
        elif infix == "a549trained":
            train_set = "a549"
        else:
            train_set = "ipsc"
    else:
        train_set = base_train

    model = _EVAL_NAME_MODEL_TO_CODE.get(body_wo_infix)
    if model is None:
        # Some celldiff dirs carry the bare celldiff_r2 with a variant already folded.
        model = _split_model_variant_legacy(body_wo_infix)
    if model is None:
        return None

    # Deconv-provenance for ER/mito A549/joint evals.
    gt_repr = "raw"
    if organelle in ("er", "mito"):
        if train_set == "a549":
            train_set = "a549__deconv"
        elif train_set == "joint":
            train_set = _LEGACY_JOINT_DECONV
        # iPSC-trained ER/mito evaluated against deconv A549 GT (case b): the model
        # train_set stays ipsc, but the eval was scored against deconv GT.
        elif train_set == "ipsc" and test_set == "a549":
            gt_repr = "deconv"

    if not _tuple_is_valid(organelle, train_set, model):
        return None
    return CanonicalKey(
        organelle=organelle,
        model=model,
        train_set=train_set,
        test_set=test_set,
        condition=condition,
        component=None,
        track=track,
        gt_repr=gt_repr,
    )


def _last_organelle_token(body: str) -> str | None:
    """Return the canonical organelle if ``body`` ends with a paper organelle token."""
    org, _ = _strip_last_organelle(body)
    return org


def _strip_last_organelle(body: str) -> tuple[str | None, str]:
    """Strip a trailing paper organelle token, returning ``(canonical, remainder)``."""
    for paper_org in sorted(_PAPER_ORG_TO_CANONICAL, key=len, reverse=True):
        token = f"_{paper_org}"
        if body.endswith(token):
            return _PAPER_ORG_TO_CANONICAL[paper_org], body[: -len(token)]
    return None, body


def normalize_legacy(path: str | Path, data_root: str | Path = DATA_ROOT) -> CanonicalKey | None:
    """Map a legacy on-disk artifact path to its :class:`CanonicalKey`.

    Recognizes prediction zarrs (``predictions``/``joint_predictions``, modern +
    the ``__<gene>_<cond>`` double-underscore form + bare-a549=iPSC-trained + the
    ER/mito deconv-provenance rule) and eval dirs (the ``*_with_embeddings``
    triad, ``evaluations_instance_ap``, and the ablation parents). Deconv
    provenance: pre-flip ER/mito A549 artifacts -> ``a549__deconv``; ER/mito joint
    -> ``joint__legacy_deconvgt``; iPSC-trained ER/mito A549 *evals* ->
    ``gt_repr="deconv"``.

    Returns ``None`` (caller -> UNMAPPED) for a deliberately-skipped stale/alias-dup
    filename, a stale pre-2D eval parent, an ablation/dual track this grammar does
    not fold, or anything it cannot classify — never a guess.
    """
    path = Path(path)
    data_root = Path(data_root)
    try:
        rel = path.relative_to(data_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    subdir = parts[1]
    if subdir in ("predictions", "joint_predictions", "joint_predictions_v2"):
        return _normalize_prediction_zarr(path, data_root)
    if subdir.startswith("evaluations") or subdir == "joint_evaluations":
        return _normalize_eval_dir(path, data_root)
    return None
