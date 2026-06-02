"""Recompute and patch FSC columns in eval ``pixel_metrics.csv`` files.

Why
---
VisCy commit ``92ef5c1f`` dropped the outer ``target - target.mean()`` /
``prediction - prediction.mean()`` before ``cubic.metrics.fsc_resolution`` on
the (wrong) assumption that cubic mean-centers internally before the FFT. It
does mean-center, but only *after* Hamming-windowing — so on volumes with a
large DC offset (raw fluorescence, mean in the thousands) the window multiplies
the offset into a huge low-frequency artifact that shifts the FSC=0.143
crossing. Result: ``XY_FSC_Resolution`` blew up 5-14x on the high-intensity FOVs
in every eval run after that commit.

With the cubic-side fix (mean-center before windowing), the correct call is the
raw-array one the pipeline already makes. This tool recomputes *only* the two
FSC columns through the pipeline's own ``compute_pixel_metrics`` (identical code
path) and patches them in place, leaving PCC/SSIM/NRMSE/PSNR/Spectral_PCC/
MicroSSIM and every mask/feature/embedding artifact byte-identical.

Prerequisite
------------
Run in a venv with the **fixed** cubic installed (the script prints the cubic
version it used). FSC is otherwise recomputed faithfully but still wrong.

Mapping
-------
Each eval dir needs (pred zarr + channel, GT zarr + channel, spacing). The
grouped benchmark configs
(``configs/benchmarks/.../leaf/grouped/*/eval_grouped.yaml``) give an
authoritative ``save_dir -> (dataset, target, pred_path)`` for the conditions
they cover. For dirs they do not cover, the naming atoms (organelle->prefix,
(model, pool)->pred-token, (cell_type, pool)->pred-root, condition suffix) are
*learned* from the covered dirs and composed, then verified against on-disk
zarrs. GT path/channel/spacing always come from ``resolve_dataset_ref`` (the
same manifest resolution the pipeline uses); ``pred_channel = f"{target_channel}
_prediction"``. Any dir that cannot be mapped or whose pred zarr is missing is
reported and skipped — never guessed.

Usage
-----
Dry run (default; prints mapping + sample old->new per dir, writes nothing)::

    cd /hpc/mydata/alex.kalinin/VisCy
    uv run python applications/dynacell/tools/recompute_fsc.py

Apply (backs up each CSV to ``pixel_metrics.csv.prefsc_bak`` once, then patches)::

    uv run python applications/dynacell/tools/recompute_fsc.py --apply
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cubic
import numpy as np
import pandas as pd
import yaml
from iohub.ngff import open_ome_zarr

from dynacell.data.manifests import DatasetRef
from dynacell.data.resolver import resolve_dataset_ref
from dynacell.evaluation.metrics import compute_pixel_metrics

# --- Defaults --------------------------------------------------------------

DATA_ROOT = Path("/hpc/projects/virtual_staining/training/dynacell")

# The new-schema eval roots (post-torch-fidelity). The old `evaluations/` /
# `joint_evaluations/` roots were produced *before* 92ef5c1f, so their outer
# mean-center was still present and their FSC is already correct; they are also
# being discarded for the torch-fidelity reason. Pass --roots to override.
DEFAULT_ROOTS = [
    DATA_ROOT / "ipsc/evaluations_with_embeddings",
    DATA_ROOT / "ipsc/evaluations_a549trained_with_embeddings",
    DATA_ROOT / "ipsc/evaluations_jointtrained_with_embeddings",
    DATA_ROOT / "a549/evaluations_with_embeddings",
    DATA_ROOT / "a549/evaluations_a549trained_with_embeddings",
    DATA_ROOT / "a549/evaluations_jointtrained_with_embeddings",
]

GROUPED_GLOB = (
    Path(__file__).resolve().parents[1]
    / "configs/benchmarks/virtual_staining/_internal/leaf/grouped/*/eval_grouped.yaml"
)

ORGANELLES = ("membrane", "nucleus", "er", "mitochondria", "sec61b", "tomm20")
ORG_CANON = {"sec61b": "er", "tomm20": "mitochondria"}
CONDITIONS = ("mock", "denv", "zikv")
POOL_TOKENS = {"a549trained": "a549trained", "jointtrained": "jointtrained"}

XY_COL = "XY_FSC_Resolution"
Z_COL = "Z_FSC_Resolution"

# (cell_type, canonical organelle) -> (a549 marker / ipsc target). Used to build
# the dataset_ref for dirs not covered by a grouped config. Cross-checked
# against every covered grouped dataset_ref; a conflict aborts.
A549_MARKER = {"membrane": "caax", "nucleus": "h2b", "er": "sec61b", "mitochondria": "tomm20"}
IPSC_TARGET = {"membrane": "membrane", "nucleus": "nucleus", "er": "sec61b", "mitochondria": "tomm20"}


# --- Eval-dir name parsing -------------------------------------------------


@dataclass(frozen=True)
class EvalKey:
    """Decoded ``eval_<...>`` directory name."""

    model_key: str  # e.g. "fnet3d", "celldiff_r2_iterative", "unext2"
    pool: str  # "ipsc_trained" | "a549trained" | "jointtrained"
    organelle: str  # canonical: membrane | nucleus | er | mitochondria
    condition: str | None  # mock | denv | zikv | None (iPSC-test)

    @property
    def cell_type(self) -> str:
        """``"a549"`` for infection-conditioned dirs, else ``"ipsc"`` (iPSC-test)."""
        return "a549" if self.condition is not None else "ipsc"


def parse_eval_dir(name: str) -> EvalKey | None:
    """Parse ``eval_<...>`` basename into an :class:`EvalKey`, or None if unparseable."""
    if not name.startswith("eval_"):
        return None
    rest = name[len("eval_") :]

    condition = None
    for c in CONDITIONS:
        if rest.endswith(f"_{c}"):
            condition = c
            rest = rest[: -len(f"_{c}")]
            break

    organelle = None
    for org in ORGANELLES:
        if rest.endswith(f"_{org}"):
            organelle = ORG_CANON.get(org, org)
            rest = rest[: -len(f"_{org}")]
            break
    if organelle is None:
        return None

    pool = "ipsc_trained"
    for tok, poolname in POOL_TOKENS.items():
        if rest.endswith(f"_{tok}"):
            pool = poolname
            rest = rest[: -len(f"_{tok}")]
            break

    model_key = rest
    if not model_key:
        return None
    return EvalKey(model_key=model_key, pool=pool, organelle=organelle, condition=condition)


# --- Mapping (authoritative grouped configs + learned atoms) ---------------


@dataclass
class Atoms:
    """Naming atoms learned from grouped-config-covered dirs."""

    direct: dict[str, tuple[str, str, str]] = field(default_factory=dict)  # norm(save_dir)->(ds,tgt,pred)
    org_prefix: dict[str, str] = field(default_factory=dict)  # organelle->"memb"/"nucl"/...
    token: dict[tuple[str, str], str] = field(default_factory=dict)  # (model_key,pool)->pred token
    roots: dict[str, list[str]] = field(default_factory=dict)  # cell_type->[pred dirs seen]
    ds_target: dict[tuple[str, str, str | None], tuple[str, str]] = field(default_factory=dict)


def learn_atoms() -> Atoms:
    """Build the direct mapping + naming atoms from the grouped eval configs."""
    atoms = Atoms()
    files = glob.glob(str(GROUPED_GLOB))
    for g in files:
        doc = yaml.safe_load(open(g))
        for cond in doc.get("conditions", []):
            save_dir = cond.get("save", {}).get("save_dir")
            pred_path = cond.get("io", {}).get("pred_path")
            ref = cond.get("benchmark", {}).get("dataset_ref", {})
            if not (save_dir and pred_path and ref.get("dataset") and ref.get("target")):
                continue
            ds, tgt = ref["dataset"], ref["target"]
            atoms.direct[os.path.normpath(save_dir)] = (ds, tgt, pred_path)

            key = parse_eval_dir(os.path.basename(save_dir))
            if key is None:
                continue
            pred_base = os.path.basename(pred_path)
            if not pred_base.endswith(".zarr"):
                continue
            prefix, _, remainder = pred_base[: -len(".zarr")].partition("_")
            if key.condition and remainder.endswith(f"_{key.condition}"):
                remainder = remainder[: -len(f"_{key.condition}")]
            # ER/Mito predictions carry a doubled ``__{marker}`` infix
            # (e.g. ``sec61b_fnet3d_paper__sec61b_zikv``). That marker is a
            # prediction-naming artifact, not part of the model token — strip it
            # so the token is consistent across organelles. resolve_pred re-adds
            # the ``__{marker}`` variant as a candidate.
            token = remainder.split("__", 1)[0]
            atoms.org_prefix.setdefault(key.organelle, prefix)
            atoms.token.setdefault((key.model_key, key.pool), token)
            pred_root = os.path.dirname(pred_path)
            roots = atoms.roots.setdefault(key.cell_type, [])
            if pred_root not in roots:
                roots.append(pred_root)
            atoms.ds_target.setdefault((key.cell_type, key.organelle, key.condition), (ds, tgt))
    return atoms


def derived_ds_target(key: EvalKey) -> tuple[str, str]:
    """Build the (dataset, target) ref for a dir not covered by a grouped config."""
    if key.cell_type == "ipsc":
        return "aics-hipsc", IPSC_TARGET[key.organelle]
    marker = A549_MARKER[key.organelle]
    return f"a549-mantis-{marker}-{key.condition}", marker


def resolve_pred(key: EvalKey, atoms: Atoms) -> Path | None:
    """Compose the pred zarr path from learned atoms; return the first that exists.

    The prediction-zarr naming is not fully consistent: membrane/nucleus use the
    plain ``{prefix}_{token}{cond}.zarr``, but the iPSC-trained ER/Mito A549
    predictions carry a doubled ``__{marker}`` infix
    (e.g. ``tomm20_fnet3d_paper__tomm20_denv.zarr``). We try the plain form first,
    then the doubled-marker form, and return the first existing path. Returns
    None if none exist (caller reports + skips — never guesses).
    """
    prefix = atoms.org_prefix.get(key.organelle)
    token = atoms.token.get((key.model_key, key.pool))
    roots = atoms.roots.get(key.cell_type, [])
    if not (prefix and token and roots):
        return None
    cond = f"_{key.condition}" if key.condition else ""
    marker = A549_MARKER.get(key.organelle, "")
    names = [
        f"{prefix}_{token}{cond}.zarr",
        f"{prefix}_{token}__{marker}{cond}.zarr",
    ]
    # The pool token is baked into ``token`` (e.g. ``..._jointtrained``) and the
    # condition into ``cond``, so a name is unambiguous across roots — but the
    # same pool's predictions can be split across roots (ER jointtrained preds
    # landed in predictions/ while membrane/nucleus/mito went to
    # joint_predictions/). Search every root seen for this cell type.
    for root in roots:
        for name in names:
            cand = Path(root) / name
            if cand.exists():
                return cand
    return None


@dataclass
class Job:
    """Everything needed to recompute one eval dir's FSC columns."""

    eval_dir: Path
    pred_path: Path
    pred_channel: str
    gt_path: Path
    gt_channel: str
    spacing: list[float]


def build_job(eval_dir: Path, atoms: Atoms) -> tuple[Job | None, str]:
    """Return (job, reason). job is None when the dir can't be mapped."""
    norm = os.path.normpath(str(eval_dir))
    key = parse_eval_dir(eval_dir.name)
    if key is None:
        return None, "unparseable-name"

    if norm in atoms.direct:
        ds, tgt, pred = atoms.direct[norm]
        pred_path = Path(pred)
    else:
        pred_path = resolve_pred(key, atoms)
        if pred_path is None:
            return None, "pred-zarr-not-found"
        ds, tgt = derived_ds_target(key)
        # Cross-check against any learned dataset_ref for the same key.
        learned = atoms.ds_target.get((key.cell_type, key.organelle, key.condition))
        if learned is not None and learned != (ds, tgt):
            return None, f"dataset_ref-conflict learned={learned} derived={(ds, tgt)}"

    try:
        resolved = resolve_dataset_ref(DatasetRef(dataset=ds, target=tgt))
    except Exception as e:  # noqa: BLE001 - report and skip, never guess
        return None, f"resolve_dataset_ref-failed: {type(e).__name__}: {e}"

    gt_channel = resolved.target_channel
    job = Job(
        eval_dir=eval_dir,
        pred_path=pred_path,
        pred_channel=f"{gt_channel}_prediction",
        gt_path=Path(resolved.data_path_test),
        gt_channel=gt_channel,
        spacing=resolved.spacing.as_list(),
    )
    return job, "ok"


# --- FSC recompute ---------------------------------------------------------


def recompute_dir(job: Job, *, bin_delta: int, use_gpu: bool) -> pd.DataFrame:
    """Return the dir's pixel_metrics.csv with XY/Z FSC columns recomputed."""
    csv_path = job.eval_dir / "pixel_metrics.csv"
    df = pd.read_csv(csv_path)
    if XY_COL not in df.columns or Z_COL not in df.columns:
        raise RuntimeError(f"{csv_path}: missing FSC columns")

    fsc_kwargs = {"bin_delta": bin_delta}

    with (
        open_ome_zarr(job.pred_path, mode="r") as pred_plate,
        open_ome_zarr(job.gt_path, mode="r") as gt_plate,
    ):
        for fov, idx in df.groupby("FOV").groups.items():
            pos_pred = pred_plate[str(fov)]
            pos_gt = gt_plate[str(fov)]
            pci = pos_pred.get_channel_index(job.pred_channel)
            gci = pos_gt.get_channel_index(job.gt_channel)
            predict = np.asarray(pos_pred.data[:, pci])  # (T, D, H, W)
            target = np.asarray(pos_gt.data[:, gci])
            for row in idx:
                t = int(df.loc[row, "Timepoint"])
                m = compute_pixel_metrics(
                    predict[t],
                    target[t],
                    spacing=job.spacing,
                    fsc_kwargs=fsc_kwargs,
                    spectral_pcc_kwargs=None,
                    use_gpu=use_gpu,
                )
                df.loc[row, XY_COL] = float(m[XY_COL])
                df.loc[row, Z_COL] = float(m[Z_COL])

    return df


# --- Driver ----------------------------------------------------------------


def main() -> None:
    """CLI entry point: map eval dirs, recompute FSC, report or patch."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS, help="Eval roots to scan.")
    ap.add_argument("--apply", action="store_true", help="Write patched CSVs (default: dry run).")
    ap.add_argument("--bin-delta", type=int, default=5, help="cubic fsc_resolution bin_delta (eval.yaml default 5).")
    ap.add_argument("--no-gpu", action="store_true", help="Force CPU FSC.")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N eval dirs (debug).")
    ap.add_argument("--map-only", action="store_true", help="Resolve + report mapping only; no FSC recompute.")
    args = ap.parse_args()

    use_gpu = not args.no_gpu
    mode = "APPLY (writing)" if args.apply else "DRY RUN (no writes)"
    print(f"cubic version: {cubic.__version__}")
    print(f"mode: {mode}  use_gpu={use_gpu}  bin_delta={args.bin_delta}")

    atoms = learn_atoms()
    print(
        f"learned: {len(atoms.direct)} direct mappings, "
        f"{len(atoms.token)} (model,pool) tokens, {len(atoms.org_prefix)} organelle prefixes\n"
    )

    eval_dirs: list[Path] = []
    for root in args.roots:
        if not root.is_dir():
            print(f"  [warn] root missing: {root}")
            continue
        for sub in sorted(root.iterdir()):
            if (sub / "pixel_metrics.csv").is_file():
                eval_dirs.append(sub)
    if args.limit:
        eval_dirs = eval_dirs[: args.limit]
    print(f"found {len(eval_dirs)} eval dirs with pixel_metrics.csv\n")

    patched, skipped = 0, []
    for ed in eval_dirs:
        job, reason = build_job(ed, atoms)
        rel = str(ed).replace(str(DATA_ROOT) + "/", "")
        if job is None:
            skipped.append((rel, reason))
            print(f"  [SKIP] {rel}  ({reason})")
            continue

        if args.map_only:
            print(
                f"  [ MAP] {rel}  pred={job.pred_path.name}  gt={job.gt_path.name}:{job.gt_channel}  "
                f"spacing={job.spacing}"
            )
            patched += 1
            continue
        try:
            df_old = pd.read_csv(ed / "pixel_metrics.csv")
            df_new = recompute_dir(job, bin_delta=args.bin_delta, use_gpu=use_gpu)
        except Exception as e:  # noqa: BLE001 - report and continue
            skipped.append((rel, f"recompute-failed: {type(e).__name__}: {e}"))
            print(f"  [FAIL] {rel}  ({type(e).__name__}: {e})")
            continue

        old_xy = df_old[XY_COL].to_numpy(dtype=float)
        new_xy = df_new[XY_COL].to_numpy(dtype=float)
        d = np.abs(new_xy - old_xy)
        print(
            f"  [ OK ] {rel}  n={len(df_new)}  "
            f"XY mean {np.nanmean(old_xy):.3f}->{np.nanmean(new_xy):.3f}  max|Δ|={np.nanmax(d):.3f}  "
            f"pred={job.pred_path.name}"
        )

        if args.apply:
            bak = ed / "pixel_metrics.csv.prefsc_bak"
            if not bak.exists():
                shutil.copy2(ed / "pixel_metrics.csv", bak)
            df_new.to_csv(ed / "pixel_metrics.csv", index=False)
        patched += 1

    print(f"\n{'patched' if args.apply else 'would patch'}: {patched}   skipped: {len(skipped)}")
    if skipped:
        print("skipped dirs:")
        for rel, reason in skipped:
            print(f"  {rel}  <- {reason}")


if __name__ == "__main__":
    main()
