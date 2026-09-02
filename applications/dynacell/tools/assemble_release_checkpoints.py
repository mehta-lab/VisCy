#!/usr/bin/env python3
"""Assemble (and later update) the DynaCell v1 public checkpoint zoo.

This tool builds the ``models/`` tree of the public S3 mirror from the trained
checkpoints on ``/hpc/projects/comp.micro``. It is **idempotent and
re-runnable** by design: the ER/Mito A549+Joint models are trained on
*deconvolved* fluorescence in v1 and will be retrained on *raw* fluorescence;
when those raw runs finish and their ``predict__*.yml`` configs are re-pinned,
re-running this tool with ``--execute`` overwrites exactly those leaves and
refreshes the manifest. See ``RELEASING_CHECKPOINTS.md``.

Source of truth
---------------
The pinned checkpoint per (train_set, organelle, model) is read from the
canonical benchmark ``predict__*.yml`` configs (``ckpt_path:``), NOT from any
hand-maintained table, so the release always matches what the evals actually
consumed. Only the canonical per-organelle config dirs are scanned
(``er/ membrane/ mito/ nucleus/``); ``_dual_nucl_memb/`` (ablations) and
``_internal/`` (generated leaves) are excluded.

Public layout
-------------
::

    <dest>/models/
        checkpoints.csv                                   # manifest (written on --execute)
        {ipsc,a549,joint}/{nucleus,membrane,er,mito}/{model_slug}/
            epoch=NNN-step=MMMM.ckpt                       # original filename preserved
            config.yaml                                    # fit config (architecture)

Checkpoint filenames are preserved verbatim (``epoch=NNN-step=MMMM.ckpt``) so
file provenance survives even if the docs and the files diverge. ``last.ckpt``
and ``best_ep*.ckpt`` pins are resolved back to their canonical ``epoch=`` name.

Status per cell
---------------
- ``resolved``    -- pinned checkpoint exists -> copied.
- ``pending``     -- model dir exists but the pinned epoch is gone (mid-retrain /
  awaiting re-pin, e.g. the deconv->raw ER/Mito campaign) -> NOT copied, listed.
- ``not_trained`` -- model dir absent (stale config for a model never trained)
  -> dropped.

Usage
-----
::

    python assemble_release_checkpoints.py                 # dry run (default)
    python assemble_release_checkpoints.py --execute       # copy + write manifest
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

# --- fixed locations -------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]  # .../VisCy
CONFIG_ROOT = REPO / "applications/dynacell/configs/benchmarks/virtual_staining"
CANONICAL_ORGANELLE_DIRS = ("er", "membrane", "mito", "nucleus")
DEFAULT_DEST = Path("/hpc/projects/virtual_staining/dynacell_v1")

# --- source -> public taxonomy --------------------------------------------
# Both path grammars: the canonical post-migration tokens (#480 re-tokenized every
# leaf onto the paths.py grammar) and the pre-migration ones that a few un-migrated
# movie leaves still pin. These dicts double as the public-release allowlist -- the
# ablation train sets (a549__bf, ipsc__bf, randinit) are deliberately absent, which
# is why this does not normalize through paths._TRAIN_ALIAS/_ORG_ALIAS.
TRAIN_PUB = {
    "ipsc": "ipsc",
    "a549": "a549",
    "joint": "joint",
    "a549_mantis": "a549",
    "joint_ipsc_confocal_a549_mantis": "joint",
}
GENE_PUB = {
    "nucleus": "nucleus",
    "membrane": "membrane",
    "er": "er",
    "mito": "mito",
    "nucl": "nucleus",
    "memb": "membrane",
    "sec61b": "er",
    "tomm20": "mito",
}
MODEL_PUB = {
    "fnet3d_paper": "fnet3d",
    "fcmae_vscyto3d_scratch": "unext2",
    "fcmae_vscyto3d_pretrained": "vscyto3d",
    "fcmae_vscyto3d_pretrained_ws8500": "vscyto3d",
    "unetvit3d": "unetvit3d",
    "celldiff_r2": "celldiff",
    "pix2pix3d_unetvit": "pix2pix3d",
    "pix2pix3d_unetvit_modernized_lambdaL1_10_lecam_40ep": "pix2pix3d",
}
ARCH_PAPER = {
    "fnet3d": "FNet3D",
    "unext2": "UNeXt2",
    "vscyto3d": "VSCyto3D",
    "unetvit3d": "UNetViT3D",
    "celldiff": "CELL-Diff",
    "pix2pix3d": "Pix2Pix3D",
}
# Superseded predecessor dirs we deliberately never publish (celldiff_r2 is the
# evaluated CELL-Diff; the bare `celldiff` dir is the earlier run).
EXCLUDE_MODEL = {"celldiff"}

MODEL_ORDER = ("fnet3d", "unext2", "vscyto3d", "unetvit3d", "celldiff", "pix2pix3d")
TRAIN_ORDER = ("ipsc", "a549", "joint")
ORG_ORDER = ("nucleus", "membrane", "er", "mito")

# The models named in the paper / release `models/README.md`. Pix2Pix3D is an
# internal baseline not in that list, so it is excluded by default; add it back
# with `--models ...,pix2pix3d` once it belongs in the public release.
PAPER_MODELS = ("fnet3d", "unext2", "vscyto3d", "unetvit3d", "celldiff")

# In v1, ER/Mito targets for A549 (and therefore the Joint pool) are deconvolved
# GFP; membrane/nucleus and all iPSC targets are raw. These deconv cells are the
# ones the raw-fluorescence retrain will replace -- flagged so a re-run updates
# exactly them. See project_channel_raw_vs_deconv_provenance.
DECONV_TRAIN = {"a549", "joint"}
DECONV_ORG = {"er", "mito"}

CKPT_RE = re.compile(
    r"/models/(?P<root>dynacell|cell_diff_vs_viscy)/"
    r"(?P<train>[^/]+)/(?P<gene>[^/]+)/(?P<model>[^/]+)/(?:checkpoints/)?(?P<file>[^/\s]+\.ckpt)"
)
EPOCH_RE = re.compile(r"epoch=(\d+)-step=(\d+)")
BEST_EP_RE = re.compile(r"best_ep(\d+)")


def provenance(train_pub: str, organelle: str) -> str:
    """Return ``"deconv->raw"`` for the cells the raw retrain will replace, else ``"raw"``."""
    if train_pub in DECONV_TRAIN and organelle in DECONV_ORG:
        return "deconv->raw"
    return "raw"


def collect_cells(selected_models: set[str]) -> dict[tuple, dict]:
    """Scan canonical predict configs; return {(train,organelle,model): record}.

    Only cells whose public model slug is in ``selected_models`` are kept. Each
    record carries every distinct ``ckpt_path`` seen for that cell (many
    per-condition predict configs point at one training checkpoint). Selection
    of the file to publish happens in :func:`resolve`.
    """
    srcs_by_cell: dict[tuple, set[str]] = defaultdict(set)
    for organelle_dir in CANONICAL_ORGANELLE_DIRS:
        for yml in sorted((CONFIG_ROOT / organelle_dir).rglob("predict__*.yml")):
            if "smoke" in yml.name:
                continue
            for line in yml.read_text().splitlines():
                if not line.lstrip().startswith("ckpt_path:"):
                    continue
                m = CKPT_RE.search(line)
                if not m:
                    continue
                train, gene, model = m["train"], m["gene"], m["model"]
                if train not in TRAIN_PUB or gene not in GENE_PUB:
                    continue
                if model in EXCLUDE_MODEL or model not in MODEL_PUB:
                    continue
                if MODEL_PUB[model] not in selected_models:
                    continue
                cell = (TRAIN_PUB[train], GENE_PUB[gene], MODEL_PUB[model])
                srcs_by_cell[cell].add(line.split("ckpt_path:", 1)[1].strip())
    return {cell: {"srcs": sorted(srcs)} for cell, srcs in srcs_by_cell.items()}


def canonical_ckpt(src: Path) -> Path | None:
    """Resolve a pinned ``ckpt_path`` to the canonical ``epoch=NNN-step=MMMM.ckpt`` file.

    Handles three pin styles:
    - ``epoch=NNN-step=MMMM.ckpt`` -- returned as-is.
    - ``last*.ckpt`` -- the ``checkpoints/epoch=*`` sibling with the same byte
      size and the highest epoch (Lightning writes ``last`` byte-identical to the
      final top-k checkpoint).
    - ``best_ep*.ckpt`` -- the ``checkpoints/`` sibling sharing its inode
      (hardlink), falling back to the epoch number embedded in the name.

    Returns the resolved path, or ``None`` if the pin cannot be resolved (which
    the caller treats as ``pending``).
    """
    if not src.exists():
        # best_ep* pins live at the model-dir root; if a specific epoch=* pin was
        # pruned we cannot resolve it -> pending.
        if src.name.startswith("best_ep"):
            pass  # fall through to model-dir search below
        else:
            return None
    if EPOCH_RE.search(src.name) and src.exists():
        return src
    ckpt_dir = src.parent if src.parent.name == "checkpoints" else src.parent / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    siblings = sorted(ckpt_dir.glob("epoch=*.ckpt"))
    if not siblings:
        return None
    if src.name.startswith("last") and src.exists():
        size = src.stat().st_size
        matches = [
            (int(EPOCH_RE.search(s.name).group(1)), s)
            for s in siblings
            if s.stat().st_size == size and EPOCH_RE.search(s.name)
        ]
        return max(matches)[1] if matches else None
    if src.name.startswith("best_ep") and src.exists():
        ino = src.stat().st_ino
        for s in siblings:  # hardlink match is exact
            if s.stat().st_ino == ino:
                return s
        m = BEST_EP_RE.search(src.name)  # fall back to epoch number in the name
        if m:
            want = int(m.group(1))
            for s in siblings:
                em = EPOCH_RE.search(s.name)
                if em and int(em.group(1)) == want:
                    return s
    return None


def resolve(cells: dict[tuple, dict], dest_models: Path) -> list[dict]:
    """Attach status + resolved paths to each cell, sorted for stable output."""
    out = []
    for cell, rec in cells.items():
        train_pub, organelle, model_slug = cell
        chosen = resolved_file = None
        for src_str in rec["srcs"]:  # prefer a pin that resolves to a real file
            r = canonical_ckpt(Path(src_str))
            if r is not None:
                chosen, resolved_file = src_str, r
                break
        model_dir_exists = any(
            (Path(s.split("/checkpoints/", 1)[0]) if "/checkpoints/" in s else Path(s).parent).exists()
            for s in rec["srcs"]
        )
        if resolved_file is not None:
            status = "resolved"
        elif model_dir_exists:
            status = "pending"  # mid-retrain / awaiting re-pin (e.g. deconv->raw)
        else:
            status = "not_trained"
        cfg = None
        if chosen:
            model_root = Path(chosen).parent
            if model_root.name == "checkpoints":
                model_root = model_root.parent
            cfg = model_root / "config.yaml"
        dst_dir = dest_models / train_pub / organelle / model_slug
        out.append(
            {
                "train_pub": train_pub,
                "organelle": organelle,
                "model_slug": model_slug,
                "arch": ARCH_PAPER[model_slug],
                "provenance": provenance(train_pub, organelle),
                "status": status,
                "pub_name": resolved_file.name if resolved_file else "",
                "copy_src": str(resolved_file) if resolved_file else "",
                "dst_ckpt": str(dst_dir / resolved_file.name) if resolved_file else "",
                "config_src": str(cfg) if cfg and cfg.exists() else "",
                "size_gb": round(resolved_file.stat().st_size / 1e9, 3) if resolved_file else 0.0,
                "pinned_srcs": ";".join(rec["srcs"]),
            }
        )
    out.sort(
        key=lambda r: (
            TRAIN_ORDER.index(r["train_pub"]),
            ORG_ORDER.index(r["organelle"]),
            MODEL_ORDER.index(r["model_slug"]),
        )
    )
    return out


def report(plan: list[dict], dest_models: Path, execute: bool) -> None:
    """Print coverage matrix, per-cell plan, and pending/not-trained sections."""
    resolved = [p for p in plan if p["status"] == "resolved"]
    pending = [p for p in plan if p["status"] == "pending"]
    not_trained = [p for p in plan if p["status"] == "not_trained"]
    print(f"\n{'=' * 96}\nDynaCell checkpoint assembly -- {'EXECUTE' if execute else 'DRY RUN'}\n{'=' * 96}")
    print(
        f"resolved {len(resolved)}  |  pending (awaiting retrain/re-pin) {len(pending)}  |  "
        f"not_trained {len(not_trained)}\n"
    )

    have = {(p["train_pub"], p["organelle"], p["model_slug"]): p["status"] for p in plan}
    mark = {"resolved": "Y", "pending": "P", "not_trained": "-"}
    print("Coverage (Y=resolved  P=pending-raw-retrain  -/blank=absent):")
    header = "  " + " " * 17 + "".join(f"{m:>11}" for m in MODEL_ORDER)
    print(header)
    for train in TRAIN_ORDER:
        for org in ORG_ORDER:
            cells = "".join(f"{mark.get(have.get((train, org, m), ''), ''):>11}" for m in MODEL_ORDER)
            print(f"  {train:5s} {org:9s}:{cells}")
    print(f"\nTotal to copy: {sum(p['size_gb'] for p in resolved):.1f} GB across {len(resolved)} checkpoints\n")

    print("Resolved (will copy):")
    for p in resolved:
        prov = "" if p["provenance"] == "raw" else f"  [{p['provenance']}]"
        nocfg = "  [NO config.yaml]" if not p["config_src"] else ""
        print(
            f"  [{p['size_gb']:5.2f} GB] {p['train_pub']}/{p['organelle']}/{p['model_slug']}/"
            f"{p['pub_name']}{prov}{nocfg}"
        )

    if pending:
        print("\nPENDING -- pinned checkpoint gone (raw retrain in progress / re-pin needed):")
        for p in pending:
            print(f"  {p['train_pub']}/{p['organelle']}/{p['model_slug']}  [{p['provenance']}]")
            print(f"      pinned: {p['pinned_srcs']}")
    if not_trained:
        print("\nNOT TRAINED (stale config, model dir absent -- dropped):")
        for p in not_trained:
            print(f"  {p['train_pub']}/{p['organelle']}/{p['model_slug']}")


def write_manifest(plan: list[dict], path: Path) -> None:
    """Write the full manifest CSV (all cells, all statuses)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "train_set",
                "organelle",
                "model",
                "arch",
                "provenance",
                "status",
                "pub_filename",
                "size_gb",
                "dst_path",
                "src_ckpt",
                "config_src",
            ]
        )
        for p in plan:
            w.writerow(
                [
                    p["train_pub"],
                    p["organelle"],
                    p["model_slug"],
                    p["arch"],
                    p["provenance"],
                    p["status"],
                    p["pub_name"],
                    p["size_gb"],
                    p["dst_ckpt"],
                    p["copy_src"],
                    p["config_src"],
                ]
            )


def do_copy(plan: list[dict]) -> None:
    """Copy resolved checkpoints + their config.yaml into the release tree (overwrite)."""
    import shutil

    for p in plan:
        if p["status"] != "resolved":
            continue
        dst = Path(p["dst_ckpt"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p["copy_src"], dst)
        if p["config_src"]:
            shutil.copy2(p["config_src"], dst.parent / "config.yaml")
        print(f"  copied {p['train_pub']}/{p['organelle']}/{p['model_slug']}/{p['pub_name']}")


def main() -> None:
    """Parse args, resolve the plan, report, and (with ``--execute``) copy + write the manifest."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="copy files + write manifest (default: dry run)")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"release root (default: {DEFAULT_DEST})")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest CSV path (default: <dest>/models/checkpoints.csv on execute; stdout-adjacent temp on dry run)",
    )
    ap.add_argument(
        "--models",
        default=",".join(PAPER_MODELS),
        help=f"comma-separated model slugs to publish (default: paper set {','.join(PAPER_MODELS)}; "
        "pix2pix3d excluded -- add it to include)",
    )
    args = ap.parse_args()

    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    unknown = selected - set(MODEL_ORDER)
    if unknown:
        ap.error(f"unknown model slug(s): {sorted(unknown)}; choose from {list(MODEL_ORDER)}")

    dest_models = args.dest / "models"
    plan = resolve(collect_cells(selected), dest_models)
    report(plan, dest_models, args.execute)

    manifest = args.manifest or (
        dest_models / "checkpoints.csv" if args.execute else args.dest / "checkpoints_manifest_dryrun.csv"
    )
    # A run that resolves nothing is always a bug -- a path-grammar drift, a bad
    # --models filter -- never a legitimate no-op. Fail before write_manifest can
    # replace a good manifest with dead rows while the published .ckpt stay on disk.
    if not any(p["status"] == "resolved" for p in plan):
        raise SystemExit(
            f"no checkpoint resolved from {len(plan)} candidate cell(s); refusing to write a "
            "manifest. Check TRAIN_PUB/GENE_PUB against the ckpt_path grammar in the predict leaves."
        )
    if args.execute:
        print("\nEXECUTING copies...")
        do_copy(plan)
        write_manifest(plan, manifest)
        print(f"\nmanifest: {manifest}\ndone.")
    else:
        write_manifest(plan, manifest)
        print(f"\n(dry run) manifest preview: {manifest}\nre-run with --execute to copy.")


if __name__ == "__main__":
    main()
