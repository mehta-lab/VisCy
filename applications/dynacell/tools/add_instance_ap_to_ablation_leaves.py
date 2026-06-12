"""Wire instance AP into the vscyto3d-ablation eval leaves (nucleus & membrane).

The ablation tracks (randinit / cytoland / infectionft / cytolandft /
infectionft_dynacellft) are hand-authored single-leaf configs under
``_internal/leaf/<organelle>/<model>/<train_set>/eval__{a549_mantis,ipsc_confocal}.yaml``
— they are NOT produced by ``generate_grouped_eval_configs.py`` (that walker
skips ablation zarrs). So the instance-AP wiring folded into the grouped
generator does not reach them. This patcher applies the SAME wiring directly,
in place, to the nucleus & membrane ablation leaves:

- nucleus  -> compute_instance_ap + segmentation.backend=cellpose
- membrane -> compute_instance_ap + segmentation.backend=cellpose_watershed
  + nuclei_channel_name=Nuclei, and per-A549-condition io.nuclei_gt_path
  pointing at the separate H2B nuclei store (iPSC nuclei live in the same
  cell.zarr, so the iPSC leaf gets no nuclei_gt_path).

ER/mito ablation leaves are left untouched (no cell-instance interpretation).

Text-surgical (PyYAML/ruamel would drop the hand-written header comments):
inserts the instance block after the ``compute_feature_metrics: true`` line and
a ``nuclei_gt_path`` sibling after each per-condition ``pred_path`` whose zarr
carries an A549 ``_<cond>`` suffix. Idempotent — a leaf that already declares
``compute_instance_ap`` is skipped.

Usage::

    uv run python applications/dynacell/tools/add_instance_ap_to_ablation_leaves.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from generate_grouped_eval_configs import a549_nuclei_store  # noqa: E402

_LEAF_ROOT = _TOOLS_DIR.parent / "configs/benchmarks/virtual_staining/_internal/leaf"
_ABLATION_MODEL_DIRS: tuple[str, ...] = (
    "fcmae_vscyto3d_pretrained_randinit",
    "fcmae_vscyto3d_pretrained_cytoland",
    "fcmae_vscyto3d_pretrained_infectionft",
    "vscyto3d_cytolandft",
    "vscyto3d_infectionft_dynacellft",
)
_INSTANCE_ORGANELLES: tuple[str, ...] = ("nucleus", "membrane")
_FEATURE_ANCHOR = "compute_feature_metrics: true"
# A per-condition prediction zarr on A549 ends in _<cond>.zarr; the iPSC leaf's
# single pred_path has no condition suffix and must not match.
_PRED_COND_RE = re.compile(r"^(\s*)pred_path: .*_(mock|denv|zikv)\.zarr\s*$")


def _instance_block(organelle: str) -> list[str]:
    """Top-level instance-AP overlay lines for ``organelle`` (cellpose / watershed)."""
    lines = ["compute_instance_ap: true\n", "segmentation:\n"]
    if organelle == "nucleus":
        lines.append("  backend: cellpose\n")
    else:  # membrane
        lines.append("  backend: cellpose_watershed\n")
        lines.append("  nuclei_channel_name: Nuclei\n")
    return lines


def patch_leaf(path: Path, organelle: str) -> bool:
    """Inject the instance-AP wiring into one ablation leaf. Returns True if changed."""
    text = path.read_text()
    if "compute_instance_ap" in text:
        return False  # idempotent
    if _FEATURE_ANCHOR not in text:
        raise ValueError(f"{path}: missing '{_FEATURE_ANCHOR}' anchor; structure unexpected")

    out: list[str] = []
    for line in text.splitlines(keepends=True):
        out.append(line)
        if line.strip() == _FEATURE_ANCHOR:
            out.extend(_instance_block(organelle))
        elif organelle == "membrane":
            m = _PRED_COND_RE.match(line)
            if m:
                indent, cond = m.group(1), m.group(2)
                out.append(f"{indent}nuclei_gt_path: {a549_nuclei_store(cond)}\n")
    path.write_text("".join(out))
    return True


def ablation_leaves() -> list[tuple[Path, str]]:
    """All (leaf_path, organelle) pairs for the nucleus & membrane ablation tracks."""
    pairs: list[tuple[Path, str]] = []
    for organelle in _INSTANCE_ORGANELLES:
        for model_dir in _ABLATION_MODEL_DIRS:
            base = _LEAF_ROOT / organelle / model_dir
            if not base.is_dir():
                continue
            for leaf in sorted(base.rglob("eval__*.yaml")):
                pairs.append((leaf, organelle))
    return pairs


def main(argv: list[str] | None = None) -> int:
    """Patch every nucleus/membrane ablation eval leaf with the instance-AP wiring."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    args = ap.parse_args(argv)

    pairs = ablation_leaves()
    changed = skipped = 0
    for leaf, organelle in pairs:
        rel = leaf.relative_to(_LEAF_ROOT)
        if args.dry_run:
            already = "compute_instance_ap" in leaf.read_text()
            print(f"  {'skip (already)' if already else 'would patch'}: {organelle:8} {rel}")
            skipped += already
            changed += not already
            continue
        if patch_leaf(leaf, organelle):
            print(f"  patched ({organelle}): {rel}")
            changed += 1
        else:
            print(f"  skip (already): {rel}")
            skipped += 1
    print(f"[ablation-instance-ap] {changed} patched, {skipped} already-wired, {len(pairs)} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
