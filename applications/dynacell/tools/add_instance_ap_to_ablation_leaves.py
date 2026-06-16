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
  + nuclei_channel_name=Nuclei + watershed.subtract_nuclei=false (score the full
  whole cell, not the carved cytoplasm shell), and per-A549-condition
  io.nuclei_gt_path pointing at the separate H2B nuclei store (iPSC nuclei live
  in the same cell.zarr, so the iPSC leaf gets no nuclei_gt_path).

ER/mito ablation leaves are left untouched (no cell-instance interpretation).

Text-surgical (PyYAML/ruamel would drop the hand-written header comments):
inserts the instance block after the ``compute_feature_metrics: true`` line and
a ``nuclei_gt_path`` sibling after each per-condition ``pred_path`` whose zarr
carries an A549 ``_<cond>`` suffix. Idempotent — a fully-wired leaf is skipped;
a membrane leaf wired before the no-carve fix gets only the
``watershed.subtract_nuclei=false`` block inserted under its existing seg block.

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
# Membrane seg anchor + the no-carve watershed block inserted beneath it.
_NUCLEI_CHANNEL_LINE = "  nuclei_channel_name: Nuclei\n"
_WATERSHED_NOCARVE = "  watershed:\n    subtract_nuclei: false\n"


def _instance_block(organelle: str) -> list[str]:
    """Top-level instance-AP overlay lines for ``organelle`` (cellpose / watershed)."""
    lines = ["compute_instance_ap: true\n", "segmentation:\n"]
    if organelle == "nucleus":
        lines.append("  backend: cellpose\n")
    else:  # membrane
        lines.append("  backend: cellpose_watershed\n")
        lines.append(_NUCLEI_CHANNEL_LINE)
        # Score the full whole cell, not the carved cytoplasm shell — the
        # eval.yaml default subtract_nuclei=true carves the shared nucleus core
        # and collapses AP@0.50 to ~0.04 even in-distribution (the IoU-brittle
        # cytoplasm boundary). See generate_grouped_eval_configs.build_leaf_yaml.
        lines.append(_WATERSHED_NOCARVE)
    return lines


def plan_patch(text: str, organelle: str, leaf_id: str) -> tuple[str | None, str]:
    """Compute the patched text for one ablation leaf without writing.

    Returns ``(new_text, action)``. ``new_text`` is ``None`` when nothing
    changes; ``action`` is a short label for the report (``"fresh"``,
    ``"watershed-upgrade"``, or ``"skip"``).
    """
    if "compute_instance_ap" in text:
        # Already wired. Membrane leaves wired before the no-carve fix still lack
        # the watershed.subtract_nuclei=false block; upgrade them in place so the
        # whole-cell AP scores the full cell, not the carved cytoplasm shell.
        if organelle != "membrane" or "subtract_nuclei" in text:
            return None, "skip"
        if _NUCLEI_CHANNEL_LINE not in text:
            raise ValueError(f"{leaf_id}: instance-AP-wired membrane leaf missing the nuclei_channel anchor")
        return text.replace(_NUCLEI_CHANNEL_LINE, _NUCLEI_CHANNEL_LINE + _WATERSHED_NOCARVE, 1), "watershed-upgrade"
    if _FEATURE_ANCHOR not in text:
        raise ValueError(f"{leaf_id}: missing '{_FEATURE_ANCHOR}' anchor; structure unexpected")

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
    return "".join(out), "fresh"


def patch_leaf(path: Path, organelle: str) -> str:
    """Inject/upgrade the instance-AP wiring in one ablation leaf. Returns the action label."""
    new_text, action = plan_patch(path.read_text(), organelle, str(path))
    if new_text is not None:
        path.write_text(new_text)
    return action


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
            _, action = plan_patch(leaf.read_text(), organelle, str(rel))
        else:
            action = patch_leaf(leaf, organelle)
        if action == "skip":
            print(f"  skip (already): {organelle:8} {rel}")
            skipped += 1
        else:
            verb = "would patch" if args.dry_run else "patched"
            print(f"  {verb} ({action}): {organelle:8} {rel}")
            changed += 1
    print(f"[ablation-instance-ap] {changed} patched, {skipped} already-wired, {len(pairs)} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
