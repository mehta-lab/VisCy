"""Tests for the a549-mantis pool-manifest generator.

Guards the manifest-shape contract the eval chain depends on — in
particular that the generator emits the additive stores.cell_segmentation
and stores.gt_cache_dir fields (a prior version omitted both, which would
strip eval-critical paths on regeneration). These are pure-function tests;
the full deterministic regen against source zarrs is an HPC/E2E concern.
"""

from pathlib import Path

from dynacell.preprocess.a549_mantis.generate_pool_manifests import (
    _dominant_spacing,
    _manifest_yaml,
)

OUTPUT_ROOT = Path("/hpc/projects/virtual_staining/training/dynacell/a549/mantis")


class TestManifestStores:
    """_manifest_yaml emits the full stores block eval reads."""

    def test_er_single_store_emits_seg_and_cache(self):
        """ER (separate store) emits train/test + cell_segmentation + gt_cache_dir."""
        m = _manifest_yaml(
            target="sec61b",
            condition="mock",
            spacing=(0.174, 0.1494, 0.1494),
            output_root=OUTPUT_ROOT,
            splits_relpath="splits/sec61b_train_test.yaml",
        )
        assert m["name"] == "a549-mantis-sec61b-mock"
        stores = m["targets"]["sec61b"]["stores"]
        assert stores["train"] == str(OUTPUT_ROOT / "train" / "SEC61B_all.zarr")
        assert stores["test"] == str(OUTPUT_ROOT / "test" / "SEC61B_mock.zarr")
        assert stores["cell_segmentation"] == str(OUTPUT_ROOT / "test" / "SEC61B_mock_seg_cleaned.zarr")
        assert stores["gt_cache_dir"] == str(OUTPUT_ROOT.parent / "eval_cache" / "sec61b_mock")

    def test_combined_store_and_condition_casing(self):
        """h2b/caax share dual_nucl_memb; seg keeps verbatim condition casing,
        gt_cache_dir lowercases it."""
        m = _manifest_yaml(
            target="caax",
            condition="ZIKV",
            spacing=(0.174, 0.1494, 0.1494),
            output_root=OUTPUT_ROOT,
            splits_relpath="splits/caax_train_test.yaml",
        )
        assert m["name"] == "a549-mantis-caax-zikv"
        stores = m["targets"]["caax"]["stores"]
        assert stores["test"] == str(OUTPUT_ROOT / "test" / "dual_nucl_memb_ZIKV.zarr")
        assert stores["cell_segmentation"] == str(OUTPUT_ROOT / "test" / "dual_nucl_memb_ZIKV_seg_cleaned.zarr")
        assert stores["gt_cache_dir"] == str(OUTPUT_ROOT.parent / "eval_cache" / "caax_zikv")


class TestDominantSpacing:
    """_dominant_spacing picks the modal per-pool spacing (Z is eval-critical)."""

    def test_modal_selection(self):
        """The most common spacing across contributing plates wins."""
        common = (0.174, 0.1494, 0.1494)
        spacings = [common, common, (0.29, 0.108, 0.108)]
        assert _dominant_spacing(spacings) == common

    def test_first_plate_tiebreak(self):
        """On a count tie, the first-seen spacing wins (Counter.most_common)."""
        first = (0.174, 0.1494, 0.1494)
        second = (0.29, 0.108, 0.108)
        assert _dominant_spacing([first, second]) == first
