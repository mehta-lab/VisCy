"""Tests for A549 mantis pool assembly.

Unit tests for the grid + channels + authoring + helper modules run
without HPC data. An HPC-gated round-trip test exercises the full
``assemble_pool`` pipeline on real plate zarrs.
"""

import json
import os
import shutil
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from omegaconf import OmegaConf

from dynacell.preprocess.a549_mantis import (
    CANONICAL_TARGET_CHANNELS,
    COMBINED_TARGETS,
    GENE_TO_FILENAME,
    VALID_CONDITIONS,
    assemble_pool,
    build_grid,
    load_platemap,
    load_splits,
    resolve_channels,
    resolve_target_genes,
)
from dynacell.preprocess.a549_mantis import assemble as assemble_mod
from dynacell.preprocess.a549_mantis.assemble import (
    _apply_center_crop_yx,
    _assembly_inputs_hash,
    _check_output_safety,
    _default_authoring_root,
    _derive_raw_only_store,
    _list_authored_plates,
    _PlateContribution,
    _pool_filename,
    _pool_output_path,
    _provenance_sidecar_path,
    _raw_only_channel_indices,
    _read_source_spacing,
    _resample_yx_to_pixel_size,
)

AUTHORING_ROOT = Path(str(files("dynacell") / "_configs" / "datasets" / "a549-mantis" / "authoring"))
PLATE_ZARR_ROOT = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics")
PLATE_11_07 = "2024_11_07_A549_SEC61_DENV"


class TestBuildGrid:
    """Unit tests for the 2-h odd-hpi grid with ±1.5 h tail snap."""

    def test_11_07_full_window(self):
        """10-min native, 109 frames from hpi=4: 9 in-window + 1 tail snap."""
        frames = build_grid(
            native_delta_t_min=10.0,
            native_t=109,
            hpi_start=4.0,
        )
        assert len(frames) == 10
        tick_hpis = [f.tick_hpi for f in frames]
        assert tick_hpis == [5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0]
        assert frames[-1].hpi == pytest.approx(22.0)
        assert frames[-1].tick_hpi == pytest.approx(23.0)
        assert frames[-1].native_idx == 108

    def test_10_31_tail_snap(self):
        """30-min native, 24 frames from hpi=4: 6 in-window + 1 snap at 15.5."""
        frames = build_grid(
            native_delta_t_min=30.0,
            native_t=24,
            hpi_start=4.0,
        )
        assert len(frames) == 7
        assert frames[-1].hpi == pytest.approx(15.5)
        assert frames[-1].tick_hpi == pytest.approx(17.0)


class TestResolveChannels:
    """Unit tests for per-well channel selection + rename."""

    # Real v1 ER/mito plate channel order (verified against the source
    # zarrs for 2024_10_31_A549_SEC61 / 2024_10_29_A549_TOMM20):
    #   [0] Phase3D
    #   [1] raw GFP EX488 EM525-45        (raw GFP  -> Structure)
    #   [2] raw mCherry EX561 EM600-37
    #   [3] GFP EX488 EM525-45            (deconv   -> Structure_deconvolved)
    #   [4] mCherry EX561 EM600-37
    #   [5] BF                            (Brightfield)
    V1_NATIVE = [
        "Phase3D",
        "raw GFP EX488 EM525-45",
        "raw mCherry EX561 EM600-37",
        "GFP EX488 EM525-45",
        "mCherry EX561 EM600-37",
        "BF",
    ]

    def test_11_07_sec61b(self):
        """ER target emits raw Structure + deconvolved, plus passthroughs."""
        sel = resolve_channels(
            self.V1_NATIVE,
            {"GFP EX488 EM525-45": "sec61b"},
            "sec61b",
        )
        # Phase3D=0, BF=5, raw-GFP=1, deconv-GFP=3.
        assert sel.input_indices == [0, 5, 1, 3]
        assert sel.output_names == [
            "Phase3D",
            "Brightfield",
            "Structure",
            "Structure_deconvolved",
        ]

    def test_tomm20_raw_and_deconv(self):
        """Mito target follows the same raw+deconvolved layout as ER."""
        sel = resolve_channels(
            self.V1_NATIVE,
            {"GFP EX488 EM525-45": "tomm20"},
            "tomm20",
        )
        assert sel.input_indices == [0, 5, 1, 3]
        assert sel.output_names == [
            "Phase3D",
            "Brightfield",
            "Structure",
            "Structure_deconvolved",
        ]

    def test_dual_nucl_memb_combined(self):
        """Combined v2 target emits Nuclei + Membrane in one selection."""
        # Real v2 CAAX_H2B plate channel order:
        #   [0] BF, [1] raw mCherry (caax), [2] raw Cy5 (h2b),
        #   [3] Phase3D, [4] nuclei_prediction, [5] membrane_prediction
        native = [
            "BF",
            "raw mCherry EX561 EM600-37",
            "raw Cy5 EX639 EM698-70",
            "Phase3D",
            "nuclei_prediction",
            "membrane_prediction",
        ]
        gene_map = {
            "raw mCherry EX561 EM600-37": "caax",
            "raw Cy5 EX639 EM698-70": "h2b",
        }
        sel = resolve_channels(native, gene_map, "dual_nucl_memb")
        # Passthrough order is canonical [Phase3D, Brightfield] regardless
        # of native index; then constituent genes in COMBINED_TARGETS order
        # (h2b -> Nuclei, then caax -> Membrane).
        assert sel.output_names == ["Phase3D", "Brightfield", "Nuclei", "Membrane"]
        assert sel.input_indices == [3, 0, 2, 1]

    def test_missing_cube_raises(self):
        """Map points to a cube not on the plate."""
        with pytest.raises(ValueError, match="not found"):
            resolve_channels(
                ["Phase3D", "BF", "raw GFP EX488 EM525-45"],
                {"GFP EX488 EM525-45": "caax"},  # caax on GFP -> single-gene raw path
                "caax",
            )

    def test_missing_raw_counterpart_raises(self):
        """ER/mito require the raw GFP counterpart; absence raises."""
        native = [
            "Phase3D",
            "GFP EX488 EM525-45",  # deconv present, but no `raw GFP ...`
            "BF",
        ]
        with pytest.raises(ValueError, match="raw counterpart"):
            resolve_channels(
                native,
                {"GFP EX488 EM525-45": "sec61b"},
                "sec61b",
            )

    def test_gene_not_mapped_raises(self):
        """Target gene not in gene_channel_map."""
        with pytest.raises(ValueError, match="does not map"):
            resolve_channels(
                ["Phase3D", "GFP EX488 EM525-45", "raw GFP EX488 EM525-45"],
                {"GFP EX488 EM525-45": "tomm20"},
                "sec61b",
            )

    def test_unknown_gene_raises(self):
        """Target key neither a canonical single gene nor a combined key."""
        with pytest.raises(ValueError, match="not a known"):
            resolve_channels(
                ["Phase3D"],
                {"GFP EX488 EM525-45": "exotic_gene"},
                "exotic_gene",
            )


class TestResolveTargetGenes:
    """Combined vs single-gene target resolution."""

    def test_single_gene_targets_resolve_to_self(self):
        """Each canonical gene resolves to a 1-tuple of itself."""
        for gene in ("sec61b", "tomm20", "h2b", "caax"):
            assert resolve_target_genes(gene) == (gene,)

    def test_combined_target_resolves_to_constituents(self):
        """dual_nucl_memb resolves to its ordered constituent genes."""
        assert resolve_target_genes("dual_nucl_memb") == ("h2b", "caax")
        assert COMBINED_TARGETS["dual_nucl_memb"] == ("h2b", "caax")

    def test_unknown_target_raises(self):
        """Unknown target key raises listing both known sets."""
        with pytest.raises(ValueError, match="not a known"):
            resolve_target_genes("bogus")


class TestRawOnlyChannelIndices:
    """The ozx-slim publication artifact drops *_deconvolved channels."""

    def test_drops_deconvolved(self):
        """ER/mito internal store → raw-only keeps all but Structure_deconvolved."""
        names = ["Phase3D", "Brightfield", "Structure", "Structure_deconvolved"]
        keep = _raw_only_channel_indices(names)
        assert keep == [0, 1, 2]
        assert [names[i] for i in keep] == ["Phase3D", "Brightfield", "Structure"]

    def test_noop_when_no_deconvolved(self):
        """Nuclei/membrane store has no deconvolved channel → keep everything."""
        names = ["Phase3D", "Brightfield", "Nuclei", "Membrane"]
        assert _raw_only_channel_indices(names) == [0, 1, 2, 3]


class TestDeriveRawOnlyStore:
    """Integration test for the raw-only ``.ozx`` derivation.

    Regression guard: the internal store is 4-channel and carries its channel
    list in the NGFF v0.5 ``ome`` zattr. The derivation must NOT copy that
    block verbatim onto the 3-channel subset store, or ``channel_names``
    reports 4 channels and the normalization pass indexes out of bounds.
    """

    def _write_internal_4ch(self, path: Path) -> None:
        """Write a 1-position 4-channel v0.5 store with provenance zattrs."""
        rng = np.random.default_rng(0)
        data = rng.random((2, 4, 3, 64, 64)).astype(np.float32)
        with open_ome_zarr(
            path,
            layout="hcs",
            mode="w",
            channel_names=[
                "Phase3D",
                "Brightfield",
                "Structure",
                "Structure_deconvolved",
            ],
            version="0.5",
        ) as plate:
            pos = plate.create_position("0", "0", "fov0000")
            pos.create_image(
                name="0",
                data=data,
                transform=[TransformationMeta(type="scale", scale=[1.0, 1.0, 0.174, 0.1494, 0.1494])],
            )
            pos.zattrs.update({"source_position": "B/1/000000", "plate_id": "synthetic"})

    def test_drops_deconv_and_preserves_provenance(self, tmp_path):
        """4-channel internal store → raw-only ozx keeps 3 channels + provenance."""
        internal = tmp_path / "SEC61B_mock.zarr"
        self._write_internal_4ch(internal)
        ozx = tmp_path / "SEC61B_mock.ozx"

        assert _derive_raw_only_store(internal, ozx) is True

        with open_ome_zarr(ozx, mode="r", layout="hcs") as s:
            assert list(s.channel_names) == ["Phase3D", "Brightfield", "Structure"]
            _, pos = next(iter(s.positions()))
            assert pos["0"].shape[1] == 3  # channel axis subset to 3
            pz = dict(pos.zattrs)
            assert pz["source_position"] == "B/1/000000"  # provenance preserved
            assert pz["plate_id"] == "synthetic"
            assert "normalization" in pz  # recomputed for the kept channels

        # Internal store is untouched (still 4 channels).
        with open_ome_zarr(internal, mode="r", layout="hcs") as s:
            assert len(list(s.channel_names)) == 4

    def test_noop_when_no_deconvolved(self, tmp_path):
        """A store with no *_deconvolved channel returns False and writes no ozx."""
        path = tmp_path / "dual_nucl_memb_mock.zarr"
        data = np.zeros((1, 4, 2, 32, 32), dtype=np.float32)
        with open_ome_zarr(
            path,
            layout="hcs",
            mode="w",
            channel_names=["Phase3D", "Brightfield", "Nuclei", "Membrane"],
            version="0.5",
        ) as plate:
            pos = plate.create_position("0", "0", "fov0000")
            pos.create_image(
                name="0",
                data=data,
                transform=[TransformationMeta(type="scale", scale=[1.0, 1.0, 0.174, 0.1494, 0.1494])],
            )
        ozx = tmp_path / "dual_nucl_memb_mock.ozx"
        assert _derive_raw_only_store(path, ozx) is False
        assert not ozx.exists()


class TestReadSourceSpacing:
    """Unit tests for the per-plate NGFF spacing reader."""

    def _write_fixture(self, path: Path, scale: list[float]) -> None:
        """Write a 1-position HCS zarr with the given 5D scale."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.zeros((1, 1, 1, 4, 4), dtype=np.float32)
        with open_ome_zarr(
            path,
            layout="hcs",
            mode="w",
            channel_names=["c0"],
            version="0.4",
        ) as plate:
            pos = plate.create_position("B", "1", "000000")
            pos.create_image(
                name="0",
                data=data,
                transform=[TransformationMeta(type="scale", scale=scale)],
            )

    def test_v1_fixture(self, tmp_path):
        """v1-shaped scale → returns (0.174, 0.1494, 0.1494)."""
        zarr_path = tmp_path / "v1.zarr"
        self._write_fixture(zarr_path, [1.0, 1.0, 0.174, 0.1494, 0.1494])
        assert _read_source_spacing(zarr_path) == pytest.approx((0.174, 0.1494, 0.1494))

    def test_v2_fixture(self, tmp_path):
        """v2-shaped scale → returns (0.174, 0.116, 0.116)."""
        zarr_path = tmp_path / "v2.zarr"
        self._write_fixture(zarr_path, [1.0, 1.0, 0.174, 0.116, 0.116])
        assert _read_source_spacing(zarr_path) == pytest.approx((0.174, 0.116, 0.116))

    def test_invalid_spacing_raises(self, tmp_path):
        """Non-finite or non-positive spacing raises ValueError."""
        zarr_path = tmp_path / "bad.zarr"
        self._write_fixture(zarr_path, [1.0, 1.0, 0.174, -0.116, 0.116])
        with pytest.raises(ValueError, match="invalid y spacing"):
            _read_source_spacing(zarr_path)


class TestResampleYxToPixelSize:
    """Unit tests for the YX pixel-size resample helper."""

    def test_no_op_when_source_equals_target(self):
        """Source pixel size equal to target on both axes returns input unchanged."""
        rng = np.random.default_rng(0)
        block = rng.standard_normal((1, 1, 1, 8, 8)).astype(np.float32)
        out = _resample_yx_to_pixel_size(block, source_y_um=0.1494, source_x_um=0.1494, target_yx_um=0.1494)
        assert out is block

    def test_v2_to_v1_downsample_shape(self):
        """1600x1332 @ 0.116 → 1242x1034 @ 0.1494 — round(src*ratio)."""
        block = np.zeros((1, 1, 1, 1600, 1332), dtype=np.float32)
        out = _resample_yx_to_pixel_size(block, source_y_um=0.116, source_x_um=0.116, target_yx_um=0.1494)
        assert out.shape == (1, 1, 1, 1242, 1034)
        assert out.dtype == np.float32

    def test_preserves_t_c_z(self):
        """Leading (T, C, Z) axes pass through; only YX is resampled."""
        rng = np.random.default_rng(0)
        block = rng.standard_normal((3, 2, 4, 100, 80)).astype(np.float32)
        out = _resample_yx_to_pixel_size(block, source_y_um=0.116, source_x_um=0.116, target_yx_um=0.1494)
        assert out.shape[:3] == (3, 2, 4)

    def test_refuses_upsample(self):
        """Source coarser than target would require upsampling — raise instead."""
        block = np.zeros((1, 1, 1, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="refusing to upsample"):
            _resample_yx_to_pixel_size(block, source_y_um=0.2, source_x_um=0.2, target_yx_um=0.1494)

    def test_moves_each_tile_to_gpu_when_device_present(self, monkeypatch):
        """Resample moves each tile to the GPU (``ascupy``) when a device is present.

        cubic dispatches by the INPUT array's device — a NumPy array silently
        runs ``skimage`` on the CPU even with cupy/cucim installed. If the
        caller stops moving the block to the device, this catches it without
        needing a real GPU.
        """
        monkeypatch.setattr(assemble_mod.CUDAManager, "get_num_gpus", lambda self: 1)
        calls = {"ascupy": 0}

        def fake_ascupy(arr):
            calls["ascupy"] += 1
            return arr  # keep NumPy so resize still runs CPU-side in the test

        monkeypatch.setattr(assemble_mod, "_ascupy", fake_ascupy)
        block = np.zeros((3, 2, 4, 100, 80), dtype=np.float32)
        out = _resample_yx_to_pixel_size(block, source_y_um=0.116, source_x_um=0.116, target_yx_um=0.1494)
        assert calls["ascupy"] == block.shape[0]  # one transfer per T-tile
        assert out.shape[:3] == (3, 2, 4)

    def test_no_gpu_stays_on_cpu(self, monkeypatch):
        """Without a CUDA device, the resample never calls ``ascupy``."""
        monkeypatch.setattr(assemble_mod.CUDAManager, "get_num_gpus", lambda self: 0)
        calls = {"ascupy": 0}

        def fake_ascupy(arr):
            calls["ascupy"] += 1
            return arr

        monkeypatch.setattr(assemble_mod, "_ascupy", fake_ascupy)
        block = np.zeros((3, 2, 4, 100, 80), dtype=np.float32)
        _resample_yx_to_pixel_size(block, source_y_um=0.116, source_x_um=0.116, target_yx_um=0.1494)
        assert calls["ascupy"] == 0


class TestCenterCropYx:
    """Unit tests for the YX center-crop helper."""

    def test_none_returns_input_unchanged(self):
        """``center_crop_yx=None`` returns the same array object."""
        block = np.arange(2 * 1 * 1 * 4 * 4, dtype=np.float32).reshape(2, 1, 1, 4, 4)
        out = _apply_center_crop_yx(block, None)
        assert out is block

    def test_centered_extraction(self):
        """Even source, even crop → exact centered slice."""
        block = np.arange(1 * 1 * 1 * 6 * 6, dtype=np.float32).reshape(1, 1, 1, 6, 6)
        out = _apply_center_crop_yx(block, (4, 4))
        assert out.shape == (1, 1, 1, 4, 4)
        np.testing.assert_array_equal(out[0, 0, 0], block[0, 0, 0, 1:5, 1:5])

    def test_odd_to_even_floor_offset(self):
        """Odd source, even crop → asymmetric offset (floor of (src-crop)/2)."""
        block = np.arange(1 * 1 * 1 * 7 * 7, dtype=np.float32).reshape(1, 1, 1, 7, 7)
        out = _apply_center_crop_yx(block, (4, 4))
        np.testing.assert_array_equal(out[0, 0, 0], block[0, 0, 0, 1:5, 1:5])

    def test_oversized_crop_raises(self):
        """Crop larger than source YX raises rather than silently shrinking."""
        block = np.zeros((1, 1, 1, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="exceeds source YX"):
            _apply_center_crop_yx(block, (8, 4))


class TestPoolPathHelpers:
    """Pool filename / output path / sidecar helpers."""

    def test_pool_filename_condition(self):
        """Per-condition pool filename is ``<GENE>_<CONDITION>``."""
        assert _pool_filename("sec61b", "mock") == "SEC61B_mock"
        assert _pool_filename("tomm20", "ZIKV") == "TOMM20_ZIKV"
        assert _pool_filename("h2b", "DENV") == "H2B_DENV"

    def test_pool_filename_all(self):
        """``condition=None`` produces the organelle-pool ``_all`` form."""
        assert _pool_filename("caax", None) == "CAAX_all"

    def test_pool_output_path_ozx(self, tmp_path):
        """OZX format → ``<root>/<split>/<GENE>_<COND>.ozx`` under output_root."""
        p = _pool_output_path(tmp_path, "train", "sec61b", "mock", "ozx")
        assert p == tmp_path / "train" / "SEC61B_mock.ozx"

    def test_pool_output_path_zarr_for_all(self, tmp_path):
        """``_all`` is conventionally zarr; helper accepts either format."""
        p = _pool_output_path(tmp_path, "train", "tomm20", None, "zarr")
        assert p == tmp_path / "train" / "TOMM20_all.zarr"

    def test_pool_output_path_invalid_format_raises(self, tmp_path):
        """Unknown format raises ValueError listing the valid set."""
        with pytest.raises(ValueError, match="not in"):
            _pool_output_path(tmp_path, "test", "sec61b", "mock", "tiff")

    def test_provenance_sidecar_colocated(self, tmp_path):
        """Sidecar lives next to the store with ``.provenance.json`` suffix."""
        store = tmp_path / "train" / "SEC61B_mock.ozx"
        sidecar = _provenance_sidecar_path(store)
        assert sidecar == tmp_path / "train" / "SEC61B_mock.provenance.json"


class TestAuthoringYamls:
    """Load + validate the authored 11_07 platemap and splits."""

    def test_11_07_platemap(self):
        """Platemap loads with expected native_t, hpi_start, and well set."""
        pm = load_platemap(AUTHORING_ROOT / "platemaps" / f"{PLATE_11_07}.yaml")
        assert pm.experiment == PLATE_11_07
        assert pm.native_t == 109
        assert pm.native_delta_t_min == 10
        assert pm.hpi_start == 4.0
        assert set(pm.wells) == {"B/1", "B/2", "B/3", "C/2"}
        assert pm.wells["B/1"].condition == "mock"
        assert pm.wells["B/2"].condition == "DENV"
        assert pm.wells["B/2"].moi == 5

    def test_11_07_splits(self):
        """Splits load with disjoint train/test for sec61b."""
        sp = load_splits(AUTHORING_ROOT / "splits" / f"{PLATE_11_07}.yaml")
        assert sp.experiment == PLATE_11_07
        assert "sec61b" in sp.targets
        sec61b = sp.targets["sec61b"]
        assert set(sec61b.train).isdisjoint(set(sec61b.test))


class TestAuthoringRootFallback:
    """assemble.py resolves authoring_root from packaged _configs/ by default."""

    def test_default_authoring_root_points_to_package_resource(self):
        """Default path resolves through importlib.resources, not a user home."""
        resolved = _default_authoring_root()
        assert resolved.is_dir(), f"Packaged authoring dir missing: {resolved}"
        assert (resolved / "platemaps" / f"{PLATE_11_07}.yaml").is_file()
        assert (resolved / "splits" / f"{PLATE_11_07}.yaml").is_file()

    def test_list_authored_plates_is_sorted(self):
        """Plate discovery is alphabetic so pool indices are stable across runs."""
        plates = _list_authored_plates(_default_authoring_root())
        assert plates == sorted(plates)
        assert PLATE_11_07 in plates


class TestPoolAssemblyInputsHash:
    """Hash determinism + sensitivity for the pool-level fingerprint."""

    @staticmethod
    def _make_contrib(
        tmp_path: Path,
        plate_name: str,
        positions: list[tuple[str, str]],
        spacing: tuple[float, float, float] = (0.174, 0.1494, 0.1494),
    ) -> _PlateContribution:
        """Synthetic contribution with stub platemap/splits text on disk."""
        pm_path = tmp_path / f"{plate_name}_platemap.yaml"
        sp_path = tmp_path / f"{plate_name}_splits.yaml"
        pm_path.write_text(f"experiment: {plate_name}\n")
        sp_path.write_text(f"experiment: {plate_name}\nposition_count: {len(positions)}\n")
        return _PlateContribution(
            plate_name=plate_name,
            platemap_path=pm_path,
            splits_path=sp_path,
            platemap=None,  # type: ignore[arg-type]  # hash never inspects it
            plate_zarr_path=Path("/dev/null"),
            spacing=spacing,
            grid_frames=[],
            native_frame_indices=np.array([0, 1, 2, 3], dtype=np.int64),
            positions=positions,
        )

    def test_hash_is_deterministic(self, tmp_path):
        """Same inputs twice → identical hash."""
        c = self._make_contrib(tmp_path, "plateA", [("B/1", "0")])
        kwargs = dict(
            contribs=[c],
            target="sec61b",
            condition="mock",
            output_channel_names=["Phase3D", "Structure"],
            center_crop_yx=(640, 960),
            target_yx_pixel_size_um=0.1494,
        )
        assert _assembly_inputs_hash(**kwargs) == _assembly_inputs_hash(**kwargs)

    def test_hash_invariant_to_plate_order(self, tmp_path):
        """Reordering contributing plates does not change the hash."""
        a = self._make_contrib(tmp_path, "plateA", [("B/1", "0")])
        b = self._make_contrib(tmp_path, "plateB", [("C/2", "1")])
        h1 = _assembly_inputs_hash(
            contribs=[a, b],
            target="sec61b",
            condition="mock",
            output_channel_names=["Phase3D", "Structure"],
            center_crop_yx=None,
            target_yx_pixel_size_um=None,
        )
        h2 = _assembly_inputs_hash(
            contribs=[b, a],
            target="sec61b",
            condition="mock",
            output_channel_names=["Phase3D", "Structure"],
            center_crop_yx=None,
            target_yx_pixel_size_um=None,
        )
        assert h1 == h2

    def test_hash_changes_with_condition(self, tmp_path):
        """Different condition → different hash, all else equal."""
        c = self._make_contrib(tmp_path, "plateA", [("B/1", "0")])
        common = dict(
            contribs=[c],
            target="sec61b",
            output_channel_names=["Phase3D", "Structure"],
            center_crop_yx=None,
            target_yx_pixel_size_um=None,
        )
        h_mock = _assembly_inputs_hash(condition="mock", **common)
        h_zikv = _assembly_inputs_hash(condition="ZIKV", **common)
        assert h_mock != h_zikv

    def test_hash_pixel_size_only_when_set(self, tmp_path):
        """Null target pixel size is omitted; previously-published hashes stable."""
        c = self._make_contrib(tmp_path, "plateA", [("B/1", "0")])
        common = dict(
            contribs=[c],
            target="sec61b",
            condition="mock",
            output_channel_names=["Phase3D", "Structure"],
            center_crop_yx=(512, 512),
        )
        h_no_pix = _assembly_inputs_hash(target_yx_pixel_size_um=None, **common)
        h_with_pix = _assembly_inputs_hash(target_yx_pixel_size_um=0.1494, **common)
        assert h_no_pix != h_with_pix


class TestOutputSafety:
    """Refuse to write into per-plate source trees."""

    def test_rejects_under_plate_zarr_root(self, tmp_path):
        """Any output directory under plate_zarr_root is rejected."""
        src = tmp_path / "src"
        src.mkdir()
        out = src / "experiment" / "dynacell" / "run_x"
        out.mkdir(parents=True)
        with pytest.raises(ValueError, match="inside plate_zarr_root"):
            _check_output_safety(out, src)

    def test_accepts_vs_training_tree_containing_dynacell_segment(self, tmp_path):
        """Legitimate output path may contain 'dynacell' as a segment."""
        out = tmp_path / "virtual_staining" / "training" / "dynacell" / "a549"
        out.mkdir(parents=True)
        src = tmp_path / "organelle_dynamics"
        src.mkdir()
        _check_output_safety(out, src)


class TestTargetChannels:
    """Sanity check that canonical channel names match iPSC convention."""

    def test_sec61b_maps_to_structure(self):
        """SEC61B output channel matches iPSC 'Structure' slot."""
        assert CANONICAL_TARGET_CHANNELS["sec61b"] == "Structure"

    def test_h2b_maps_to_nuclei(self):
        """H2B output channel matches iPSC 'Nuclei' slot."""
        assert CANONICAL_TARGET_CHANNELS["h2b"] == "Nuclei"

    def test_caax_maps_to_membrane(self):
        """CAAX output channel matches iPSC 'Membrane' slot."""
        assert CANONICAL_TARGET_CHANNELS["caax"] == "Membrane"


class TestModuleConstants:
    """Top-level constants exported by the module are stable."""

    def test_gene_to_filename(self):
        """Single genes map to UPPERCASE; combined target keeps its key."""
        assert GENE_TO_FILENAME == {
            "sec61b": "SEC61B",
            "tomm20": "TOMM20",
            "h2b": "H2B",
            "caax": "CAAX",
            "dual_nucl_memb": "dual_nucl_memb",
        }

    def test_valid_conditions(self):
        """Three conditions: mock + two viral arms."""
        assert set(VALID_CONDITIONS) == {"mock", "ZIKV", "DENV"}


@pytest.mark.skipif(
    not (PLATE_ZARR_ROOT / PLATE_11_07 / "dynacell").exists(),
    reason="11_07 dynacell zarr not available (requires HPC data mount)",
)
@pytest.mark.skipif(
    os.environ.get("RUN_A549_E2E") != "1",
    reason="E2E pool assembly writes ~10 GB; set RUN_A549_E2E=1 to opt in",
)
class TestAssemblePoolEndToEnd:
    """HPC-gated + opt-in: assemble sec61b/mock pool from real plates.

    Uses the real authoring + dynacell zarrs; restricts to the
    11_07 plate which only carries sec61b mock + DENV. Output lives at
    ``A549_E2E_OUT`` (default ``/hpc/mydata/$USER/dynacell-a549-e2e``).
    """

    @pytest.fixture(scope="class")
    def assembled(self):
        """Run assemble_pool once; share the output across tests."""
        default_out = Path(f"/hpc/mydata/{os.environ.get('USER', 'unknown')}/dynacell-a549-e2e")
        out_root = Path(os.environ.get("A549_E2E_OUT", str(default_out)))
        if out_root.exists():
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        cfg = OmegaConf.create(
            {
                "targets": ["sec61b"],
                "conditions": ["mock", "DENV"],
                "plates": [PLATE_11_07],
                "plate_zarr_root": str(PLATE_ZARR_ROOT),
                "authoring_root": str(AUTHORING_ROOT),
                "output_root": str(out_root),
                "overwrite": True,
                "condition_format": "zarr",
                "pool_all_format": "zarr",
                "emit_train_pool_all": True,
                "t_cap": 2,
                "train_center_crop_yx": [128, 128],
                "test_center_crop_yx": [128, 128],
                "train_target_yx_pixel_size_um": None,
                "test_target_yx_pixel_size_um": None,
                "splits": ["train", "test"],
                "grid": {
                    "stride_h": 2.0,
                    "window": [5.0, 23.0],
                    "tail_tol_h": 1.5,
                },
            }
        )
        assemble_pool(cfg)
        return out_root

    def test_per_condition_stores_present(self, assembled):
        """Train + test condition stores exist (mock present, DENV present)."""
        for cond in ("mock", "DENV"):
            for split in ("train", "test"):
                p = assembled / split / f"SEC61B_{cond}.zarr"
                assert p.is_dir(), f"missing {p}"

    def test_pool_all_train_only(self, assembled):
        """``_all.zarr`` lands on the train side only."""
        assert (assembled / "train" / "SEC61B_all.zarr").is_dir()
        assert not (assembled / "test" / "SEC61B_all.zarr").exists()

    def test_provenance_sidecar(self, assembled):
        """Sidecar JSON lists per-position plate provenance."""
        sidecar = assembled / "train" / "SEC61B_mock.provenance.json"
        assert sidecar.is_file()
        meta = json.loads(sidecar.read_text())
        assert meta["target"] == "sec61b"
        assert meta["condition"] == "mock"
        assert meta["positions"]
        first = next(iter(meta["positions"].values()))
        assert first["plate_id"] == PLATE_11_07
        assert first["condition"] == "mock"
        assert first["source_position"]

    def test_pool_position_naming(self, assembled):
        """HCS positions inside the pool are sequential ``0/0/fov<NNNN>``."""
        store = assembled / "train" / "SEC61B_mock.zarr"
        with open_ome_zarr(store, mode="r", layout="hcs") as plate:
            names = [name for name, _ in plate.positions()]
        assert names, "empty pool"
        assert all(name.startswith("0/0/fov") for name in names)
        # Sequential indices, no gaps.
        idx = [int(name.split("fov")[-1]) for name in sorted(names)]
        assert idx == list(range(len(idx)))

    def test_pool_root_zattrs(self, assembled):
        """Pool store carries assembly_inputs_sha256 + assembly_target zattrs."""
        store = assembled / "train" / "SEC61B_mock.zarr"
        with open_ome_zarr(store, mode="r", layout="hcs") as plate:
            attrs = dict(plate.zattrs)
        assert attrs["assembly_target"] == "sec61b"
        assert attrs["assembly_condition"] == "mock"
        assert attrs["assembly_split"] == "train"
        assert attrs["assembly_t_cap"] == 2
        assert isinstance(attrs["assembly_inputs_sha256"], str)
        assert len(attrs["assembly_inputs_sha256"]) == 64
