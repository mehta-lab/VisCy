"""Tests for dynacell.preprocess.zarr_utils."""

import pytest

np = pytest.importorskip("numpy")
open_ome_zarr = pytest.importorskip("iohub.ngff").open_ome_zarr

from dynacell.preprocess.zarr_utils import rewrite_zarr  # noqa: E402


def _create_test_zarr(path, channel_names, data, chunks=None):
    """Create a minimal OME-Zarr store for testing."""
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=channel_names, version="0.4") as dataset:
        pos = dataset.create_position("A", "1", "0")
        kwargs = {}
        if chunks is not None:
            kwargs["chunks"] = chunks
        pos.create_image("0", data=data, **kwargs)


class TestRewriteZarr:
    """Tests for the rewrite_zarr function."""

    def test_creates_output(self, tmp_path):
        """Rewriting creates output store with correct chunks."""
        input_path = tmp_path / "input.zarr"
        output_path = tmp_path / "output.zarr"
        data = np.random.rand(1, 2, 4, 8, 8).astype(np.float32)
        target_chunks = (1, 1, 2, 4, 4)
        _create_test_zarr(input_path, ["ch0", "ch1"], data)
        rewrite_zarr(input_path, output_path, chunks=target_chunks)
        assert output_path.exists()
        with open_ome_zarr(output_path, mode="r", layout="hcs") as ds:
            positions = list(ds.positions())
            assert len(positions) == 1
            _, pos = positions[0]
            assert pos["0"].chunks == target_chunks

    def test_preserves_data(self, tmp_path):
        """Array data is identical after rewriting."""
        input_path = tmp_path / "input.zarr"
        output_path = tmp_path / "output.zarr"
        data = np.random.rand(1, 2, 4, 8, 8).astype(np.float32)
        _create_test_zarr(input_path, ["ch0", "ch1"], data)
        rewrite_zarr(input_path, output_path, chunks=(1, 1, 2, 4, 4))
        with open_ome_zarr(output_path, mode="r", layout="hcs") as ds:
            _, pos = list(ds.positions())[0]
            np.testing.assert_array_equal(pos["0"].numpy(), data)

    def test_preserves_metadata(self, tmp_path):
        """Channel names and coordinate transforms are copied."""
        input_path = tmp_path / "input.zarr"
        output_path = tmp_path / "output.zarr"
        channel_names = ["Phase3D", "Nuclei", "Membrane"]
        data = np.random.rand(1, 3, 4, 8, 8).astype(np.float32)
        _create_test_zarr(input_path, channel_names, data)
        rewrite_zarr(input_path, output_path, chunks=(1, 1, 2, 4, 4))
        with open_ome_zarr(output_path, mode="r", layout="hcs") as ds:
            assert ds.channel_names == channel_names
            _, pos = list(ds.positions())[0]
            transforms = pos.metadata.multiscales[0].datasets[0].coordinate_transformations
            assert transforms is not None

    def test_custom_shards(self, tmp_path):
        """Sharding ratio is applied correctly to the output store."""
        input_path = tmp_path / "input.zarr"
        output_path = tmp_path / "output.zarr"
        data = np.random.rand(1, 2, 4, 8, 8).astype(np.float32)
        target_chunks = (1, 1, 2, 4, 4)
        shards = (1, 1, 2, 2, 2)
        expected_shard_size = tuple(c * s for c, s in zip(target_chunks, shards))
        _create_test_zarr(input_path, ["ch0", "ch1"], data)
        rewrite_zarr(input_path, output_path, chunks=target_chunks, shards_ratio=shards)
        with open_ome_zarr(output_path, mode="r", layout="hcs") as ds:
            _, pos = list(ds.positions())[0]
            assert pos["0"].chunks == target_chunks
            assert pos["0"].shards == expected_shard_size
            np.testing.assert_array_equal(pos["0"].numpy(), data)
