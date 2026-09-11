"""Tests for dynacell.preprocess.segmentation."""

from unittest.mock import MagicMock, patch

import numpy as np
from iohub.ngff import open_ome_zarr

from dynacell.preprocess.segmentation import run_cellpose_segmentation


def _create_test_zarr(path, channel_names, data):
    """Create a minimal OME-Zarr store for testing."""
    with open_ome_zarr(
        path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
        version="0.4",
    ) as dataset:
        pos = dataset.create_position("A", "1", "0")
        pos.create_image("0", data=data)


class TestRunCellposeSegmentation:
    """Tests for the run_cellpose_segmentation function."""

    def test_3d_passes_correct_kwargs(self, tmp_path):
        """3D mode passes z_axis, batch_size, do_3D, flow3D_smooth to model.eval."""
        input_path = tmp_path / "input.zarr"
        output_path = tmp_path / "output.zarr"
        data = np.random.default_rng(0).random((1, 2, 4, 8, 8)).astype(np.float32)
        _create_test_zarr(input_path, ["Phase3D", "Nuclei"], data)

        D, H, W = 4, 8, 8
        mock_model = MagicMock()
        mock_model.eval.return_value = (
            [np.zeros((D, H, W), dtype=np.uint8)],
            None,
            None,
        )

        with patch("cellpose.models.CellposeModel", return_value=mock_model):
            run_cellpose_segmentation(
                data_dir=input_path,
                save_dir=output_path,
                input_channel_names=["Phase3D", "Nuclei"],
                do_3d=True,
                use_gpu=False,
                batch_size=16,
                z_axis=2,
                flow_3d_smooth=5,
            )

        call_kwargs = mock_model.eval.call_args
        assert call_kwargs.kwargs["channel_axis"] == 0
        assert call_kwargs.kwargs["do_3D"] is True
        assert call_kwargs.kwargs["z_axis"] == 2
        assert call_kwargs.kwargs["batch_size"] == 16
        assert call_kwargs.kwargs["flow3D_smooth"] == 5

    def test_label_overflow_raises_instead_of_wrapping(self, tmp_path):
        """>255 labels must raise under the uint8 default, not wrap modulo 256.

        Cellpose returns an int32 label image and `astype` narrows by wrapping,
        so label 256 became background and 257 merged with a distant cell. The
        existing shape guard cannot see that, and instance-AP downstream would
        score the corrupted field as if it were valid.
        """
        import pytest

        input_path = tmp_path / "input.zarr"
        output_path = tmp_path / "output.zarr"
        data = np.random.default_rng(0).random((1, 2, 4, 16, 16)).astype(np.float32)
        _create_test_zarr(input_path, ["Phase3D", "Nuclei"], data)

        D, H, W = 4, 16, 16
        labels = np.zeros((D, H, W), dtype=np.int32)
        labels.reshape(-1)[:300] = np.arange(1, 301)  # 300 distinct instances
        mock_model = MagicMock()
        mock_model.eval.return_value = ([labels], None, None)

        with patch("cellpose.models.CellposeModel", return_value=mock_model):
            with pytest.raises(ValueError, match="does not fit output_dtype"):
                run_cellpose_segmentation(
                    data_dir=input_path,
                    save_dir=output_path,
                    input_channel_names=["Phase3D", "Nuclei"],
                    do_3d=True,
                    use_gpu=False,
                )

    def test_uint16_accepts_more_than_255_labels(self, tmp_path):
        """The documented escape hatch works: uint16 carries the labels through."""
        input_path = tmp_path / "input.zarr"
        output_path = tmp_path / "output.zarr"
        data = np.random.default_rng(0).random((1, 2, 4, 16, 16)).astype(np.float32)
        _create_test_zarr(input_path, ["Phase3D", "Nuclei"], data)

        D, H, W = 4, 16, 16
        labels = np.zeros((D, H, W), dtype=np.int32)
        labels.reshape(-1)[:300] = np.arange(1, 301)
        mock_model = MagicMock()
        mock_model.eval.return_value = ([labels], None, None)

        with patch("cellpose.models.CellposeModel", return_value=mock_model):
            run_cellpose_segmentation(
                data_dir=input_path,
                save_dir=output_path,
                input_channel_names=["Phase3D", "Nuclei"],
                do_3d=True,
                use_gpu=False,
                output_dtype="uint16",
            )

        with open_ome_zarr(output_path, mode="r") as out:
            written = np.asarray(out["A/1/0"]["0"][0, 0])
        assert int(written.max()) == 300

    def test_2d_does_max_projection(self, tmp_path):
        """2D mode calls model.eval with niter and repeats masks across depth."""
        input_path = tmp_path / "input.zarr"
        output_path = tmp_path / "output.zarr"
        data = np.random.default_rng(0).random((1, 2, 4, 8, 8)).astype(np.float32)
        _create_test_zarr(input_path, ["Phase3D", "Nuclei"], data)

        H, W = 8, 8
        mock_model = MagicMock()
        mock_model.eval.return_value = (
            [np.zeros((H, W), dtype=np.uint8)],
            None,
            None,
        )

        with patch("cellpose.models.CellposeModel", return_value=mock_model):
            run_cellpose_segmentation(
                data_dir=input_path,
                save_dir=output_path,
                input_channel_names=["Phase3D", "Nuclei"],
                do_3d=False,
                use_gpu=False,
                niter_2d=500,
            )

        call_kwargs = mock_model.eval.call_args
        assert call_kwargs.kwargs["channel_axis"] == 0
        assert call_kwargs.kwargs["niter"] == 500
        assert "do_3D" not in call_kwargs.kwargs

    def test_output_uses_context_manager(self, tmp_path):
        """Output zarr store is properly closed after function returns."""
        input_path = tmp_path / "input.zarr"
        output_path = tmp_path / "output.zarr"
        data = np.random.default_rng(0).random((1, 2, 4, 8, 8)).astype(np.float32)
        _create_test_zarr(input_path, ["Phase3D", "Nuclei"], data)

        D, H, W = 4, 8, 8
        mock_model = MagicMock()
        mock_model.eval.return_value = (
            [np.zeros((D, H, W), dtype=np.uint8)],
            None,
            None,
        )

        with patch("cellpose.models.CellposeModel", return_value=mock_model):
            run_cellpose_segmentation(
                data_dir=input_path,
                save_dir=output_path,
                input_channel_names=["Phase3D", "Nuclei"],
                do_3d=True,
                use_gpu=False,
            )

        # If context manager worked, we can reopen the output store
        with open_ome_zarr(output_path, mode="r", layout="hcs") as ds:
            positions = list(ds.positions())
            assert len(positions) == 1
            _, pos = positions[0]
            assert pos["0"].shape == (1, 1, 4, 8, 8)
