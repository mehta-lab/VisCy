"""Tests for tensor conversion utilities."""

import numpy as np
import pytest
import torch

from viscy_utils.tensor_utils import to_numpy


class TestToNumpy:
    def test_bf16_to_float32(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        result = to_numpy(t)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-2)

    def test_fp16_to_float32(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        result = to_numpy(t)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-3)

    def test_fp32_passthrough(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = to_numpy(t)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_fp64_to_float32(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = to_numpy(t)
        assert result.dtype == np.float32

    def test_int_preserved(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = to_numpy(t)
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_int64_preserved(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int64)
        result = to_numpy(t)
        assert result.dtype == np.int64

    def test_bool_preserved(self):
        t = torch.tensor([True, False, True])
        result = to_numpy(t)
        assert result.dtype == np.bool_
        np.testing.assert_array_equal(result, [True, False, True])

    def test_cpu_tensor(self):
        t = torch.rand(3, 4)
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 4)

    def test_noncontiguous_tensor(self):
        t = torch.rand(3, 4).permute(1, 0)
        assert not t.is_contiguous()
        result = to_numpy(t)
        assert result.shape == (4, 3)

    def test_5d_image_tensor(self):
        t = torch.rand(2, 1, 8, 256, 256, dtype=torch.bfloat16)
        result = to_numpy(t)
        assert result.dtype == np.float32
        assert result.shape == (2, 1, 8, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_to_cpu(self):
        t = torch.rand(3, 4, device="cuda", dtype=torch.bfloat16)
        result = to_numpy(t)
        assert result.dtype == np.float32
        assert result.shape == (3, 4)

    def test_detaches_from_graph(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
