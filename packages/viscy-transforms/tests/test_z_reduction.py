import pytest
import torch

from viscy_transforms import BatchedChannelWiseZReduction, BatchedChannelWiseZReductiond


def _make_img(B=4, C=1, Z=11, Y=8, X=8):
    """Create a test image with distinct z-slices for easy verification."""
    img = torch.randn(B, C, Z, Y, X)
    return img


class TestBatchedChannelWiseZReduction:
    def test_mip_only(self):
        img = _make_img()
        reducer = BatchedChannelWiseZReduction(default_strategy="mip")
        out = reducer(img)
        assert out.shape == (4, 1, 1, 8, 8)
        expected = img.amax(dim=2, keepdim=True)
        torch.testing.assert_close(out, expected)

    def test_center_only(self):
        img = _make_img()
        reducer = BatchedChannelWiseZReduction(default_strategy="center")
        out = reducer(img)
        assert out.shape == (4, 1, 1, 8, 8)
        expected = img[:, :, 5:6]
        torch.testing.assert_close(out, expected)

    def test_mixed_mask(self):
        img = _make_img()
        mask = torch.tensor([True, False, True, False])
        reducer = BatchedChannelWiseZReduction()
        out = reducer(img, is_labelfree=mask)
        assert out.shape == (4, 1, 1, 8, 8)
        center = img[:, :, 5:6]
        mip = img.amax(dim=2, keepdim=True)
        torch.testing.assert_close(out[0], center[0])
        torch.testing.assert_close(out[1], mip[1])
        torch.testing.assert_close(out[2], center[2])
        torch.testing.assert_close(out[3], mip[3])

    def test_noop_z1(self):
        img = _make_img(Z=1)
        reducer = BatchedChannelWiseZReduction()
        out = reducer(img)
        assert out.shape == img.shape
        torch.testing.assert_close(out, img)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            BatchedChannelWiseZReduction(default_strategy="invalid")


class TestBatchedChannelWiseZReductiond:
    def test_bag_of_channels_with_mask(self):
        data = {
            "channel_0": _make_img(),
            "_is_labelfree": torch.tensor([True, False, False, True]),
        }
        transform = BatchedChannelWiseZReductiond(keys=["channel_0"])
        out = transform(data)
        assert out["channel_0"].shape == (4, 1, 1, 8, 8)
        assert "_is_labelfree" not in out

    def test_all_channels_with_labelfree_keys(self):
        phase_img = _make_img()
        fluor_img = _make_img()
        expected_center = phase_img[:, :, 5:6].clone()
        expected_mip = fluor_img.amax(dim=2, keepdim=True)
        data = {"Phase3D": phase_img, "TOMM20": fluor_img}
        transform = BatchedChannelWiseZReductiond(
            keys=["Phase3D", "TOMM20"],
            labelfree_keys=["Phase3D"],
        )
        out = transform(data)
        assert out["Phase3D"].shape == (4, 1, 1, 8, 8)
        assert out["TOMM20"].shape == (4, 1, 1, 8, 8)
        torch.testing.assert_close(out["Phase3D"], expected_center)
        torch.testing.assert_close(out["TOMM20"], expected_mip)

    def test_pops_is_labelfree(self):
        data = {
            "channel_0": _make_img(),
            "_is_labelfree": torch.tensor([False, False, False, False]),
        }
        transform = BatchedChannelWiseZReductiond(keys=["channel_0"])
        out = transform(data)
        assert "_is_labelfree" not in out

    def test_missing_keys(self):
        data = {"channel_0": _make_img()}
        transform = BatchedChannelWiseZReductiond(
            keys=["channel_0", "channel_1"],
            allow_missing_keys=True,
        )
        out = transform(data)
        assert out["channel_0"].shape == (4, 1, 1, 8, 8)
        assert "channel_1" not in out

    def test_noop_z1_dict(self):
        data = {"channel_0": _make_img(Z=1)}
        transform = BatchedChannelWiseZReductiond(keys=["channel_0"])
        out = transform(data)
        assert out["channel_0"].shape == (4, 1, 1, 8, 8)

    def test_no_mask_uses_default(self):
        img = _make_img()
        expected = img[:, :, 5:6].clone()
        data = {"channel_0": img}
        transform = BatchedChannelWiseZReductiond(keys=["channel_0"], default_strategy="center")
        out = transform(data)
        torch.testing.assert_close(out["channel_0"], expected)

    def test_labelfree_keys_noop_z1(self):
        data = {
            "Phase3D": _make_img(Z=1),
            "TOMM20": _make_img(Z=1),
        }
        transform = BatchedChannelWiseZReductiond(
            keys=["Phase3D", "TOMM20"],
            labelfree_keys=["Phase3D"],
        )
        out = transform(data)
        torch.testing.assert_close(out["Phase3D"], data["Phase3D"])
        torch.testing.assert_close(out["TOMM20"], data["TOMM20"])
