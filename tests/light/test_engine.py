from viscy.light.engine import FcmaeUNet


def test_fcmae_vsunet() -> None:
    model = FcmaeUNet(
        model_config=dict(in_channels=3, out_channels=1), fit_mask_ratio=0.6
    )
