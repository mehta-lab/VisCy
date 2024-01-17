from viscy.light.engine import FcmaeUNet


def test_fcmae_vsunet() -> None:
    model = FcmaeUNet(
        architecture="fcmae",
        model_config=dict(in_channels=3),
        train_mask_ratio=0.6,
    )

