from monai.transforms import (
    MapTransform,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandWeightedCropd,
)


def get_augmentations(
    source: list[str], target: list[str], noise_std: float = 2.0
) -> list[MapTransform]:
    return [
        RandWeightedCropd(
            source + target, target[0], spatial_size=[-1, 768, 768], num_samples=2
        ),
        RandAffined(
            source + target,
            prob=0.5,
            rotate_range=[3.14, 0.0, 0.0],
            shear_range=[0.0, 0.05, 0.05],
            scale_range=[0.2, 0.3, 0.3],
        ),
        RandAdjustContrastd(source, prob=0.3, gamma=[0.75, 1.5]),
        RandScaleIntensityd(source, prob=0.3, factors=0.5),
        RandGaussianNoised(source, prob=0.5, mean=0.0, std=noise_std),
        RandGaussianSmoothd(
            source,
            prob=0.5,
            sigma_z=[0.25, 1.5],
            sigma_y=[0.25, 1.5],
            sigma_x=[0.25, 1.5],
        ),
    ]
