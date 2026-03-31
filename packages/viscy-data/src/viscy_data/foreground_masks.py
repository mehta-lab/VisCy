"""Foreground mask support for SlidingWindowDataset.

Encapsulates all fg_mask/Spotlight logic as an optional collaborator
that SlidingWindowDataset delegates to when ``fg_mask_key`` is set.
"""

from collections.abc import Callable

from iohub.ngff import ImageArray, Position
from monai.transforms import CenterSpatialCropd, RandAffined, RandFlipd, RandWeightedCropd
from torch import Tensor

from viscy_transforms import (
    BatchedCenterSpatialCropd,
    BatchedRandAffined,
    BatchedRandFlipd,
    BatchedRandSpatialCropd,
)

# Spatial transforms that preserve pixel correspondence and must co-transform
# fg_mask alongside source/target. Intensity transforms are excluded — applying
# gamma correction or noise to a binary mask would corrupt it.
_SPATIAL_TRANSFORMS: tuple[type, ...] = (
    RandAffined,
    RandFlipd,
    CenterSpatialCropd,
    RandWeightedCropd,
    BatchedRandAffined,
    BatchedRandFlipd,
    BatchedCenterSpatialCropd,
    BatchedRandSpatialCropd,
)


class ForegroundMaskSupport:
    """Optional collaborator that adds fg_mask loading and spatial co-alignment.

    Instantiated by ``SlidingWindowDataset`` when ``fg_mask_key`` is set.
    Owns all mask-related state (array references, temp key names) so the
    dataset itself stays mask-agnostic.

    Parameters
    ----------
    fg_mask_key : str
        Zarr array key for precomputed foreground masks.
    target_channels : list[str]
        Target channel names (used for per-channel temp key naming).
    """

    def __init__(self, fg_mask_key: str, target_channels: list[str]) -> None:
        self.fg_mask_key = fg_mask_key
        self.target_channels = target_channels
        self._mask_keys = self.mask_temp_keys(target_channels)
        self._mask_arrays: list[ImageArray] = []

    @staticmethod
    def mask_temp_keys(target_channels: list[str]) -> tuple[str, ...]:
        """Return ``("__fg_mask_{ch}", ...)`` for the given channel names.

        Single source of truth for temp-key naming.  Used by
        ``ForegroundMaskSupport.__init__``, ``HCSDataModule._fit_transform``,
        and ``HCSDataModule._final_crop``.
        """
        return tuple(f"__fg_mask_{ch}" for ch in target_channels)

    @property
    def mask_keys(self) -> tuple[str, ...]:
        """Return precomputed ``("__fg_mask_{ch}", ...)`` tuple."""
        return self._mask_keys

    def validate_and_store(self, fov: Position) -> None:
        """Validate mask array exists in *fov* and store a reference.

        Called once per position during ``SlidingWindowDataset._get_windows``.
        The stored arrays are later looked up by index in ``read_window``.

        Parameters
        ----------
        fov : Position
            NGFF position to validate.
        """
        if self.fg_mask_key not in fov:
            raise FileNotFoundError(
                f"Mask array '{self.fg_mask_key}' not found in position. "
                "Run preprocessing with --compute_fg_masks first."
            )
        self._mask_arrays.append(fov[self.fg_mask_key])

    def read_window(
        self,
        arr_idx: int,
        target_ch_idx: list[int],
        tz: int,
        read_fn: Callable[..., tuple[list[Tensor], object]],
    ) -> list[Tensor]:
        """Read a mask window using the dataset's image reader.

        Parameters
        ----------
        arr_idx : int
            Index into the stored mask arrays (matches ``window_arrays`` order).
        target_ch_idx : list[int]
            Channel indices for target channels.
        tz : int
            Window index within the FOV (counted Z-first).
        read_fn : Callable
            The dataset's ``_read_img_window`` method.

        Returns
        -------
        list[Tensor]
            Per-channel mask tensors with shape ``(1, Z, Y, X)``.
        """
        mask_images, _ = read_fn(self._mask_arrays[arr_idx], target_ch_idx, tz)
        return mask_images

    def inject_into_sample(
        self,
        sample_images: dict[str, Tensor],
        mask_images: list[Tensor],
    ) -> None:
        """Inject per-channel mask tensors as temporary keys for spatial co-alignment.

        Must be called **before** the transform pipeline so that MONAI spatial
        transforms (e.g. ``CenterSpatialCropd``) co-transform the masks.

        Parameters
        ----------
        sample_images : dict[str, Tensor]
            Mutable sample dict to inject into.
        mask_images : list[Tensor]
            Per-channel mask tensors from ``read_window``.
        """
        for key, mask_tensor in zip(self._mask_keys, mask_images):
            sample_images[key] = mask_tensor

    @staticmethod
    def patch_spatial_transforms(
        transforms: list,
        target_keys: tuple[str, ...],
        mask_keys: tuple[str, ...],
    ) -> None:
        """Append mask keys to spatial transforms that operate on target keys.

        Only modifies transforms in the ``_SPATIAL_TRANSFORMS`` allowlist.
        Intensity transforms are never modified.  Idempotent — skips
        transforms that already contain the mask keys.

        Parameters
        ----------
        transforms : list
            Mutable list of transform instances to patch in place.
        target_keys : tuple[str, ...]
            Keys that identify target channels (e.g. channel names or
            ``"target"``).
        mask_keys : tuple[str, ...]
            Keys to append (e.g. ``("__fg_mask_Nuclei",)`` or
            ``("fg_mask",)``).
        """
        for t in transforms:
            if (
                isinstance(t, _SPATIAL_TRANSFORMS)
                and any(k in t.keys for k in target_keys)
                and not any(k in t.keys for k in mask_keys)
            ):
                t.keys = t.keys + mask_keys
                t.allow_missing_keys = True
