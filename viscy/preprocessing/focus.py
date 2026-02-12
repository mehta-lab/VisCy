from typing import Literal

import numpy as np
import tensorstore
import torch

from viscy.preprocessing.qc_metrics import QCMetric


def _estimate_batch_size(
    shape: tuple[int, ...],
    device: str | torch.device,
    memory_fraction: float = 0.7,
) -> int | None:
    """Estimate max timepoints per batch from available GPU memory.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape as (T, Z, Y, X).
    device : str or torch.device
        Target device. Returns None for CPU (no limit needed).
    memory_fraction : float
        Fraction of free GPU memory to use (default 0.7).

    Returns
    -------
    int or None
        Estimated batch size, or None if CPU.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return None

    free_mem, _ = torch.cuda.mem_get_info(device)
    usable = free_mem * memory_fraction

    T, Z, Y, X = shape
    # Per-timepoint memory: Z slices on device as float32 + FFT output (complex64)
    # float32 = 4 bytes, complex64 = 8 bytes -> ~12 bytes per element per slice
    bytes_per_timepoint = Z * Y * X * 12
    batch = max(1, int(usable // bytes_per_timepoint))
    return min(batch, T)


class FocusSliceMetric(QCMetric):
    """In-focus z-slice detection using midband spatial frequency power.

    Parameters
    ----------
    NA_det : float
        Detection numerical aperture.
    lambda_ill : float
        Illumination wavelength (same units as pixel_size).
    pixel_size : float
        Object-space pixel size (camera pixel size / magnification).
    channel_names : list[str] or -1
        Channel names to compute focus for. Use -1 for all channels.
    midband_fractions : tuple[float, float]
        Inner and outer fractions of cutoff frequency.
    device : str
        Torch device for FFT computation.
    batch_size : int or None
        Max timepoints to process at once. If None, automatically
        estimated from available GPU memory (or unlimited on CPU).
    """

    field_name = "focus_slice"

    def __init__(
        self,
        NA_det: float,
        lambda_ill: float,
        pixel_size: float,
        channel_names: list[str] | Literal[-1] = -1,
        midband_fractions: tuple[float, float] = (0.125, 0.25),
        device: str = "cpu",
        batch_size: int | None = None,
    ):
        self.NA_det = NA_det
        self.lambda_ill = lambda_ill
        self.pixel_size = pixel_size
        self.channel_names = channel_names
        self.midband_fractions = midband_fractions
        self.device = device
        self.batch_size = batch_size

    def channels(self) -> list[str] | Literal[-1]:
        return self.channel_names

    def __call__(self, position, channel_name, channel_index, num_workers=4):
        from waveorder.focus import compute_focus_slice_batch

        tzyx = (
            position["0"]
            .tensorstore(
                context=tensorstore.Context(
                    {"data_copy_concurrency": {"limit": num_workers}}
                )
            )[:, channel_index]
            .read()
            .result()
        )

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = _estimate_batch_size(tzyx.shape, self.device)

        focus_indices = compute_focus_slice_batch(
            tzyx,
            NA_det=self.NA_det,
            lambda_ill=self.lambda_ill,
            pixel_size=self.pixel_size,
            midband_fractions=self.midband_fractions,
            device=self.device,
            batch_size=batch_size,
        )

        if isinstance(focus_indices, int):
            focus_indices = np.array([focus_indices])

        per_timepoint = {str(t): int(idx) for t, idx in enumerate(focus_indices)}
        fov_stats = {
            "z_focus_mean": float(np.mean(focus_indices)),
            "z_focus_std": float(np.std(focus_indices)),
        }
        return {
            "fov_statistics": fov_stats,
            "per_timepoint": per_timepoint,
        }
