"""In-focus z-slice detection using midband spatial frequency power."""

import numpy as np
import torch
from waveorder.focus import focus_from_transverse_band

from qc.qc_metrics import QCMetric


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
    channel_names : list[str]
        Channel names to compute focus for.
    midband_fractions : tuple[float, float]
        Inner and outer fractions of cutoff frequency.
    device : str
        Torch device for FFT computation (e.g. "cpu", "cuda").
    """

    field_name = "focus_slice"

    def __init__(
        self,
        NA_det: float,
        lambda_ill: float,
        pixel_size: float,
        channel_names: list[str],
        midband_fractions: tuple[float, float] = (0.125, 0.25),
        device: str = "cpu",
    ):
        self.NA_det = NA_det
        self.lambda_ill = lambda_ill
        self.pixel_size = pixel_size
        self.channel_names = channel_names
        self.midband_fractions = midband_fractions
        self.device = torch.device(device)

    def channels(self) -> list[str]:
        """Return the channels this metric is configured for."""
        return self.channel_names

    def __call__(self, position, channel_name, channel_index, num_workers=4):
        """Compute focus-slice index per timepoint for one channel of ``position``."""
        # Tensorstore concurrency is configured on the plate at
        # open-time (see qc_metrics.run_qc_metrics); num_workers is
        # retained here only to match the QCMetric abstract interface.
        del num_workers
        tzyx = position["0"].native[:, channel_index].read().result()

        T = tzyx.shape[0]
        focus_indices = np.empty(T, dtype=int)

        for t in range(T):
            zyx = torch.as_tensor(np.asarray(tzyx[t]), device=self.device)
            focus_indices[t] = focus_from_transverse_band(
                zyx,
                NA_det=self.NA_det,
                lambda_ill=self.lambda_ill,
                pixel_size=self.pixel_size,
                midband_fractions=self.midband_fractions,
            )

        per_timepoint = {str(t): int(idx) for t, idx in enumerate(focus_indices)}
        fov_stats = {
            "z_focus_mean": float(np.mean(focus_indices)),
            "z_focus_std": float(np.std(focus_indices)),
        }
        return {
            "fov_statistics": fov_stats,
            "per_timepoint": per_timepoint,
        }

    def aggregate_dataset(self, all_results: list[dict]) -> dict:
        """Compute dataset-level focus statistics across all positions.

        Parameters
        ----------
        all_results : list[dict]
            List of dicts returned by ``__call__`` for each position.

        Returns
        -------
        dict
            Dataset-level z-focus statistics.
        """
        all_values = []
        for result in all_results:
            all_values.extend(result["per_timepoint"].values())
        arr = np.array(all_values, dtype=float)
        return {
            "z_focus_mean": float(np.mean(arr)),
            "z_focus_std": float(np.std(arr)),
            "z_focus_min": int(np.min(arr)),
            "z_focus_max": int(np.max(arr)),
        }
