import numpy as np
import tensorstore

from viscy.preprocessing.qc_metrics import QCMetric


class FocusSliceMetric(QCMetric):
    """In-focus z-slice detection using midband spatial frequency power.

    Parameters
    ----------
    channel_params : dict[str, dict]
        Per-channel optical parameters. Keys are channel names.
        Each value must have: NA_det, lambda_ill, pixel_size.
        Optional: midband_fractions (default (0.125, 0.25)).
        Example::

            {
                "Phase": {
                    "NA_det": 0.55,
                    "lambda_ill": 0.532,
                    "pixel_size": 0.325,
                },
                "GFP": {
                    "NA_det": 1.2,
                    "lambda_ill": 0.488,
                    "pixel_size": 0.103,
                },
            }
    device : str
        Torch device for FFT computation.
    batch_size : int or None
        Max timepoints to process at once (GPU memory control).
    """

    field_name = "focus_slice"

    def __init__(
        self,
        channel_params: dict[str, dict],
        device: str = "cpu",
        batch_size: int | None = None,
    ):
        self.channel_params = channel_params
        self.device = device
        self.batch_size = batch_size

    def channels(self) -> list[str]:
        return list(self.channel_params.keys())

    def __call__(self, position, channel_name, channel_index, num_workers=4):
        from waveorder.focus import compute_focus_slice_batch

        params = self.channel_params[channel_name]

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

        focus_indices = compute_focus_slice_batch(
            tzyx,
            NA_det=params["NA_det"],
            lambda_ill=params["lambda_ill"],
            pixel_size=params["pixel_size"],
            midband_fractions=params.get("midband_fractions", (0.125, 0.25)),
            device=self.device,
            batch_size=self.batch_size,
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
