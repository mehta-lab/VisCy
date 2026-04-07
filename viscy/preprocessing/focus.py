import warnings
from typing import Literal, Union

import numpy as np
import tensorstore
import torch

from viscy.preprocessing.qc_metrics import QCMetric


def _gen_coordinate(
    img_dim: tuple[int, int],
    ps: float,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N, M = img_dim
    ps = float(ps)

    fx = torch.fft.fftfreq(M, ps, device=device)
    fy = torch.fft.fftfreq(N, ps, device=device)
    x = torch.fft.ifftshift(
        (torch.arange(M, dtype=torch.float32, device=device) - M / 2) * ps
    )
    y = torch.fft.ifftshift(
        (torch.arange(N, dtype=torch.float32, device=device) - N / 2) * ps
    )

    xx, yy = torch.meshgrid(x, y, indexing="xy")
    fxx, fyy = torch.meshgrid(fx, fy, indexing="xy")

    return (xx, yy, fxx, fyy)


def _compute_midband_power(
    input_tensor: torch.Tensor,
    NA_det: float,
    lambda_ill: float,
    pixel_size: float,
    midband_fractions: tuple[float, float] = (0.125, 0.25),
) -> torch.Tensor:
    if input_tensor.ndim == 2:
        input_tensor = input_tensor.unsqueeze(0)
        squeeze = True
    elif input_tensor.ndim == 3:
        squeeze = False
    else:
        raise ValueError(
            f"{input_tensor.ndim}D tensor supplied. `_compute_midband_power` only accepts 2D or 3D tensors."
        )

    Z, Y, X = input_tensor.shape
    device = input_tensor.device

    _, _, fxx, fyy = _gen_coordinate((Y, X), pixel_size, device=device)
    frr = torch.sqrt(fxx**2 + fyy**2)
    cutoff = 2 * NA_det / lambda_ill
    mask = torch.logical_and(
        frr > cutoff * midband_fractions[0],
        frr < cutoff * midband_fractions[1],
    )

    abs_fft = torch.abs(torch.fft.fftn(input_tensor.float(), dim=(-2, -1)))
    result = abs_fft[:, mask].sum(dim=-1)

    if squeeze:
        return result.squeeze(0)
    return result


def _focus_from_transverse_band(
    zyx_array: Union[np.ndarray, torch.Tensor],
    NA_det: float,
    lambda_ill: float,
    pixel_size: float,
    midband_fractions: tuple[float, float] = (0.125, 0.25),
    mode: Literal["min", "max"] = "max",
) -> int:
    """Estimate the in-focus slice from a 3D stack using midband spatial frequency power.

    Parameters
    ----------
    zyx_array : np.ndarray or torch.Tensor
        Data stack in (Z, Y, X) order.
    NA_det : float
        Detection NA.
    lambda_ill : float
        Illumination wavelength (same units as pixel_size).
    pixel_size : float
        Object-space pixel size = camera pixel size / magnification.
    midband_fractions : tuple[float, float]
        Inner and outer fractions of the cutoff frequency defining the midband.
    mode : {'min', 'max'}
        Whether to find the slice with minimum or maximum midband power.

    Returns
    -------
    int
        Index of the in-focus slice.
    """
    N = len(zyx_array.shape)
    if N != 3:
        raise ValueError(f"{N}D array supplied. Only 3D arrays are accepted.")

    if zyx_array.shape[0] == 1:
        warnings.warn(
            "The dataset only contained a single slice. Returning trivial slice index = 0."
        )
        return 0

    if isinstance(zyx_array, np.ndarray):
        zyx_tensor = torch.from_numpy(zyx_array).float()
    else:
        zyx_tensor = zyx_array.float()

    midband_sum = (
        _compute_midband_power(
            zyx_tensor, NA_det, lambda_ill, pixel_size, midband_fractions
        )
        .cpu()
        .numpy()
    )

    if mode == "max":
        return int(np.argmax(midband_sum))
    elif mode == "min":
        return int(np.argmin(midband_sum))
    else:
        raise ValueError("mode must be either 'min' or 'max'")


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
    ):
        self.NA_det = NA_det
        self.lambda_ill = lambda_ill
        self.pixel_size = pixel_size
        self.channel_names = channel_names
        self.midband_fractions = midband_fractions
        self.device = device

    def channels(self) -> list[str] | Literal[-1]:
        return self.channel_names

    def __call__(self, position, channel_name, channel_index, num_workers=4):
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

        T = tzyx.shape[0]
        focus_indices = np.empty(T, dtype=int)

        for t in range(T):
            zyx = torch.from_numpy(np.asarray(tzyx[t])).to(self.device)
            focus_indices[t] = _focus_from_transverse_band(
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
