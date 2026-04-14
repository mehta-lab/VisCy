import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_1d(kernel_size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g


def _create_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    channels: int,
    spatial_dims: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Create a Gaussian kernel for SSIM computation.

    Returns
    -------
    torch.Tensor
        2D: [C, 1, k, k]
        3D: [C, 1, k, k, k]
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    g1 = _gaussian_1d(kernel_size, sigma, device=device, dtype=dtype)

    if spatial_dims == 2:
        kernel = (g1[:, None] * g1[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
    else:
        kernel = (g1[:, None, None] * g1[None, :, None] * g1[None, None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,k,k,k]

    kernel = kernel.expand(channels, 1, *kernel.shape[2:]).contiguous()
    return kernel


def _ssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    kernel_size: int,
    spatial_dims: int,
    data_range: float,
    k1: float,
    k2: float,
) -> torch.Tensor:
    channels = x.shape[1]
    conv = F.conv2d if spatial_dims == 2 else F.conv3d

    padding = kernel_size // 2
    if spatial_dims == 2:
        pad_tuple = (padding, padding, padding, padding)
    else:
        pad_tuple = (padding, padding, padding, padding, padding, padding)  # type: ignore[assignment]

    x_pad = F.pad(x, pad_tuple, mode="replicate")
    y_pad = F.pad(y, pad_tuple, mode="replicate")

    mu_x = conv(x_pad, kernel, padding=0, groups=channels)
    mu_y = conv(y_pad, kernel, padding=0, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = conv(x_pad * x_pad, kernel, padding=0, groups=channels) - mu_x_sq
    sigma_y_sq = conv(y_pad * y_pad, kernel, padding=0, groups=channels) - mu_y_sq
    sigma_xy = conv(x_pad * y_pad, kernel, padding=0, groups=channels) - mu_xy

    sigma_x_sq = F.relu(sigma_x_sq)
    sigma_y_sq = F.relu(sigma_y_sq)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return ssim_map


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    spatial_dims: int | None = None,
    reduction: str = "mean",
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Compute SSIM for 2D or 3D tensors.

    Parameters
    ----------
    x, y : torch.Tensor
        2D: [N, C, H, W]
        3D: [N, C, D, H, W]
    kernel_size : int
        Odd integer.
    sigma : float
        Gaussian sigma.
    data_range : float
        Value range of input (e.g. 1.0 or 255.0).
    spatial_dims : int or None
        2 or 3. If None, inferred from input ndim.
    reduction : str
        - "mean": return scalar
        - "none": return per-sample tensor [N]
    k1, k2 : float
        SSIM constants.

    Returns
    -------
    torch.Tensor
        Scalar if reduction="mean", [N] if reduction="none".
    """
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x {x.shape} vs y {y.shape}")

    if x.ndim not in (4, 5):
        raise ValueError(f"Expected 4D or 5D input, got x.ndim={x.ndim}")

    if spatial_dims is None:
        spatial_dims = x.ndim - 2  # 4D->2, 5D->3

    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    expected_ndim = spatial_dims + 2
    if x.ndim != expected_ndim:
        raise ValueError(
            f"Input ndim ({x.ndim}) does not match spatial_dims={spatial_dims}; expected ndim={expected_ndim}"
        )

    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction}")

    x = x.float()
    y = y.float()

    channels = x.shape[1]
    kernel = _create_gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=channels,
        spatial_dims=spatial_dims,
        device=x.device,
        dtype=x.dtype,
    )

    ssim_map = _ssim_per_channel(
        x=x,
        y=y,
        kernel=kernel,
        kernel_size=kernel_size,
        spatial_dims=spatial_dims,
        data_range=data_range,
        k1=k1,
        k2=k2,
    )

    if reduction == "mean":
        return ssim_map.mean()

    # reduction == "none" -> per-sample [N]
    reduce_dims = tuple(range(1, ssim_map.ndim))  # average over C and spatial dims
    return ssim_map.mean(dim=reduce_dims)


class SSIM(nn.Module):
    """nn.Module wrapper for 2D / 3D SSIM.

    Examples
    --------
    >>> metric_2d = SSIM(spatial_dims=2)
    >>> metric_3d = SSIM(spatial_dims=3)
    >>> val = metric_2d(x2d, y2d)  # x2d: [N,C,H,W]
    >>> val = metric_3d(x3d, y3d)  # x3d: [N,C,D,H,W]
    """

    def __init__(
        self,
        spatial_dims: int,
        kernel_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        reduction: str = "mean",
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction}")

        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.data_range = data_range
        self.reduction = reduction
        self.k1 = k1
        self.k2 = k2

        self._cached_channels = None
        self.register_buffer("_kernel", torch.empty(0), persistent=False)

    def _get_kernel(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        need_rebuild = (
            self._kernel.numel() == 0
            or self._cached_channels != channels
            or self._kernel.device != x.device
            or self._kernel.dtype != x.dtype
        )

        if need_rebuild:
            new_kernel = _create_gaussian_kernel(
                kernel_size=self.kernel_size,
                sigma=self.sigma,
                channels=channels,
                spatial_dims=self.spatial_dims,
                device=x.device,
                dtype=x.dtype,
            )
            self.register_buffer("_kernel", new_kernel, persistent=False)
            self._cached_channels = channels

        return self._kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: x {x.shape} vs y {y.shape}")

        expected_ndim = self.spatial_dims + 2
        if x.ndim != expected_ndim:
            raise ValueError(f"Expected {expected_ndim}D input for spatial_dims={self.spatial_dims}, got {x.ndim}D")

        x = x.float()
        y = y.float()

        kernel = self._get_kernel(x)

        ssim_map = _ssim_per_channel(
            x=x,
            y=y,
            kernel=kernel,
            kernel_size=self.kernel_size,
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            k1=self.k1,
            k2=self.k2,
        )

        if self.reduction == "mean":
            return ssim_map.mean()

        reduce_dims = tuple(range(1, ssim_map.ndim))
        return ssim_map.mean(dim=reduce_dims)
