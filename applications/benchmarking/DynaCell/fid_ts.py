import argparse
from pathlib import Path

import torch
from iohub.ngff import open_ome_zarr
from torch import Tensor
from tqdm import tqdm

# ----------------------------------------------------------------------------- #
#                              Helper functions                                 #
# ----------------------------------------------------------------------------- #

def read_zarr(zarr_path: str):
    plate = open_ome_zarr(zarr_path, mode="r")
    return [pos for _, pos in plate.positions()]

def normalise(volume: torch.Tensor) -> torch.Tensor:
    """Per-sample min max → [-1,1]. Shape: (D, H, W) or (B, D, H, W)."""
    v_min = volume.amin(dim=(-3, -2, -1), keepdim=True)
    v_max = volume.amax(dim=(-3, -2, -1), keepdim=True)
    volume = (volume - v_min) / (v_max - v_min + 1e-6)        # → [0,1]
    return volume * 2.0 - 1.0                                 # → [-1,1]

@torch.jit.script_if_tracing
def sqrtm(sigma: Tensor) -> Tensor:
    r"""Returns the square root of a positive semi-definite matrix.

    .. math:: \sqrt{\Sigma} = Q \sqrt{\Lambda} Q^T

    where :math:`Q \Lambda Q^T` is the eigendecomposition of :math:`\Sigma`.

    Args:
        sigma: A positive semi-definite matrix, :math:`(*, D, D)`.

    Example:
        >>> V = torch.randn(4, 4, dtype=torch.double)
        >>> A = V @ V.T
        >>> B = sqrtm(A @ A)
        >>> torch.allclose(A, B)
        True
    """

    L, Q = torch.linalg.eigh(sigma)
    L = L.relu().sqrt()

    return Q @ (L[..., None] * Q.mT)

@torch.jit.script_if_tracing
def frechet_distance(
    mu_x: Tensor,
    sigma_x: Tensor,
    mu_y: Tensor,
    sigma_y: Tensor,
) -> Tensor:
    r"""Returns the Fréchet distance between two multivariate Gaussian distributions.

    .. math:: d^2 = \left\| \mu_x - \mu_y \right\|_2^2 +
        \operatorname{tr} \left( \Sigma_x + \Sigma_y - 2 \sqrt{\Sigma_y^{\frac{1}{2}} \Sigma_x \Sigma_y^{\frac{1}{2}}} \right)

    Wikipedia:
        https://wikipedia.org/wiki/Frechet_distance

    Args:
        mu_x: The mean :math:`\mu_x` of the first distribution, :math:`(*, D)`.
        sigma_x: The covariance :math:`\Sigma_x` of the first distribution, :math:`(*, D, D)`.
        mu_y: The mean :math:`\mu_y` of the second distribution, :math:`(*, D)`.
        sigma_y: The covariance :math:`\Sigma_y` of the second distribution, :math:`(*, D, D)`.

    Example:
        >>> mu_x = torch.arange(3).float()
        >>> sigma_x = torch.eye(3)
        >>> mu_y = 2 * mu_x + 1
        >>> sigma_y = 2 * sigma_x + 1
        >>> frechet_distance(mu_x, sigma_x, mu_y, sigma_y)
        tensor(15.8710)
    """

    sigma_y_12 = sqrtm(sigma_y)

    a = (mu_x - mu_y).square().sum(dim=-1)
    b = sigma_x.trace() + sigma_y.trace()
    c = sqrtm(sigma_y_12 @ sigma_x @ sigma_y_12).trace()

    return a + b - 2 * c

@torch.no_grad()
def fid_from_features(f1, f2, eps=1e-6):
    mu1, sigma1 = f1.mean(0), torch.cov(f1.T)
    mu2, sigma2 = f2.mean(0), torch.cov(f2.T)

    eye = torch.eye(sigma1.size(0), device=sigma1.device, dtype=sigma1.dtype)
    sigma1 = sigma1 + eps * eye
    sigma2 = sigma2 + eps * eye

    return frechet_distance(mu1, sigma1, mu2, sigma2).clamp_min_(0).item()

@torch.no_grad()
def encode_fovs(
    fovs,
    vae,
    channel_name: str,
    device: str = "cuda",
    batch_size: int = 4,
    input_spatial_size: tuple = (32, 512, 512), 
):
    """
    For each FOV pair:
        • take all T time-frames  (shape: T, D, H, W)
        • normalise to [-1, 1]
        • feed through VAE in chunks of ≤ batch_size frames
        • average the resulting T latent vectors  →  one embedding / FOV
    Returns
        emb: (N, latent_dim) tensors
    """
    emb = []

    for pos in tqdm(fovs, desc="Encoding FOVs"):
        # ---------------- load & normalise ---------------- #
        v = torch.as_tensor(
            pos.data[:, pos.get_channel_index(channel_name)],
            dtype=torch.float32, device=device,
        )                                                  # (T, D, H, W)

        v = normalise(v)                                 # still (T, D, H, W)

        # ---------------- chunked VAE inference ----------- #
        for t0 in range(0, v.shape[0], batch_size):
            slice = v[t0 : t0 + batch_size].unsqueeze(1)  # (b, 1, D, H, W)

            # resize to input spatial size
            slice = torch.nn.functional.interpolate(
                slice, size=input_spatial_size, mode="trilinear", align_corners=False,
            )  # (b, 1, D, H, W)

            feat = vae.encode(slice)[0]  # mean, 
            feat = feat.flatten(start_dim=1)  # (b, latent_dim)
            emb.append(feat)

    return torch.cat(emb, 0)

# ----------------------------------------------------------------------------- #
#                                   Main                                        #
# ----------------------------------------------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_path1", type=Path, required=True)
    p.add_argument("--data_path2", type=Path, required=True)
    p.add_argument("--channel_name", type=str, default=None)
    p.add_argument("--channel_name1", type=str, default=None)
    p.add_argument("--channel_name2", type=str, default=None)
    p.add_argument("--input_spatial_size", type=str, default="32,512,512",
                   help="Input spatial size for the VAE, e.g. '32,512,512'.")
    p.add_argument("--loadcheck_path", type=Path, default=None,
                   help="Path to the VAE model checkpoint for loading.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_fov", type=int, default=None,
                   help="Limit number of FOV pairs (for quick tests).")
    return p

def main(args) -> None:
    device = args.device

    # ----------------- VAE ----------------- #
    vae = torch.jit.load(args.loadcheck_path).to(device)
    vae.eval()

    # ----------------- FOV list  ------------ #
    fovs1, fovs2 = read_zarr(args.data_path1), read_zarr(args.data_path2)
    if args.max_fov:
        fovs1 = fovs1[:args.max_fov]
        fovs2 = fovs2[:args.max_fov]

    # ----------------- Embeddings ----------- #
    input_spatial_size = [int(dim) for dim in args.input_spatial_size.split(",")]

    if args.channel_name is not None:
        args.channel_name1 = args.channel_name2 = args.channel_name
    
    emb1 = encode_fovs(
        fovs1, vae,
        args.channel_name1, 
        device, args.batch_size,
        input_spatial_size,
    )

    emb2 = encode_fovs(
        fovs2, vae,
        args.channel_name2, 
        device, args.batch_size,
        input_spatial_size,
    )

    # ----------------- FID ------------------ #
    fid_val = fid_from_features(emb1, emb2)
    print(f"\nFID: {fid_val:.6f}")

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)