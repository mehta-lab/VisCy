from pathlib import Path

import click
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
    """Normalize volume to [-1, 1] range using min-max normalization.
    
    Parameters
    ----------
    volume : torch.Tensor
        Input volume with shape (D, H, W) or (B, D, H, W)
        
    Returns
    -------
    torch.Tensor
        Normalized volume in [-1, 1] range with same shape as input
    """
    v_min = volume.amin(dim=(-3, -2, -1), keepdim=True)
    v_max = volume.amax(dim=(-3, -2, -1), keepdim=True)
    volume = (volume - v_min) / (v_max - v_min + 1e-6)        # → [0,1]
    return volume * 2.0 - 1.0                                 # → [-1,1]

@torch.jit.script_if_tracing
def sqrtm(sigma: Tensor) -> Tensor:
    r"""Compute the square root of a positive semi-definite matrix.

    Uses eigendecomposition: :math:`\sqrt{\Sigma} = Q \sqrt{\Lambda} Q^T`
    where :math:`Q \Lambda Q^T` is the eigendecomposition of :math:`\Sigma`.

    Parameters
    ----------
    sigma : Tensor
        A positive semi-definite matrix with shape (*, D, D)
        
    Returns
    -------
    Tensor
        Square root of the input matrix with same shape
        
    Examples
    --------
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
    r"""Compute the Fréchet distance between two multivariate Gaussian distributions.

    The Fréchet distance is given by:
    .. math:: d^2 = \left\| \mu_x - \mu_y \right\|_2^2 +
        \operatorname{tr} \left( \Sigma_x + \Sigma_y - 2 \sqrt{\Sigma_y^{\frac{1}{2}} \Sigma_x \Sigma_y^{\frac{1}{2}}} \right)

    Parameters
    ----------
    mu_x : Tensor
        Mean of the first distribution with shape (*, D)
    sigma_x : Tensor
        Covariance of the first distribution with shape (*, D, D)
    mu_y : Tensor
        Mean of the second distribution with shape (*, D)
    sigma_y : Tensor
        Covariance of the second distribution with shape (*, D, D)
        
    Returns
    -------
    Tensor
        Fréchet distance between the two distributions
        
    References
    ----------
    .. [1] https://wikipedia.org/wiki/Frechet_distance
        
    Examples
    --------
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
    """Compute Fréchet Inception Distance (FID) from feature embeddings.
    
    Parameters
    ----------
    f1 : torch.Tensor
        Features from first dataset with shape (N1, D)
    f2 : torch.Tensor
        Features from second dataset with shape (N2, D)
    eps : float, default=1e-6
        Small value added to diagonal for numerical stability
        
    Returns
    -------
    float
        FID score between the two feature sets
    """
    mu1, sigma1 = f1.mean(0), torch.cov(f1.T)
    mu2, sigma2 = f2.mean(0), torch.cov(f2.T)

    eye = torch.eye(sigma1.size(0), device=sigma1.device, dtype=sigma1.dtype)
    sigma1 = sigma1 + eps * eye
    sigma2 = sigma2 + eps * eye

    return frechet_distance(mu1, sigma1, mu2, sigma2).clamp_min_(0).item()

@torch.inference_mode()
def encode_fovs(
    fovs,
    vae,
    channel_name: str,
    device: str = "cuda",
    batch_size: int = 4,
    input_spatial_size: tuple = (32, 512, 512), 
):
    """Encode field-of-view (FOV) data using a variational autoencoder.
    
    For each FOV:
    - Extract all time-frames with shape (T, D, H, W)
    - Normalize to [-1, 1] range
    - Process through VAE in batches of ≤ batch_size frames
    - Collect all latent vectors from all time points
    
    Parameters
    ----------
    fovs : list
        List of FOV position objects
    vae : torch.nn.Module
        Pre-trained VAE model for encoding
    channel_name : str
        Name of the channel to extract from each FOV
    device : str, default="cuda"
        Device to run computations on
    batch_size : int, default=4
        Number of frames to process simultaneously
    input_spatial_size : tuple, default=(32, 512, 512)
        Target spatial dimensions for VAE input (D, H, W)
        
    Returns
    -------
    torch.Tensor
        Concatenated embeddings from all FOVs and timepoints with shape (N_total_timepoints, latent_dim)
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

@click.command()
@click.option("--source_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--target_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--channel_names", type=str, multiple=True, required=True,
              help="Channel names for source and target (1 or 2 values). If 1 value, same channel used for both.")
@click.option("--input_spatial_size", type=str, default="32,512,512",
              help="Input spatial size for the VAE, e.g. '32,512,512'.")
@click.option("--loadcheck_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to the VAE model checkpoint for loading.")
@click.option("--batch_size", type=int, default=4)
@click.option("--device", type=str, default="cuda")
@click.option("--max_fov", type=int, default=None,
              help="Limit number of FOV pairs (for quick tests).")
def main(source_path, target_path, channel_names, 
         input_spatial_size, loadcheck_path, batch_size, device, max_fov) -> None:

    # ----------------- VAE ----------------- #
    vae = torch.jit.load(loadcheck_path).to(device)
    vae.eval()

    # ----------------- FOV list  ------------ #
    fovs1, fovs2 = read_zarr(source_path), read_zarr(target_path)
    if max_fov:
        fovs1 = fovs1[:max_fov]
        fovs2 = fovs2[:max_fov]

    # ----------------- Embeddings ----------- #
    input_spatial_size = [int(dim) for dim in input_spatial_size.split(",")]

    # Handle channel names: use same for both if only one provided
    if len(channel_names) == 1:
        channel_name1 = channel_name2 = channel_names[0]
    elif len(channel_names) == 2:
        channel_name1, channel_name2 = channel_names
    else:
        raise ValueError("Must provide 1 or 2 channel names")
    
    emb1 = encode_fovs(
        fovs1, vae,
        channel_name1, 
        device, batch_size,
        input_spatial_size,
    )

    emb2 = encode_fovs(
        fovs2, vae,
        channel_name2, 
        device, batch_size,
        input_spatial_size,
    )

    # ----------------- FID ------------------ #
    fid_val = fid_from_features(emb1, emb2)
    print(f"\nFID: {fid_val:.6f}")

if __name__ == "__main__":
    main()