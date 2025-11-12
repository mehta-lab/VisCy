import warnings
from pathlib import Path

import click
import numpy as np
import torch
import xarray as xr
from iohub.ngff import Position, open_ome_zarr
from torch import Tensor
from tqdm import tqdm

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
def embed_position(
    position: Position,
    vae: torch.nn.Module,
    channel_name: str,
    device: str = "cuda",
    batch_size: int = 4,
    input_spatial_size: tuple = (32, 512, 512), 
):
    """Encode position data using a variational autoencoder with metadata.
    
    Parameters
    ----------
    position : Position
        Single position object from zarr plate
    vae : torch.nn.Module
        Pre-trained VAE model for encoding
    channel_name : str
        Name of the channel to extract from the position
    device : str, default="cuda"
        Device to run computations on
    batch_size : int, default=4
        Number of frames to process simultaneously
    input_spatial_size : tuple, default=(32, 512, 512)
        Target spatial dimensions for VAE input (D, H, W)
        
    Returns
    -------
    xr.Dataset
        Dataset with embeddings and metadata
    """
    position_name = position.zgroup.name
    embeddings_list = []
    timepoints_list = []

    v = torch.as_tensor(
        position.data[:, position.get_channel_index(channel_name)],
        dtype=torch.float32, device=device,
    )                                                  # (T, D, H, W)

    v = normalise(v)                                 # still (T, D, H, W)

    timepoint = 0
    for t0 in tqdm(range(0, v.shape[0], batch_size), desc=f"Encoding {position_name}/{channel_name}"):
        batch_slice = v[t0 : t0 + batch_size].unsqueeze(1)
        batch_slice = torch.nn.functional.interpolate(
            batch_slice, size=input_spatial_size, mode="trilinear", align_corners=False,
        )

        feat = vae.encode(batch_slice)[0]  # mean, 
        feat = feat.flatten(start_dim=1)  # (b, latent_dim)
        
        feat_np = feat.cpu().numpy()
        for i, embedding in enumerate(feat_np):
            embeddings_list.append(embedding)
            timepoints_list.append(timepoint + i)
        timepoint += feat.shape[0]

    embeddings_array = np.stack(embeddings_list)
    n_samples, n_features = embeddings_array.shape
    
    ds = xr.Dataset({
        'embeddings': (['sample', 'feature'], embeddings_array)
    }, coords={
        'sample': range(n_samples),
        'feature': range(n_features),
        't': ('sample', timepoints_list)
    })
    
    ds.attrs['position_name'] = position_name
    ds.attrs['channel_name'] = channel_name
    
    return ds

@click.command()
@click.option("--source_position", "-s", type=click.Path(exists=True, path_type=Path), required=True, help="Full path to source position (e.g., '/path/to/plate.zarr/A/1/0')")
@click.option("--target_position", "-t", type=click.Path(exists=True, path_type=Path), required=True, help="Full path to target position (e.g., '/path/to/plate.zarr/B/2/0')")
@click.option("--source_channel", "-sc", type=str, required=True, help="Channel name for source position")
@click.option("--target_channel", "-tc", type=str, required=True, help="Channel name for target position")
@click.option("-z", type=int, default=32, help="Depth dimension for VAE input")
@click.option("-y", type=int, default=512, help="Height dimension for VAE input") 
@click.option("-x", type=int, default=512, help="Width dimension for VAE input")
@click.option("--ckpt_path", "-c", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to the VAE model checkpoint for loading.")
@click.option("--batch_size", "-b", type=int, default=4)
@click.option("--device", "-d", type=str, default="cuda")
@click.option("--output_dir", "-o", type=click.Path(path_type=Path), help="Path to save source embeddings")
def embed_dataset(source_position, target_position, source_channel, target_channel, z, y, x,
         ckpt_path, batch_size, device, output_dir) -> None:
    """Encode positions using a pre-trained VAE and optionally compute FID or save embeddings.
    
    This function loads two zarr positions, encodes them using a variational autoencoder,
    and can either compute FID scores or save embeddings with metadata to a parquet file.
    
    Parameters
    ----------
    source_position : Path
        Full path to the source position (e.g., '/path/to/plate.zarr/A/1/0')
    target_position : Path  
        Full path to the target position (e.g., '/path/to/plate.zarr/B/2/0')
    source_channel : str
        Channel name for source position
    target_channel : str
        Channel name for target position
    z : int
        Depth dimension for VAE input
    y : int
        Height dimension for VAE input
    x : int
        Width dimension for VAE input
    ckpt_path : Path
        Path to the pre-trained VAE model checkpoint (.pt file)
    batch_size : int
        Number of timepoints to process simultaneously through the VAE
    device : str
        Device to run computations on ("cuda" or "cpu")
    output_dir : Path
        Path to save embeddings
    """

    # ----------------- VAE ----------------- #
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA is not available, using CPU instead")
        device = "cpu"

    vae = torch.jit.load(ckpt_path).to(device)
    vae.eval()

    source_position = open_ome_zarr(source_position)
    source_channel_names = source_position.channel_names
    assert source_channel in source_channel_names, f"Channel {source_channel} not found in source position"

    target_position = open_ome_zarr(target_position)
    target_channel_names = target_position.channel_names
    assert target_channel in target_channel_names, f"Channel {target_channel} not found in target position"

    input_spatial_size = (z, y, x)

    if output_dir:
        output_dir = Path(output_dir)
        source_name = source_position.zgroup.name.split('/')[-1] if source_position.zgroup.name else "source"
        target_name = target_position.zgroup.name.split('/')[-1] if target_position.zgroup.name else "target"
        source_output = output_dir / f"{source_name}_{source_channel}.zarr"
        target_output = output_dir / f"{target_name}_{target_channel}.zarr"

        source_ds = embed_position(
            position=source_position, vae=vae,
            channel_name=source_channel,
            device=device, batch_size=batch_size,
            input_spatial_size=input_spatial_size,
        )
        source_ds.to_zarr(source_output, mode='w')
        print(f"Source embeddings saved to: {source_output}")
        
        target_ds = embed_position(
            position=target_position, vae=vae,
            channel_name=target_channel,
            device=device, batch_size=batch_size,
            input_spatial_size=input_spatial_size,
        )
        target_ds.to_zarr(target_output, mode='w')
        print(f"Target embeddings saved to: {target_output}")

@click.command()
@click.option("--source_path", "-s", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the source embeddings zarr file")
@click.option("--target_path", "-t", type=click.Path(exists=True, path_type=Path), required=True, help="Path to the target embeddings zarr file")
def compute_fid_cli(source_path: Path, target_path: Path) -> None:
    """Compute FID score between two embedding datasets.
    
    Parameters
    ----------
    source_path : Path
        Path to the source embeddings zarr file
    target_path : Path
        Path to the target embeddings zarr file
        
    Examples
    --------
    $ python fid_ts.py compute-fid \\
        -s source_embeddings.zarr \\
        -t target_embeddings.zarr
    """
    # Load the datasets
    source_ds = xr.open_zarr(source_path)
    target_ds = xr.open_zarr(target_path)
    
    # Get embeddings arrays
    source_embeddings = torch.tensor(source_ds.embeddings.values, dtype=torch.float32)
    target_embeddings = torch.tensor(target_ds.embeddings.values, dtype=torch.float32)
    
    fid_score = fid_from_features(source_embeddings, target_embeddings)
    
    # Get metadata from attributes
    source_channel = source_ds.attrs.get('channel_name', 'unknown')
    target_channel = target_ds.attrs.get('channel_name', 'unknown')
    source_position = source_ds.attrs.get('position_name', 'unknown')
    target_position = target_ds.attrs.get('position_name', 'unknown')
    
    print(f"Source: {source_position}/{source_channel} ({len(source_embeddings)} samples)")
    print(f"Target: {target_position}/{target_channel} ({len(target_embeddings)} samples)")
    print(f"FID score: {fid_score:.6f}")

    return fid_score

@click.group()
def cli():
    """VAE embedding and FID computation tools."""
    pass

cli.add_command(embed_dataset, name="embed")
cli.add_command(compute_fid_cli, name="compute-fid")

if __name__ == "__main__":
    cli()