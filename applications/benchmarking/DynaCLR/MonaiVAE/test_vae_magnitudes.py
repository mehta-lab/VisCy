#%%
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    NormalizeIntensity,
)
from torch.nn import KLDivLoss, MSELoss
from torchview import draw_graph

from viscy.representation.vae import BetaVae25D, BetaVaeMonai


def compute_vae_losses(model_output, target, beta=1.0):
    """Compute VAE losses: reconstruction (MSE) and KL divergence.
    """
    mse_loss_fn = MSELoss(reduction='mean')
    recon_loss = mse_loss_fn(model_output.recon_x, target)
    
    # Standard VAE: per-sample, per-dimension KL loss normalization
    batch_size = target.size(0)
    latent_dim = model_output.mean.size(1)  # Get latent dimension
    normalizer = batch_size * latent_dim  # Normalize by both batch size and latent dim
    
    kl_loss = -0.5 * torch.sum(1 + model_output.logvar - model_output.mean.pow(2) - model_output.logvar.exp())
    print(f"  Debug - KL raw: {kl_loss.item():.6f}, normalizer: {normalizer}, batch_size: {target.size(0)}")
    kl_loss = kl_loss / normalizer
    
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'mu': model_output.mean, 
        'logvar': model_output.logvar,  
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'total_loss': total_loss.item(),
        'beta': beta,
        # 'recon_magnitude': torch.abs(model_output.recon_x).mean().item(),
        # 'target_magnitude': torch.abs(target).mean().item(),
        # 'latent_mean_magnitude': torch.abs(model_output.mean).mean().item(),
        # 'latent_std_magnitude': torch.exp(0.5 * model_output.logvar).mean().item(),
    }


def create_synthetic_data(batch_size=2, channels=2, depth=16, height=256, width=256):
    """Create synthetic microscopy-like data with known statistics.
    These are from one FOV of the Phase3D
    - mean: 8.196415001293644e-05 ≈ 0.0001
    - std: 0.09095408767461777 ≈ 0.091
    """
    torch.manual_seed(42) 
    synthetic_data = torch.randn(batch_size, channels, depth, height, width) * 0.091 + 0.0001
    
    for b in range(batch_size):
        for c in range(channels):
            for d in range(depth):
                # Add some blob-like structures
                y_center, x_center = np.random.randint(50, height-50), np.random.randint(50, width-50)
                y, x = np.ogrid[:height, :width]
                mask = (y - y_center)**2 + (x - x_center)**2 < np.random.randint(400, 1600)
                synthetic_data[b, c, d][mask] += np.random.normal(0.05, 0.02)
    
    synthetic_data = torch.clamp(synthetic_data, min=0)
    
    return synthetic_data


def create_known_target(input_data, noise_level=0.1):
    """Create a target with known relationship to input for testing MSE magnitude.
    """
    target = input_data.clone()
    
    noise = torch.randn_like(target) * noise_level * target.std()
    target = target + noise
    
    target = target * 0.95 + 0.01
    
    return torch.clamp(target, min=0)


def test_vae_magnitudes():
    """Test VAE models with both real dataloader and synthetic data."""
    print("=== VAE Magnitude Testing ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_configs = [
    #     {
    #         'name': 'BetaVae25D_ResNet50',
    #         'model_class': BetaVae25D,
    #         'kwargs': {
    #             'backbone': 'resnet50',
    #             'in_channels': 2,
    #             'in_stack_depth': 16,
    #             'latent_dim': 1024,
    #             'input_spatial_size': (256, 256),
    #         }
    #     },
        # Uncomment to test MONAI version
        {
            'name': 'BetaVaeMonai',
            'model_class': BetaVaeMonai,
            'kwargs': {
                'spatial_dims': 3,
                'in_shape': (2, 16, 256, 256),  # (C, D, H, W)
                'out_channels': 2,
                'latent_size': 1024,
                'channels': (32, 64, 128, 256),
                'strides': (2, 2, 2, 2),
            }
        }
    ]
    
    # Test different beta values
    beta_values = [0.1, 1.0, 4.0, 10.0]
    
    for model_config in model_configs:
        print(f"\n{'='*50}")
        print(f"Testing {model_config['name']}")
        print(f"{'='*50}")
        
        # Initialize model
        model = model_config['model_class'](**model_config['kwargs'])
        model = model.to(device)
        model.eval()
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Draw model graph
        print(f"\n--- Model Architecture ---")
        sample_input = create_synthetic_data(batch_size=1).to(device)
        try:
            model_graph = draw_graph(
                model, 
                input_data=sample_input, 
                expand_nested=True,
                depth=6,
                save_graph=True,
                filename=f'{model_config["name"]}_graph',
                directory='./model_graphs/'
            )
            print(f"Model graph saved to: ./model_graphs/{model_config['name']}_graph.png")
        except Exception as e:
            print(f"Could not generate model graph: {e}")
        
        # Test 1: Synthetic data with known target
        print(f"\n--- Test 1: Synthetic Data ---")
        synthetic_input = create_synthetic_data().to(device)
        synthetic_target = create_known_target(synthetic_input).to(device)
        
        print(f"Input shape: {synthetic_input.shape}")
        print(f"Input stats - mean: {synthetic_input.mean():.6f}, std: {synthetic_input.std():.6f}")
        print(f"Target stats - mean: {synthetic_target.mean():.6f}, std: {synthetic_target.std():.6f}")
        
        with torch.no_grad():
            synthetic_output = model(synthetic_input)
            
        print(f"Output shape: {synthetic_output.recon_x.shape}")
        print(f"Latent shape: {synthetic_output.z.shape}")
        
        for beta in beta_values:
            losses = compute_vae_losses(model_output=synthetic_output, target=synthetic_target, beta=beta)
            print(f"\nBeta = {beta}:")
            print(f"  Mu shape: {losses['mu'].shape}, mean: {losses['mu'].mean():.6f}, std: {losses['mu'].std():.6f}")
            print(f"  Logvar shape: {losses['logvar'].shape}, mean: {losses['logvar'].mean():.6f}, std: {losses['logvar'].std():.6f}")
            print(f"  Reconstruction Loss: {losses['recon_loss']:.6f}")
            print(f"  KL Loss: {losses['kl_loss']:.6f}")
            print(f"  Total Loss: {losses['total_loss']:.6f}")
            # print(f"  Recon magnitude: {losses['recon_magnitude']:.6f}")
            # print(f"  Target magnitude: {losses['target_magnitude']:.6f}")
            # print(f"  Latent mean magnitude: {losses['latent_mean_magnitude']:.6f}")
            # print(f"  Latent std magnitude: {losses['latent_std_magnitude']:.6f}")

        #TODO: use the dataloader to run it with real data
        # data_path = "/hpc/projects/organelle_phenotyping/datasets/organelle/SEC61B/2024_10_16_A549_SEC61_ZIKV_DENV"
        # zarr_path = Path(data_path) / "2024_10_16_A549_SEC61_ZIKV_DENV_2.zarr"
        zarr_path = None
        if not zarr_path:
            print(f"Found real data at: {zarr_path}")
            
            normalizations = [
                NormalizeIntensity()
            ]
            
            print("Testing with real data format...")
            
            real_like_data = create_synthetic_data(batch_size=1, channels=2, depth=16, height=256, width=256)
            
            normalized_data = (real_like_data - real_like_data.mean()) / real_like_data.std()
            normalized_data = normalized_data.to(device)
            
            print(f"Normalized data stats - mean: {normalized_data.mean():.6f}, std: {normalized_data.std():.6f}")
            
            with torch.no_grad():
                real_output = model(normalized_data)
            
            losses = compute_vae_losses(model_output=real_output, target=normalized_data, beta=1.0)
            print(f"\nPerfect reconstruction test (beta=1.0):")
            print(f"  Reconstruction Loss: {losses['recon_loss']:.6f}")
            print(f"  KL Loss: {losses['kl_loss']:.6f}")
            print(f"  Total Loss: {losses['total_loss']:.6f}")
            
        else:
            raise NotImplementedError("not implemented")


def print_expected_ranges():
    """Print expected ranges for VAE loss components."""
    print("\n" + "="*60)
    print("EXPECTED LOSS MAGNITUDE RANGES")
    print("="*60)
    print("""
For Beta-VAE with normalized input (0-mean, 1-std):

NOTES

1. RECONSTRUCTION LOSS (MSE):
   - Well-trained model: 0.01 - 0.1
   - Untrained/poorly trained: 0.5 - 2.0
   - Perfect reconstruction: < 0.001

2. KL DIVERGENCE LOSS:
    - Posterior collapse (BAD): < 10 (model ignores latent space)
    - Well-regularized: depends on latent dim, but should allow reconstruction
    - Over-regularized (BAD): Forces posterior too close to prior, hurts reconstruction
    - Typical untrained: can be very high as posterior is random

3. BETA PARAMETER EFFECTS:
   - Beta < 1.0: Prioritizes reconstruction (lower MSE, higher KL)
   - Beta = 1.0: Standard VAE balance
   - Beta > 1.0: Prioritizes disentanglement (higher MSE, lower KL)
    """
    )


if __name__ == "__main__":
    print_expected_ranges()
    test_vae_magnitudes()
    print("\n=== Testing Complete ===")
# %%
