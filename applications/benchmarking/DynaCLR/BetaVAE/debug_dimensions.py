# %%
import torch
from viscy.representation.vae import VaeEncoder, VaeDecoder


def debug_vae_dimensions():
    """Debug VAE encoder/decoder dimension compatibility."""

    print("=== VAE Dimension Debugging (Updated Architecture) ===\n")

    # Configuration matching current config
    z_stack_depth = 16
    input_shape = (1, 1, z_stack_depth, 192, 192)  # 1 channel to match config
    latent_dim = 1024  # Updated to new default

    print(f"Input shape: {input_shape}")
    print(f"Expected latent dim: {latent_dim}")
    print()

    # Create encoder
    encoder = VaeEncoder(
        backbone="resnet50",
        in_channels=1,
        in_stack_depth=z_stack_depth,
        latent_dim=latent_dim,
        stem_kernel_size=(4, 2, 2),
        stem_stride=(4, 2, 2),
    )

    # Create decoder
    decoder = VaeDecoder(
        decoder_channels=[1024, 512, 256, 128],
        latent_dim=latent_dim,
        out_channels=1,
        out_stack_depth=z_stack_depth,
        head_expansion_ratio=2,
        head_pool=False,
        upsample_mode="pixelshuffle",
        conv_blocks=2,
        norm_name="batch",
        strides=[2, 2, 2, 1],
    )

    print("=== ENCODER FORWARD PASS ===")

    # Test encoder
    x = torch.randn(*input_shape)
    print(f"Input to encoder: {x.shape}")

    try:
        # Step through encoder
        print("\\n1. Stem processing:")
        x_stem = encoder.stem(x)
        print(f"   After stem: {x_stem.shape}")

        print("\\n2. Backbone processing:")
        features = encoder.encoder(x_stem)
        for i, feat in enumerate(features):
            print(f"   Feature {i}: {feat.shape}")

        print("\\n3. Final processing:")
        x_final = features[-1]
        print(f"   Final features: {x_final.shape}")

        # Flatten spatial dimensions (new approach)
        batch_size = x_final.size(0)
        x_flat = x_final.view(batch_size, -1)
        print(f"   After flatten: {x_flat.shape}")

        # Full encoder output
        encoder_output = encoder(x)
        mu = encoder_output.embedding
        logvar = encoder_output.log_covariance
        print(f"   Final mu: {mu.shape}")
        print(f"   Final logvar: {logvar.shape}")

        print("\\n=== DECODER FORWARD PASS ===")

        # Test decoder with latent vector
        z = torch.randn(1, latent_dim)
        print(f"Input to decoder: {z.shape}")

        print("\\n1. Reshape to spatial:")
        batch_size = z.size(0)
        z_spatial = decoder.latent_reshape(z)
        print(f"   After linear reshape: {z_spatial.shape}")

        z_spatial_reshaped = z_spatial.view(
            batch_size,
            decoder.spatial_channels,
            decoder.spatial_size,
            decoder.spatial_size,
        )
        print(f"   After view to spatial: {z_spatial_reshaped.shape}")

        print("\\n2. Latent projection:")
        x_proj = decoder.latent_proj(z_spatial_reshaped)
        print(f"   After conv projection: {x_proj.shape}")

        print("\\n3. Decoder stages:")
        x_current = x_proj
        for i, stage in enumerate(decoder.decoder_stages):
            x_current = stage(x_current)
            print(f"   After stage {i}: {x_current.shape}")

        print("\\n4. Head processing:")
        final_output = decoder.head(x_current)
        print(f"   Final output: {final_output.shape}")

        # Full decoder output (now returns tensor directly, not dict)
        reconstruction = decoder(z)
        print(f"   Full reconstruction: {reconstruction.shape}")

        print("\\n=== DIMENSION ANALYSIS ===")
        print(f"✓ Encoder input:  {input_shape}")
        print(f"✓ Encoder output: {mu.shape}")
        print(f"✓ Decoder input:  {z.shape}")
        print(f"✓ Decoder output: {reconstruction.shape}")

        # Calculate tensor sizes and compression ratio
        input_size = torch.numel(x)
        latent_size = torch.numel(mu)
        recon_size = torch.numel(reconstruction)

        print(f"  Input tensor size: {input_size:,}")
        print(f"  Latent tensor size: {latent_size:,}")
        print(f"  Reconstruction tensor size: {recon_size:,}")
        print(f"  Compression ratio: {input_size / latent_size:.1f}:1")
        print(f"  Size ratio (recon/input): {recon_size / input_size:.2f}")

        # Check if reconstruction matches input
        if reconstruction.shape == x.shape:
            print("✓ SUCCESS: Reconstruction shape matches input shape!")
        else:
            print(f"✗ ERROR: Shape mismatch!")
            print(f"  Input:         {x.shape}")
            print(f"  Reconstruction: {reconstruction.shape}")

            # Analyze each dimension
            for i, (inp_dim, recon_dim) in enumerate(
                zip(x.shape, reconstruction.shape)
            ):
                if inp_dim != recon_dim:
                    print(
                        f"  Dimension {i}: {inp_dim} → {recon_dim} (factor: {recon_dim/inp_dim:.2f})"
                    )

        print("\\n=== VAE LOSS COMPUTATION TEST ===")

        # Simulate full VAE forward pass with loss computation
        print("Testing full VAE forward pass with loss computation...")

        # Sample from latent distribution (reparameterization trick)
        eps = torch.randn_like(mu)
        z_sampled = mu + torch.exp(0.5 * logvar) * eps
        print(f"Sampled latent z: {z_sampled.shape}")

        # Decode the sampled latent
        reconstruction_from_sampled = decoder(z_sampled)
        print(f"Reconstruction from sampled z: {reconstruction_from_sampled.shape}")

        # Compute VAE losses
        import torch.nn.functional as F

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction_from_sampled, x, reduction="mean")
        print(f"Reconstruction loss (MSE): {recon_loss.item():.6e}")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        print(f"KL divergence loss: {kl_loss.item():.6e}")

        # Total VAE loss with different beta values
        betas = [0.1, 1.0, 1.5, 4.0]
        for beta in betas:
            total_loss = recon_loss + beta * kl_loss
            print(f"Total loss (β={beta}): {total_loss.item():.6e}")

        # Check for problematic values
        print("\\n=== LOSS HEALTH CHECK ===")

        if torch.isnan(recon_loss):
            print("✗ CRITICAL: Reconstruction loss is NaN!")
        elif torch.isinf(recon_loss):
            print("✗ CRITICAL: Reconstruction loss is Inf!")
        elif recon_loss.item() > 1e6:
            print(f"⚠ WARNING: Very high reconstruction loss: {recon_loss.item():.2e}")
        elif recon_loss.item() < 1e-10:
            print(f"⚠ WARNING: Very low reconstruction loss: {recon_loss.item():.2e}")
        else:
            print(f"✓ Reconstruction loss looks reasonable: {recon_loss.item():.6f}")

        if torch.isnan(kl_loss):
            print("✗ CRITICAL: KL loss is NaN!")
        elif torch.isinf(kl_loss):
            print("✗ CRITICAL: KL loss is Inf!")
        else:
            print(f"✓ KL loss looks reasonable: {kl_loss.item():.6f}")

        # Check reconstruction value ranges
        recon_min, recon_max = (
            reconstruction_from_sampled.min(),
            reconstruction_from_sampled.max(),
        )
        input_min, input_max = x.min(), x.max()

        print(f"\\nValue ranges:")
        print(f"  Input range: [{input_min.item():.3f}, {input_max.item():.3f}]")
        print(
            f"  Reconstruction range: [{recon_min.item():.3f}, {recon_max.item():.3f}]"
        )

        if recon_max.item() > 100 or recon_min.item() < -100:
            print(
                "⚠ WARNING: Reconstruction values are very large - possible gradient explosion"
            )

        # Check latent statistics
        mu_mean, mu_std = mu.mean(), mu.std()
        logvar_mean, logvar_std = logvar.mean(), logvar.std()

        print(f"\\nLatent statistics:")
        print(f"  μ mean/std: {mu_mean.item():.3f} / {mu_std.item():.3f}")
        print(f"  log(σ²) mean/std: {logvar_mean.item():.3f} / {logvar_std.item():.3f}")

        if mu_std.item() > 10:
            print("⚠ WARNING: μ has very high variance - possible gradient explosion")
        if logvar_mean.item() > 10:
            print("⚠ WARNING: log(σ²) is very large - possible numerical instability")

    except Exception as e:
        print(f"✗ ERROR during forward pass: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()

        # Check flattened feature size for new architecture
        print("\\n=== ENCODER FLATTENED SIZE ANALYSIS ===")
        try:
            x_stem = encoder.stem(x)
            features = encoder.encoder(x_stem)
            final_feat = features[-1]
            print(f"Final feature shape: {final_feat.shape}")

            flattened_size = final_feat.view(1, -1).shape[1]
            print(f"Flattened size: {flattened_size:,}")
            print(f"Expected latent dim: {latent_dim:,}")

            compression_ratio = flattened_size / latent_dim
            print(f"Compression ratio: {compression_ratio:.1f}:1")

        except Exception as inner_e:
            print(f"Error in flattened size analysis: {inner_e}")


if __name__ == "__main__":
    debug_vae_dimensions()
# %%
