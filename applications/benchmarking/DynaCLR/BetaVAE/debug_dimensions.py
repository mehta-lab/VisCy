# %%
import torch
from viscy.representation.vae import VaeEncoder, VaeDecoder


def debug_vae_dimensions():
    """Debug VAE encoder/decoder dimension compatibility."""

    print("=== VAE Dimension Debugging ===\n")

    # Configuration from test_run.py
    z_stack_depth = 32
    input_shape = (1, 1, z_stack_depth, 192, 192)
    latent_dim = 256
    latent_spatial_size = 3

    print(f"Input shape: {input_shape}")
    print(f"Expected latent dim: {latent_dim}")
    print(f"Expected latent spatial size: {latent_spatial_size}")
    print()

    # Create encoder
    encoder = VaeEncoder(
        backbone="resnet50",
        in_channels=1,
        in_stack_depth=z_stack_depth,
        embedding_dim=latent_dim,
        stem_kernel_size=(8, 4, 4),
        stem_stride=(8, 4, 4),
    )

    # Create decoder
    decoder = VaeDecoder(
        decoder_channels=[1024, 512, 256, 128],
        latent_dim=latent_dim,
        out_channels=1,
        out_stack_depth=z_stack_depth,
        latent_spatial_size=latent_spatial_size,
        head_expansion_ratio=1,
        head_pool=False,
        upsample_mode="deconv",
        conv_blocks=2,
        norm_name="batch",
        upsample_pre_conv=None,
        strides=[2, 2, 2, 2],
    )

    print("=== ENCODER FORWARD PASS ===")

    # Test encoder
    x = torch.randn(*input_shape)
    print(f"Input to encoder: {x.shape}")

    try:
        # Step through encoder
        print("\n1. Stem processing:")
        x_stem = encoder.stem(x)
        print(f"   After stem: {x_stem.shape}")

        print("\n2. Backbone processing:")
        features = encoder.encoder(x_stem)
        for i, feat in enumerate(features):
            print(f"   Feature {i}: {feat.shape}")

        print("\n3. Final processing:")
        x_final = features[-1]
        print(f"   Final features: {x_final.shape}")

        x_pooled = encoder.global_pool(x_final)
        print(f"   After global pool: {x_pooled.shape}")

        x_flat = x_pooled.flatten(1)
        print(f"   After flatten: {x_flat.shape}")

        # Full encoder output
        encoder_output = encoder(x)
        mu = encoder_output.embedding
        logvar = encoder_output.log_covariance
        print(f"   Final mu: {mu.shape}")
        print(f"   Final logvar: {logvar.shape}")

        print("\n=== DECODER FORWARD PASS ===")

        # Test decoder with latent vector
        z = torch.randn(1, latent_dim)
        print(f"Input to decoder: {z.shape}")

        print("\n1. Latent projection:")
        x_proj = decoder.latent_proj(z)
        print(f"   After projection: {x_proj.shape}")

        x_reshaped = x_proj.view(1, -1, latent_spatial_size, latent_spatial_size)
        print(f"   After reshape: {x_reshaped.shape}")

        print("\n2. Decoder stages:")
        x_current = x_reshaped
        for i, stage in enumerate(decoder.decoder_stages):
            x_current = stage(x_current)
            print(f"   After stage {i}: {x_current.shape}")

        print("\n3. Head processing:")
        final_output = decoder.head(x_current)
        print(f"   Final output: {final_output.shape}")

        # Full decoder output
        decoder_output = decoder(z)
        reconstruction = decoder_output["reconstruction"]
        print(f"   Full reconstruction: {reconstruction.shape}")

        print("\n=== DIMENSION ANALYSIS ===")
        print(f"✓ Encoder input:  {input_shape}")
        print(f"✓ Encoder output: {mu.shape}")
        print(f"✓ Decoder input:  {z.shape}")
        print(f"✓ Decoder output: {reconstruction.shape}")

        # Calculate tensor sizes
        input_size = torch.numel(x)
        recon_size = torch.numel(reconstruction)
        print(f"  Input tensor size: {input_size}")
        print(f"  Reconstruction tensor size: {recon_size}")
        print(f"  Size ratio: {recon_size / input_size:.2f}")

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

    except Exception as e:
        print(f"✗ ERROR during forward pass: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()

        # Let's check what spatial size the encoder actually produces
        print("\n=== ENCODER SPATIAL SIZE ANALYSIS ===")
        try:
            x_stem = encoder.stem(x)
            features = encoder.encoder(x_stem)
            final_feat = features[-1]
            actual_spatial_size = final_feat.shape[-1]  # Assuming square
            print(f"Actual spatial size from encoder: {actual_spatial_size}")
            print(f"Expected spatial size for decoder: {latent_spatial_size}")

            if actual_spatial_size != latent_spatial_size:
                print(
                    f"✗ MISMATCH: Encoder produces {actual_spatial_size}x{actual_spatial_size}, decoder expects {latent_spatial_size}x{latent_spatial_size}"
                )
                print(f"  Suggested fix: Set latent_spatial_size={actual_spatial_size}")

        except Exception as inner_e:
            print(f"Error in spatial size analysis: {inner_e}")


if __name__ == "__main__":
    debug_vae_dimensions()
# %%
