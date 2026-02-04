# %%
"""
Test script to visualize SAM2 input images and feature processing.
This script helps debug what images are being passed to SAM2 and how they're processed.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
from sam2_embeddings import SAM2Module, load_config, load_normalization_from_config

from viscy.data.triplet import TripletDataModule


def visualize_rgb_conversion(x_original, x_rgb_list, save_dir="./debug_images"):
    """Visualize the RGB conversion process"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"Original input shape: {x_original.shape}")
    print(f"Original input range: [{x_original.min():.3f}, {x_original.max():.3f}]")

    # Plot original channels
    B, C = x_original.shape[:2]
    fig, axes = plt.subplots(3, max(3, C), figsize=(15, 12))

    # Plot original channels
    for c in range(C):
        ax = axes[0, c] if C > 1 else axes[0, 0]
        img = x_original[0, c].cpu().numpy()
        im = ax.imshow(img, cmap="gray")
        ax.set_title(f"Original Channel {c}")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

    # Plot RGB conversion
    rgb_img = x_rgb_list[0]  # First batch item
    print(f"RGB image shape: {rgb_img.shape}")
    print(f"RGB image range: [{rgb_img.min():.3f}, {rgb_img.max():.3f}]")

    for c in range(3):
        ax = axes[1, c]
        im = ax.imshow(rgb_img[:, :, c], cmap="gray")
        ax.set_title(f"RGB Channel {c}")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

    # Plot merged RGB image
    ax = axes[2, 0]
    # Normalize to 0-1 for display
    rgb_display = rgb_img.copy()
    rgb_display = (rgb_display - rgb_display.min()) / (
        rgb_display.max() - rgb_display.min()
    )
    ax.imshow(rgb_display)
    ax.set_title("Merged RGB Image")
    ax.axis("off")

    # Check if RGB is properly scaled to 0-255
    ax = axes[2, 1]
    ax.text(
        0.1,
        0.8,
        f"RGB Range: [{rgb_img.min():.1f}, {rgb_img.max():.1f}]",
        transform=ax.transAxes,
    )
    ax.text(0.1, 0.6, "Expected: [0, 255]", transform=ax.transAxes)
    ax.text(
        0.1,
        0.4,
        f"Properly scaled: {rgb_img.min() >= 0 and rgb_img.max() <= 255}",
        transform=ax.transAxes,
    )
    ax.text(0.1, 0.2, f"Mean: {rgb_img.mean():.1f}", transform=ax.transAxes)
    ax.set_title("RGB Scaling Check")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/rgb_conversion.png", dpi=150, bbox_inches="tight")
    plt.close()


def test_sam2_processing(config_path, num_samples=3):
    """Test SAM2 processing with visualization"""

    # Load configuration
    cfg = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # Setup data module (same as in main function)
    dm_params = {}
    dm_params["data_path"] = cfg["paths"]["data_path"]
    dm_params["tracks_path"] = cfg["paths"]["tracks_path"]

    # Setup normalizations
    norm_configs = cfg["datamodule"]["normalizations"]
    normalizations = [load_normalization_from_config(norm) for norm in norm_configs]
    dm_params["normalizations"] = normalizations

    # Copy other datamodule parameters
    for param, value in cfg["datamodule"].items():
        if param != "normalizations":
            if param == "patch_size":
                dm_params["initial_yx_patch_size"] = value
                dm_params["final_yx_patch_size"] = value
            else:
                dm_params[param] = value

    print("Setting up data module...")
    dm = TripletDataModule(**dm_params)
    dm.setup(stage="predict")

    # Get model parameters
    channel_reduction_methods = {}
    if "model" in cfg and "channel_reduction_methods" in cfg["model"]:
        channel_reduction_methods = cfg["model"]["channel_reduction_methods"]

    # Initialize SAM2 model
    print("Loading SAM2 model...")
    model = SAM2Module(
        model_name=cfg["model"]["model_name"],
        channel_reduction_methods=channel_reduction_methods,
    )

    # Get dataloader
    predict_dataloader = dm.predict_dataloader()

    print(f"Testing with {num_samples} samples...")

    # Test processing
    for i, batch in enumerate(predict_dataloader):
        if i >= num_samples:
            break

        print(f"\n--- Sample {i + 1} ---")
        x = batch["anchor"]
        print(f"Input tensor shape: {x.shape}")
        print(f"Input tensor range: [{x.min():.3f}, {x.max():.3f}]")

        # Test 5D reduction if needed
        if x.dim() == 5:
            print("Applying 5D reduction...")
            x_reduced = model._reduce_5d_input(x)
            print(f"After 5D reduction: {x_reduced.shape}")
            print(f"Reduction methods: {model.channel_reduction_methods}")
        else:
            x_reduced = x

        # Test RGB conversion
        print("Converting to RGB...")
        x_rgb_list = model._convert_to_rgb(x_reduced)
        print(f"RGB conversion result: {len(x_rgb_list)} images")
        print(f"First RGB image shape: {x_rgb_list[0].shape}")

        # Visualize the conversion
        visualize_rgb_conversion(x_reduced, x_rgb_list, f"./debug_images/sample_{i}")

        # Test feature extraction (if model is available)
        try:
            print("Testing feature extraction...")
            model.model = model.model or model.on_predict_start()
            model.model.set_image_batch(x_rgb_list)

            # Check what features are available
            features_dict = model.model._features
            print(f"Available features: {list(features_dict.keys())}")

            if "high_res_feats" in features_dict:
                high_res_feats = features_dict["high_res_feats"]
                print(f"High-res features length: {len(high_res_feats)}")
                for j, feat in enumerate(high_res_feats):
                    print(f"  Layer {j}: {feat.shape}")

            if "image_embed" in features_dict:
                image_embed = features_dict["image_embed"]
                print(f"Image embed shape: {image_embed.shape}")

            # Extract final features (current approach)
            features = model.model._features["high_res_feats"][1].mean(dim=(2, 3))
            print(f"Final features shape: {features.shape}")
            print(f"Final features range: [{features.min():.3f}, {features.max():.3f}]")

        except Exception as e:
            print(f"Feature extraction failed: {e}")

        print("-" * 50)


def main():
    """Main function to run the test"""
    config_path = "/home/eduardo.hirata/repos/viscy/applications/benchmarking/DynaCLR/SAM2/sam2_sensor_only.yml"

    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        print("Please provide a valid config file path")
        return

    try:
        test_sam2_processing(config_path, num_samples=3)
        print("\nTest completed successfully!")
        print("Check ./debug_images/ for visualization outputs")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()


# %%
if __name__ == "__main__":
    main()

# %%
