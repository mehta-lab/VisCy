"""
Demo script for using MLFlowCytoland2D

This script demonstrates:
1. Logging a FcmaeUNet model to MLflow
2. Loading a model from MLflow
3. Making predictions using the model
4. Saving a model to disk without MLflow tracking
"""

import datetime
import logging
import os

import mlflow
import numpy as np
from mlflow.models import get_model_info

from viscy.deployment.ml_flow.model_cytoland2D import (
    MLFlowCytoLand,
    log_cytoland_model,
    save_cytoland_model,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("mlflow").setLevel(logging.DEBUG)

# Use a sample checkpoint path - replace with your actual checkpoint path
MODEL_CHECKPOINT_PATH = "/hpc/projects/comp.micro/virtual_staining/datasets/public/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"  # Update this path

# Model configuration for CytoLand
MODEL_CONFIG = {
    "in_channels": 1,
    "out_channels": 2,
    "encoder_blocks": [3, 3, 9, 3],
    "dims": [96, 192, 384, 768],
    "decoder_conv_blocks": 2,
    "stem_kernel_size": [1, 2, 2],
    "in_stack_depth": 1,
    "pretraining": False,
}


def check_model_exists():
    """Check if the model checkpoint file exists"""

    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        raise ValueError(
            f"ERROR: Model checkpoint not found at: {MODEL_CHECKPOINT_PATH}"
        )
    return True


def demo_log_model():
    """Demonstrates logging a model to MLflow"""
    if not check_model_exists():
        return

    print("\n=== Logging CytoLand model to MLflow ===")
    # Set tracking URI if needed
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Output path for predictions - set to None to use temporary directories
    output_path = None  # or specify a path like "output.zarr"

    # TODO: hardcoding to 2D random example for now
    y, x = np.ogrid[0:256, 0:256]  # Simple 2D example
    example_input = np.exp(-(1 / 8) * ((x - 32) ** 2 + (y - 32) ** 2) / 64**2).astype(
        np.float32
    )
    print(
        f"Creating example input with shape: {example_input.shape}, dtype: {example_input.dtype}"
    )

    run_id = log_cytoland_model(
        model_config=MODEL_CONFIG,
        model_path=MODEL_CHECKPOINT_PATH,
        model_name="cytoland2d_test",
        output_path=output_path,
        registered_model_name="CytoLand2D",
        input_example=example_input,  # Include the example input
        skip_signature_inference=True,  # Skip automatic signature inference to avoid Zarr errors
        pip_requirements=["viscy"],
    )

    print(f"Model logged to MLflow with run_id: {run_id}")
    return run_id


def demo_save_model():
    """Demonstrates saving a model to disk without MLflow tracking"""
    if not check_model_exists():
        return

    logger.info("\n=== Saving CytoLand model to disk ===")

    # Create a unique directory name for the saved model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create parent directory if using a nested structure
    parent_dir = "./saved_model"
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    output_directory = f"{parent_dir}/{timestamp}"
    logger.debug(f"Using output directory: {output_directory}")

    # Output path for predictions - set to None to use temporary directories
    # output_path = "/home/eduardo.hirata/repos/viscy/viscy/deployment/ml_flow/tmp/output.zarr"  # or specify a path like "output.zarr"
    output_path = None

    # Create an example input for documentation - IMPORTANT: Keep this as 2D for MLflow signature
    # This will be stored with the model to show expected format
    y, x = np.ogrid[0:256, 0:256]  # Simple 2D example
    example_input = np.exp(-(1 / 8) * ((x - 32) ** 2 + (y - 32) ** 2) / 64**2).astype(
        np.float32
    )
    logger.debug(
        f"Creating example input with shape: {example_input.shape}, dtype: {example_input.dtype}"
    )

    try:
        saved_path = save_cytoland_model(
            model_config=MODEL_CONFIG,
            model_path=MODEL_CHECKPOINT_PATH,
            output_directory=output_directory,
            output_path=output_path,
            input_example=example_input,  # Include the example input
            overwrite=True,  # Overwrite if the directory exists
            skip_signature_inference=True,  # Skip automatic signature inference to avoid Zarr errors
            pip_requirements=["viscy"],
        )

        # If the function didn't return a path, use the output_directory
        if saved_path is None:
            saved_path = output_directory

        logger.debug(f"Model saved to disk at: {saved_path}")
    except Exception as e:
        raise ValueError(f"Error saving model: {e}")

    logger.debug("\n=== Loading saved model from disk ===")

    # Verify the path exists
    if not os.path.exists(saved_path):
        raise ValueError(f"Warning: Model path does not exist: {saved_path}")

    loaded_model = mlflow.pyfunc.load_model(saved_path)
    logger.info(f"Model successfully loaded from {saved_path}")

    # Show the example input that was saved with the model
    logger.info("\n=== Examining saved input example ===")

    model_info = get_model_info(saved_path)
    if hasattr(model_info, "input_example_path") and model_info.input_example_path:
        logger.debug(
            f"Model includes an input example at {model_info.input_example_path}"
        )

    return saved_path


def demo_predict(model_uri=None):
    """Demonstrates making predictions with a model

    Args:
        model_uri: URI for the model - can be a run_id or a local path
    """
    logger.info("\n=== Making predictions with CytoLand model ===")

    # Create a sample input - a simple gradient image (2D format)
    y, x = np.ogrid[0:256, 0:256]
    sample_input = np.exp(-(1 / 8) * ((x - 32) ** 2 + (y - 32) ** 2) / 64**2).astype(
        np.float32
    )
    logger.debug(f"Input shape: {sample_input.shape}, dtype: {sample_input.dtype}")

    if model_uri:
        # Load the model from MLflow run or local path
        logger.debug(f"Loading model from: {model_uri}")
        if model_uri.startswith("runs:"):
            # It's an MLflow run URI
            loaded_model = mlflow.pyfunc.load_model(model_uri)
        else:
            # It's a local path
            loaded_model = mlflow.pyfunc.load_model(model_uri)

        # Make predictions
        predictions = loaded_model.predict(sample_input)
    else:
        # If no model_uri is provided, create a model instance directly
        logger.debug("No model URI provided. Creating model instance directly.")
        if not check_model_exists():
            return

        # Create the model directly (not through MLflow)
        model = MLFlowCytoLand(
            model_config=MODEL_CONFIG,
            model_ckpt_path=MODEL_CHECKPOINT_PATH,
        )

        predictions = model.predict(context=None, model_input=sample_input)

    if predictions.size > 0:
        logger.debug(f"Prediction shape: {predictions.shape}")
        logger.debug(
            f"Nucleus channel min/max: {predictions[0].min():.4f}/{predictions[0].max():.4f}"
        )
        logger.debug(
            f"Membrane channel min/max: {predictions[1].min():.4f}/{predictions[1].max():.4f}"
        )
    else:
        logger.debug("No predictions returned.")


def main():
    """Main function to run the demo"""

    print("CytoLand2D Demo")
    print("1. Log model to MLflow")
    print("2. Make predictions with a new model instance")
    print("3. Log model to MLflow and make predictions")
    print("4. Save model to disk and make predictions")

    choice = input("Enter your choice (1-4): ")

    if choice == "1":
        demo_log_model()
    elif choice == "2":
        demo_predict()
    elif choice == "3":
        run_id = demo_log_model()
        if run_id:
            try:
                demo_predict(f"runs:/{run_id}/cytoland2d_test")
            except Exception as e:
                print(f"Error during prediction: {e}")
    elif choice == "4":
        saved_path = demo_save_model()
        if os.path.exists(saved_path):
            try:
                demo_predict(saved_path)
            except Exception as e:
                print(f"Error during prediction: {e}")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
