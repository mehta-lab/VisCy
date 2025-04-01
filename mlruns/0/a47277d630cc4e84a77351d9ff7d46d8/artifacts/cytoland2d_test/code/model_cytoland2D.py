"""
MLflow interface for CytoLand2D model
"""

import glob
import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.models import ModelSignature

# Import the correct modules - using viscy instead of vscyto
try:
    from iohub.ngff import open_ome_zarr

    from viscy.data.hcs import HCSDataModule
    from viscy.trainer import VisCyTrainer
    from viscy.transforms import ScaleIntensityRangePercentilesd
    from viscy.translation.engine import FcmaeUNet
    from viscy.translation.predict_writer import HCSPredictionWriter

    HAS_VISCY = True
except ImportError:
    HAS_VISCY = False
    raise ImportError("Warning: viscy not installed. Running in testing mode only.")


class MLFlowCytoLand(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        model_config=None,
        model_ckpt_path=None,
        output_path=None,
    ):
        """
        Initialize the MLFlow wrapper for CytoLand2D model.

        Args:
            model_config (dict): Configuration for CytoLand2D model
            model_ckpt_path (str): Path to model checkpoint
            output_path (str, optional): Path for output files. If None,
                                        temporary directory will be used.
        """
        self.model_config = model_config
        self.model_ckpt_path = model_ckpt_path
        self.output_path = output_path

        # Initialize these as None for lazy loading
        self._model = None
        self._trainer = None
        self._temp_dirs = []

    def _initialize_model_and_trainer(self):
        """Initialize the model and trainer when needed"""
        if self._model is not None and self._trainer is not None:
            return

        if not self.model_config or not self.model_ckpt_path:
            raise ValueError("Model configuration and checkpoint path are required")

        # Check the model_ckpt_path is a valid file
        if not os.path.isfile(self.model_ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.model_ckpt_path}"
            )

        # Initialize the model
        self._model = FcmaeUNet.load_from_checkpoint(
            self.model_ckpt_path, model_config=self.model_config
        )
        self._model.eval()

        tmp_output_dir = tempfile.mkdtemp(prefix="cytoland_predict_")
        self._temp_dirs.append(tmp_output_dir)

        if self.output_path is None:
            output_path = Path(tmp_output_dir) / "output.zarr"
        else:
            output_path = self.output_path

        # Initialize the prediction writer callback
        try:
            # Initialize the trainer with prediction writer
            self._trainer = VisCyTrainer(
                accelerator="auto",
                callbacks=[HCSPredictionWriter(output_store=output_path)],
            )
        except Exception as e:
            self._cleanup_temp_dirs()
            raise RuntimeError(f"Failed to initialize trainer: {e}")

    def _cleanup_temp_dirs(self):
        """Clean up temporary directories"""
        for tmp_dir in self._temp_dirs:
            if os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except Exception as e:
                    print(
                        f"Warning: Failed to clean up temporary directory {tmp_dir}: {e}"
                    )
        self._temp_dirs = []

    @staticmethod
    def preprocess_input(data):
        """
        Preprocess input data to ensure it's in the expected format (TCZYX).

        Args:
            data: Input array in various formats (2D, 3D, 4D, or 5D)

        Returns:
            np.ndarray: Processed array in TCZYX format
        """
        data = np.asarray(data)

        # Handle different input dimensions
        if len(data.shape) == 2:  # YX -> TCZYX
            return data.reshape(1, 1, 1, *data.shape)
        elif len(data.shape) == 3:
            # Check if it's ZYX or CYX format
            if data.shape[0] <= 10:  # Assume it's CYX format
                return data.reshape(1, data.shape[0], 1, *data.shape[1:])
            else:  # Assume it's ZYX format
                return data.reshape(1, 1, *data.shape)
        elif len(data.shape) == 4:
            # Check if it's CZYX or TZYX
            if data.shape[0] <= 10:  # Assume it's CZYX format
                return data.reshape(1, *data.shape)
            else:  # Assume it's TZYX format
                return data.reshape(*data.shape[:2], 1, *data.shape[2:])
        elif len(data.shape) == 5:  # Already TCZYX
            return data
        else:
            raise ValueError(
                f"Unsupported input shape: {data.shape}. Expected 2D, 3D, 4D, or 5D array."
            )

    def load_context(self, context):
        """
        Load model context when MLflow loads the model.

        Args:
            context: MLflow model context containing artifacts
        """

        # Load model configuration from context
        if not self.model_config and context.artifacts.get("model_config.json"):
            try:
                with open(context.artifacts["model_config.json"], "r") as f:
                    self.model_config = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load model configuration: {e}")

        # Find checkpoint file in artifacts
        if not self.model_ckpt_path:
            checkpoint_files = [f for f in context.artifacts if f.endswith(".ckpt")]
            if checkpoint_files:
                self.model_ckpt_path = context.artifacts[checkpoint_files[0]]
            else:
                # Look for a directory that might contain a checkpoint
                for artifact_name, artifact_path in context.artifacts.items():
                    if os.path.isdir(artifact_path):
                        # Check if there's a checkpoint file in this directory
                        checkpoint_files = glob.glob(
                            os.path.join(artifact_path, "*.ckpt")
                        )
                        if checkpoint_files:
                            self.model_ckpt_path = checkpoint_files[0]
                            break

        if not self.model_config or not self.model_ckpt_path:
            raise ValueError("Model configuration or checkpoint not found in artifacts")

    def predict(self, context, model_input):
        """
        Predict using the CytoLand2D model.

        Args:
            context: MLflow model context
            model_input: Input data (can be 2D, 3D, 4D, or 5D)

        Returns:
            np.ndarray: Predicted output (CYX format)
        """
        # If this is a signature inference call (context may be None during signature inference)
        if context is None and getattr(
            mlflow.pyfunc, "_is_signature_inference_call", False
        ):
            print(
                "Detected signature inference call. Returning dummy output for signature inference."
            )
            # Return a properly shaped dummy output based on input shape
            input_shape = np.asarray(model_input).shape
            if len(input_shape) == 2:
                return np.zeros((2, *input_shape), dtype=np.float32)
            elif len(input_shape) >= 3:
                return np.zeros((2, *input_shape[-2:]), dtype=np.float32)
            return np.zeros((2, 256, 256), dtype=np.float32)

        # Ensure model and trainer are initialized
        self._initialize_model_and_trainer()

        # Preprocess input to ensure it's in the correct format
        processed_data = self.preprocess_input(model_input)

        # Create temporary directory for input data
        tmp_input_dir = tempfile.mkdtemp(prefix="cytoland_input_")
        self._temp_dirs.append(tmp_input_dir)

        # Create temporary input Zarr file
        tmp_input_zarr = Path(tmp_input_dir) / "input.zarr"

        try:
            # Write input data to Zarr file
            with open_ome_zarr(
                tmp_input_zarr,
                channel_names=["Phase3D"],
                mode="w",
                layout="hcs",
            ) as tmp_plate:
                position = tmp_plate.create_position("0", "0", "0")
                position.create_zeros(
                    "0", shape=(1, 1, *processed_data.shape[-3:]), dtype=np.float32
                )
            # Write the input data to the mock ome-zarr file
            with open_ome_zarr(tmp_input_zarr / "0/0/0", mode="r+") as dataset:
                dataset[0][:] = processed_data

            # Create data module for prediction
            data_module = HCSDataModule(
                data_path=tmp_input_zarr / "0/0/0",
                source_channel="Phase3D",
                target_channel=["nuclei", "membrane"],
                z_window_size=1,
                batch_size=1,
                num_workers=13,
                normalizations=[
                    ScaleIntensityRangePercentilesd(
                        keys=["Phase3D"],
                        lower=0.5,
                        upper=99.5,
                        b_min=0.0,
                        b_max=1.0,
                    )
                ],
            )
            data_module.prepare_data()
            data_module.setup(stage="predict")

            # Run prediction

            # TODO: set the matmul to high precision if possible
            # torch.set_float32_matmul_precision("high")
            self._trainer.predict(
                self._model, datamodule=data_module, return_predictions=False
            )
            # Read output from Zarr file
            if self.output_path is None:
                output_path = Path(self._temp_dirs[0]) / "output.zarr"
            else:
                output_path = self.output_path

            # Check if output Zarr exists
            if not os.path.exists(output_path):
                print(f"Output Zarr not found: {output_path}")
                if context is None:  # Might be a signature inference call
                    return np.zeros((2, 256, 256), dtype=np.float32)
                raise FileNotFoundError(f"Output Zarr not found: {output_path}")

            # Open Zarr file and read predictions
            with open_ome_zarr(output_path / "0/0/0", mode="r") as z_output:
                # Try different paths to find predictions
                predictions = z_output["0"][:]

            # Convert to CYX format (remove time and Z dimensions)
            if len(predictions.shape) == 5:  # TCZYX
                predictions = predictions[0, :, 0]  # CYX
            return predictions

        except Exception as e:
            raise RuntimeError(f"Failed to make prediction: {e}")

        finally:
            # Clean up temporary directories
            self._cleanup_temp_dirs()


def log_cytoland_model(
    model_config,
    model_path,
    model_name="cytoland2d",
    output_path=None,
    registered_model_name=None,
    input_example=None,
    pip_requirements=None,
    skip_signature_inference=False,
):
    """
    Log a CytoLand model to MLflow.

    Args:
        model_config (dict): Model configuration
        model_path (str): Path to model checkpoint
        model_name (str): Name of the model
        output_path (str, optional): Path for model output files. Defaults to None.
        registered_model_name (str, optional): Name to register the model. Defaults to None.
        input_example (ndarray, optional): Example input for model signature. Defaults to None.
        pip_requirements (list or str, optional): Pip requirements for the model. Defaults to None.
        skip_signature_inference (bool, optional): Whether to skip signature inference. Defaults to False.

    Returns:
        str: MLflow run ID
    """
    # Get the current file path for code_paths parameter

    current_file = inspect.getfile(inspect.currentframe())
    # Use the full absolute path, not just the basename
    code_path = os.path.abspath(current_file)

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Configure model
        model = MLFlowCytoLand(
            model_config=model_config,
            model_ckpt_path=model_path,
            output_path=output_path,
        )

        # Create a simple 2D example input if none is provided
        if input_example is None:
            y, x = np.ogrid[0:256, 0:256]
            input_example = np.exp(
                -(1 / 8) * ((x - 32) ** 2 + (y - 32) ** 2) / 64**2
            ).astype(np.float32)
            print(
                f"Created default input example with shape: {input_example.shape}, dtype: {input_example.dtype}"
            )
        else:
            # Ensure the input example is 2D (YX format) for MLflow signature and float32 dtype
            input_example = np.asarray(input_example, dtype=np.float32)
            if len(input_example.shape) == 5:  # TCZYX format
                print(
                    f"Converting input example from TCZYX to YX format: {input_example.shape}"
                )
                input_example = input_example[0, 0, 0]  # Extract first slice
            elif len(input_example.shape) == 4:  # CZYX or TZYX format
                print(
                    f"Converting input example from 4D to YX format: {input_example.shape}"
                )
                input_example = input_example[0, 0]  # Extract first slice
            elif len(input_example.shape) == 3:  # ZYX or CYX format
                print(
                    f"Converting input example from 3D to YX format: {input_example.shape}"
                )
                input_example = input_example[0]  # Extract first slice
            # No conversion needed if it's already 2D
            print(
                f"Using input example with shape: {input_example.shape}, dtype: {input_example.dtype}"
            )

        # Check if model checkpoint file exists
        if os.path.exists(model_path):
            # Create a checkpoints directory in artifacts
            artifacts = {}
            checkpoints_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)

            checkpoint_basename = os.path.basename(model_path)
            checkpoint_dest = os.path.join(checkpoints_dir, checkpoint_basename)
            shutil.copy(model_path, checkpoint_dest)
            artifacts["checkpoints"] = checkpoints_dir

            config_path = os.path.join(tempfile.mkdtemp(), "model_config.json")
            with open(config_path, "w") as f:
                json.dump(model_config, f)
            artifacts["model_config.json"] = config_path
        else:
            print(f"Warning: Model checkpoint file not found: {model_path}")
            artifacts = None

        # Set pip requirements
        if pip_requirements is None:
            pip_requirements = [
                "viscy>=0.2.0",
            ]

        # Prepare signature if not skipping signature inference
        signature = None
        if not skip_signature_inference:
            try:
                # Create a simple fixed signature instead of inferring
                from mlflow.types.schema import Schema, TensorSpec

                # Define input schema - 2D float array
                input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, -1))])
                # Define output schema - 2-channel 2D float array
                output_schema = Schema([TensorSpec(np.dtype(np.float32), (2, -1, -1))])
                # Create signature
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                print("Created fixed model signature")
            except Exception as e:
                print(f"Warning: Failed to create model signature: {e}")

        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=model,
            artifacts=artifacts,
            code_paths=[code_path],  # Use the full absolute path
            pip_requirements=pip_requirements,
            input_example=input_example,
            registered_model_name=registered_model_name,
            signature=signature,  # Pass our prepared signature
        )

        # Log model parameters
        mlflow.log_params(model_config)
        mlflow.log_param("model_path", model_path)
        if output_path:
            mlflow.log_param("output_path", output_path)

        return run.info.run_id


def save_cytoland_model(
    model_config,
    model_path,
    output_directory,
    output_path=None,
    input_example=None,
    pip_requirements=None,
    overwrite=False,
    skip_signature_inference=False,
):
    """
    Save a CytoLand model to disk without MLflow tracking.

    Args:
        model_config (dict): Model configuration
        model_path (str): Path to model checkpoint
        output_directory (str): Directory to save the model
        output_path (str, optional): Path for model output files. Defaults to None.
        input_example (ndarray, optional): Example input for model signature. Defaults to None.
        pip_requirements (list or str, optional): Pip requirements for the model. Defaults to None.
        overwrite (bool, optional): Whether to overwrite the output directory if it exists. Defaults to False.
        skip_signature_inference (bool, optional): Whether to skip signature inference. Defaults to False.

    Returns:
        str: Path to saved model
    """
    # Get the current file path for code_paths parameter

    current_file = inspect.getfile(inspect.currentframe())
    # Use the full absolute path, not just the basename
    code_path = os.path.abspath(current_file)

    # If overwrite is True and the directory exists, remove it first
    if overwrite and os.path.exists(output_directory):
        print(f"Overwriting existing directory: {output_directory}")
        shutil.rmtree(output_directory)

    # Configure model
    model = MLFlowCytoLand(
        model_config=model_config,
        model_ckpt_path=model_path,
        output_path=output_path,
    )

    # Create a simple 2D example input if none is provided
    if input_example is None:
        y, x = np.ogrid[0:256, 0:256]
        input_example = np.exp(
            -(1 / 8) * ((x - 32) ** 2 + (y - 32) ** 2) / 64**2
        ).astype(np.float32)
        print(
            f"Created default input example with shape: {input_example.shape}, dtype: {input_example.dtype}"
        )
    else:
        # Ensure the input example is 2D (YX format) for MLflow signature and float32 dtype
        input_example = np.asarray(input_example, dtype=np.float32)
        if len(input_example.shape) == 5:  # TCZYX format
            print(
                f"Converting input example from TCZYX to YX format: {input_example.shape}"
            )
            input_example = input_example[0, 0, 0]  # Extract first slice
        elif len(input_example.shape) == 4:  # CZYX or TZYX format
            print(
                f"Converting input example from 4D to YX format: {input_example.shape}"
            )
            input_example = input_example[0, 0]  # Extract first slice
        elif len(input_example.shape) == 3:  # ZYX or CYX format
            print(
                f"Converting input example from 3D to YX format: {input_example.shape}"
            )
            input_example = input_example[0]  # Extract first slice
        # No conversion needed if it's already 2D
        print(
            f"Using input example with shape: {input_example.shape}, dtype: {input_example.dtype}"
        )

    # Check if model checkpoint file exists
    if os.path.exists(model_path):
        # Create a checkpoints directory in artifacts
        artifacts = {}
        checkpoints_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Copy checkpoint file to artifacts
        checkpoint_basename = os.path.basename(model_path)
        checkpoint_dest = os.path.join(checkpoints_dir, checkpoint_basename)
        shutil.copy(model_path, checkpoint_dest)
        artifacts["checkpoints"] = checkpoints_dir

        # Save model config to artifacts
        config_path = os.path.join(tempfile.mkdtemp(), "model_config.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f)
        artifacts["model_config.json"] = config_path
    else:
        print(f"Warning: Model checkpoint file not found: {model_path}")
        artifacts = None

    # Set pip requirements
    if pip_requirements is None:
        pip_requirements = [
            "viscy>=0.2.0",
        ]

    # Prepare signature if not skipping signature inference
    signature = None
    if not skip_signature_inference:
        try:
            # Create a simple fixed signature instead of inferring
            from mlflow.types.schema import Schema, TensorSpec

            # Define input schema - 2D float array
            input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, -1))])
            # Define output schema - 2-channel 2D float array
            output_schema = Schema([TensorSpec(np.dtype(np.float32), (2, -1, -1))])
            # Create signature
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            print("Created fixed model signature")
        except Exception as e:
            print(f"Warning: Failed to create model signature: {e}")

    # Save the model
    # Pass the full absolute path to the file, not just the basename
    mlflow.pyfunc.save_model(
        path=output_directory,
        python_model=model,
        artifacts=artifacts,
        code_paths=[code_path],  # Use the full absolute path
        pip_requirements=pip_requirements,
        input_example=input_example,
        signature=signature,  # Pass our prepared signature
    )

    # Return the path to the saved model directory
    return output_directory
