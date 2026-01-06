"""Lightning callback to log training results to Airtable."""

import getpass
from typing import Any

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback

from viscy.airtable.manifests import AirtableManifests


class AirtableLoggingCallback(Callback):
    """
    Log model training to Airtable after training completes.

    This callback automatically records:
    - Best model checkpoint path
    - Who trained the model
    - When it was trained
    - Link to the manifest used

    Parameters
    ----------
    base_id : str
        Airtable base ID
    manifest_id : str
        Airtable manifest record ID (from config)
    model_name : str | None
        Custom model name. If None, auto-generates from model class and timestamp.
    log_metrics : bool
        Whether to log metrics to Airtable (default: False).
        If False, metrics should be viewed in TensorBoard.

    Examples
    --------
    Add to config YAML:

    >>> trainer:
    >>>   callbacks:
    >>>     - class_path: viscy.airtable.callbacks.AirtableLoggingCallback
    >>>       init_args:
    >>>         base_id: "appXXXXXXXXXXXXXX"
    >>>         manifest_id: "recYYYYYYYYYYYYYY"

    Or add programmatically:

    >>> callback = AirtableLoggingCallback(
    >>>     base_id="appXXXXXXXXXXXXXX",
    >>>     manifest_id="recYYYYYYYYYYYYYY"
    >>> )
    >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        base_id: str,
        manifest_id: str,
        model_name: str | None = None,
        log_metrics: bool = False,
    ):
        super().__init__()
        self.registry = AirtableManifests(base_id=base_id)
        self.manifest_id = manifest_id
        self.model_name = model_name
        self.log_metrics = log_metrics

    def on_fit_end(self, trainer: Trainer, pl_module: Any) -> None:
        """Log model to Airtable after training completes."""
        # Get best checkpoint path
        checkpoint_path = None
        if trainer.checkpoint_callback:
            checkpoint_path = trainer.checkpoint_callback.best_model_path
            if not checkpoint_path:  # Fallback to last checkpoint
                checkpoint_path = trainer.checkpoint_callback.last_model_path

        # Generate model name
        if self.model_name:
            model_name = self.model_name
        else:
            model_class = pl_module.__class__.__name__
            logger_version = trainer.logger.version if trainer.logger else "unknown"
            model_name = f"{model_class}_{logger_version}"

        # Optionally collect metrics
        metrics = None
        if self.log_metrics and trainer.callback_metrics:
            metrics = {}
            for key, value in trainer.callback_metrics.items():
                # Only log test metrics or validation metrics
                if "test" in key or "val" in key:
                    try:
                        metrics[key] = float(value)
                    except (TypeError, ValueError):
                        pass  # Skip non-numeric metrics

        # Get logger run ID (works with TensorBoard or MLflow)
        run_id = None
        if trainer.logger:
            if hasattr(trainer.logger, "run_id"):
                run_id = trainer.logger.run_id  # MLflow
            elif hasattr(trainer.logger, "version"):
                run_id = str(trainer.logger.version)  # TensorBoard

        # Log to Airtable
        try:
            model_id = self.registry.log_model_training(
                manifest_id=self.manifest_id,
                mlflow_run_id=run_id or "unknown",
                model_name=model_name,
                checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                trained_by=getpass.getuser(),
                metrics=metrics,
            )
            print(f"\n✓ Model logged to Airtable (record ID: {model_id})")
            print(f"  Model name: {model_name}")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"  Manifest ID: {self.manifest_id}")
        except Exception as e:
            print(f"\n✗ Failed to log to Airtable: {e}")
            # Don't fail training if Airtable logging fails
