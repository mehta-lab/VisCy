"""Lightning callback to log training results to Airtable."""

import getpass
from typing import Any

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback

from viscy.airtable.database import AirtableManager


class AirtableLoggingCallback(Callback):
    """
    Log model training to Airtable after training completes.

    This callback automatically records:
    - Best model checkpoint path
    - Who trained the model
    - When it was trained
    - Link to the collection used

    Parameters
    ----------
    base_id : str
        Airtable base ID
    collection_id : str
        Airtable collection record ID (from config)
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
    >>>         collection_id: "recYYYYYYYYYYYYYY"

    Or add programmatically:

    >>> callback = AirtableLoggingCallback(
    >>>     base_id="appXXXXXXXXXXXXXX",
    >>>     collection_id="recYYYYYYYYYYYYYY"
    >>> )
    >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        base_id: str,
        collection_id: str,
        model_name: str | None = None,
        log_metrics: bool = False,
    ):
        super().__init__()
        self.airtable_db = AirtableManager(base_id=base_id)
        self.collection_id = collection_id
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
            model_id = self.airtable_db.log_model_training(
                collection_id=self.collection_id,
                mlflow_run_id=run_id or "unknown",
                model_name=model_name,
                checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                trained_by=getpass.getuser(),
                metrics=metrics,
            )
            print(f"\n✓ Model logged to Airtable (record ID: {model_id})")
            print(f"  Model name: {model_name}")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"  Collections ID: {self.collection_id}")
        except Exception as e:
            print(f"\n✗ Failed to log to Airtable: {e}")
            # Don't fail training if Airtable logging fails


class CollectionWandbCallback(Callback):
    """
    Log collection metadata to Weights & Biases automatically.

    This callback extracts collection information from CollectionTripletDataModule
    and logs it to W&B config for searchability and lineage tracking.

    Examples
    --------
    Add to config YAML:

    >>> trainer:
    >>>   logger:
    >>>     class_path: lightning.pytorch.loggers.WandbLogger
    >>>     init_args:
    >>>       project: viscy-experiments
    >>>       log_model: false
    >>>   callbacks:
    >>>     - class_path: viscy.airtable.callbacks.CollectionWandbCallback

    Or add programmatically:

    >>> from lightning.pytorch.loggers import WandbLogger
    >>> logger = WandbLogger(project="viscy-experiments")
    >>> callback = CollectionWandbCallback()
    >>> trainer = Trainer(logger=logger, callbacks=[callback])
    """

    def on_train_start(self, trainer: Trainer, pl_module: Any) -> None:
        """Log collection metadata to W&B config at training start."""
        # Import here to avoid requiring wandb as a dependency
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError:
            return  # Skip if wandb not installed

        # Check if using WandbLogger
        if not isinstance(trainer.logger, WandbLogger):
            return

        # Check if using CollectionTripletDataModule
        from viscy.airtable.factory import CollectionTripletDataModule

        dm = trainer.datamodule

        # Log collection metadata if using CollectionTripletDataModule
        if isinstance(dm, CollectionTripletDataModule):
            collection_config = {
                "collection/name": dm.collection_name,
                "collection/version": dm.collection_version,
                "collection/base_id": dm.base_id,
                "collection/data_path": str(dm.data_path),
                "collection/tracks_path": str(dm.tracks_path),
            }
            trainer.logger.experiment.config.update(collection_config)

            print("\n✓ Collections metadata logged to W&B:")
            print(f"  Collections: {dm.collection_name} v{dm.collection_version}")
            print(f"  Data path: {dm.data_path}")
            print(f"  Tracks path: {dm.tracks_path}")

        # Also log data module hyperparameters explicitly
        if dm is not None and hasattr(dm, "hparams"):
            data_config = {f"data/{k}": v for k, v in dm.hparams.items()}
            trainer.logger.experiment.config.update(data_config, allow_val_change=True)

        # Log model hyperparameters explicitly
        if hasattr(pl_module, "hparams"):
            model_config = {f"model/{k}": v for k, v in pl_module.hparams.items()}
            trainer.logger.experiment.config.update(model_config, allow_val_change=True)
