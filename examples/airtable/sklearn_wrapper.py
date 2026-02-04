import matplotlib.pyplot as plt
import torch
from lightning.pytorch import LightningModule
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


class SklearnLogisticRegressionModule(LightningModule):
    """
    Wrap sklearn LogisticRegression in Lightning for experiment tracking.

    This module collects features/labels during training_step, then trains
    the sklearn model at the end of each epoch. This pattern allows us to
    use Lightning's logging infrastructure while using sklearn's optimized
    solvers.

    Parameters
    ----------
    input_dim : int
        Feature dimension
    lr : float
        Inverse regularization strength (C parameter for LogisticRegression)
    solver : str
        Solver algorithm
    max_iter : int
        Maximum iterations for solver
    class_weight : str | None
        Class weighting strategy
    """

    def __init__(
        self,
        input_dim: int = 768,
        lr: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Sklearn model (trained incrementally per epoch)
        self.model = LogisticRegression(
            C=lr,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42,
        )

        # Storage for batch features/labels
        self.train_features = []
        self.train_labels = []
        self.val_features = []
        self.val_labels = []

        # Store example input for Lightning compatibility
        self.example_input_array = torch.rand(2, input_dim)

    def forward(self, x):
        """Sklearn doesn't have gradients, so forward just returns input."""
        return x

    def training_step(self, batch, batch_idx):
        """Collect features and labels for end-of-epoch training."""
        features, labels = batch
        self.train_features.append(features.cpu())
        self.train_labels.append(labels.cpu())
        return None  # No loss to backprop

    def on_train_epoch_end(self):
        """Train sklearn model on collected features."""
        X_train = torch.cat(self.train_features, dim=0).numpy()
        y_train = torch.cat(self.train_labels, dim=0).numpy()

        # Train sklearn model
        self.model.fit(X_train, y_train)

        # Compute training metrics
        y_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred)
        train_f1 = f1_score(y_train, y_pred, average="binary")

        # Log metrics (viscy convention: metric/category/stage)
        self.log_dict(
            {
                "metric/accuracy/train": train_acc,
                "metric/f1_score/train": train_f1,
            },
            on_step=False,
            on_epoch=True,
        )

        print(f"\n  Training - Accuracy: {train_acc:.3f}, F1: {train_f1:.3f}")

        # Clear storage
        self.train_features.clear()
        self.train_labels.clear()

    def validation_step(self, batch, batch_idx):
        """Collect validation features and labels."""
        features, labels = batch
        self.val_features.append(features.cpu())
        self.val_labels.append(labels.cpu())
        return None

    def on_validation_epoch_end(self):
        """Evaluate sklearn model on validation set."""
        # Skip if model not fitted yet (happens during sanity check)
        if not hasattr(self.model, "classes_"):
            self.val_features.clear()
            self.val_labels.clear()
            return

        X_val = torch.cat(self.val_features, dim=0).numpy()
        y_val = torch.cat(self.val_labels, dim=0).numpy()

        # Predict
        y_pred = self.model.predict(X_val)

        # Compute metrics
        val_acc = accuracy_score(y_val, y_pred)
        val_f1 = f1_score(y_val, y_pred, average="binary")

        # Log metrics
        self.log_dict(
            {
                "metric/accuracy/val": val_acc,
                "metric/f1_score/val": val_f1,
            },
            on_step=False,
            on_epoch=True,
        )

        print(f"  Validation - Accuracy: {val_acc:.3f}, F1: {val_f1:.3f}")

        # Log confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        if hasattr(self.logger, "experiment"):
            fig, ax = plt.subplots(figsize=(6, 6))
            ConfusionMatrixDisplay(cm, display_labels=["Uninfected", "Infected"]).plot(
                ax=ax, cmap="Blues"
            )
            ax.set_title(f"Confusion Matrix - Epoch {self.current_epoch}")

            # Save and log figure
            fig_path = f"/tmp/confusion_matrix_epoch_{self.current_epoch}.png"
            fig.savefig(fig_path, dpi=100, bbox_inches="tight")
            self.logger.experiment.log_artifact(
                self.logger.run_id, fig_path, artifact_path="plots"
            )
            plt.close(fig)

        # Print classification report
        print(
            "\n"
            + classification_report(
                y_val, y_pred, target_names=["Uninfected", "Infected"]
            )
        )

        # Clear storage
        self.val_features.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        """No optimizer needed for sklearn."""
        return None
