# %% lightning moules for infection classification using the viscy library

# import torchview
from typing import Literal, Sequence

import cv2
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.cm import get_cmap
from monai.transforms import DivisiblePad
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
from torch import Tensor

# from viscy.unet.networks.Unet25D import Unet25d
from viscy.data.hcs import Sample
from viscy.unet.networks.Unet2D import Unet2d

#
# %% Methods to compute confusion matrix per cell using torchmetrics


# The confusion matrix at the single-cell resolution.
def confusion_matrix_per_cell(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
):
    """Compute confusion matrix per cell.

    Args:
        y_true (torch.Tensor): Ground truth label image (BXHXW).
        y_pred (torch.Tensor): Predicted label image (BXHXW).
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Confusion matrix per cell (BXCXC).
    """
    # Convert the image class to the nuclei class
    confusion_matrix_per_cell = compute_confusion_matrix(y_true, y_pred, num_classes)
    confusion_matrix_per_cell = torch.tensor(confusion_matrix_per_cell)
    return confusion_matrix_per_cell


# confusion matrix computation
def compute_confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
):
    """Convert the class of the image to the class of the nuclei.

    Args:
        label_image (torch.Tensor): Label image (BXHXW). Values of tensor are integers that represent semantic segmentation.
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Label images with a consensus class at the centroid of nuclei.
    """

    batch_size = y_true.size(0)
    # find centroids of nuclei from y_true
    conf_mat = np.zeros((num_classes, num_classes))
    for i in range(batch_size):
        y_true_cpu = y_true[i].cpu().numpy()
        y_pred_cpu = y_pred[i].cpu().numpy()
        y_true_reshaped = y_true_cpu.reshape(y_true_cpu.shape[-2:])
        y_pred_reshaped = y_pred_cpu.reshape(y_pred_cpu.shape[-2:])

        y_pred_resized = cv2.resize(
            y_pred_reshaped,
            dsize=y_true_reshaped.shape[::-1],
            interpolation=cv2.INTER_NEAREST,
        )
        y_pred_resized = np.where(y_true_reshaped > 0, y_pred_resized, 0)

        # find objects in every image
        label_img = label(y_true_reshaped)
        regions = regionprops(label_img)

        # Find centroids, pixel coordinates from the ground truth.
        for region in regions:
            if region.area > 0:
                row, col = region.centroid
                pred_id = y_pred_resized[int(row), int(col)]
                test_id = y_true_reshaped[int(row), int(col)]

                if pred_id == 1 and test_id == 1:
                    conf_mat[1, 1] += 1
                if pred_id == 1 and test_id == 2:
                    conf_mat[0, 1] += 1
                if pred_id == 2 and test_id == 1:
                    conf_mat[1, 0] += 1
                if pred_id == 2 and test_id == 2:
                    conf_mat[0, 0] += 1
        # Find all instances of nuclei in ground truth and compute the class of the nuclei in both ground truth and prediction.
    # Find all instances of nuclei in ground truth and compute the class of the nuclei in both ground truth and prediction.
    return conf_mat


# plot the computed confusion matrix
def plot_confusion_matrix(confusion_matrix, index_to_label_dict):
    # Create a figure and axis to plot the confusion matrix
    fig, ax = plt.subplots()

    # Create a color heatmap for the confusion matrix
    cax = ax.matshow(confusion_matrix, cmap="viridis")

    # Create a colorbar and set the label
    index_to_label_dict = dict(
        enumerate(index_to_label_dict)
    )  # Convert list to dictionary
    fig.colorbar(cax, label="Frequency")

    # Set labels for the classes
    ax.set_xticks(np.arange(len(index_to_label_dict)))
    ax.set_yticks(np.arange(len(index_to_label_dict)))
    ax.set_xticklabels(index_to_label_dict.values(), rotation=45)
    ax.set_yticklabels(index_to_label_dict.values())

    # Set labels for the axes
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Add text annotations to the confusion matrix
    for i in range(len(index_to_label_dict)):
        for j in range(len(index_to_label_dict)):
            ax.text(
                j,
                i,
                str(int(confusion_matrix[i, j])),
                ha="center",
                va="center",
                color="white",
            )

    return fig


class SemanticSegUNet2D(pl.LightningModule):

    # Model for semantic segmentation.

    def __init__(
        self,
        in_channels: int,  # Number of input channels
        out_channels: int,  # Number of output channels
        lr: float = 1e-4,  # Learning rate
        loss_function: nn.Module = nn.CrossEntropyLoss(),  # Loss function
        schedule: Literal[
            "WarmupCosine", "Constant"
        ] = "Constant",  # Learning rate schedule
        log_batches_per_epoch: int = 2,  # Number of batches to log per epoch
        log_samples_per_batch: int = 2,  # Number of samples to log per batch
        ckpt_path: str = None,  # Path to the checkpoint
    ):
        super(SemanticSegUNet2D, self).__init__()  # Call the superclass initializer
        # Initialize the UNet model
        self.unet_model = Unet2d(in_channels=in_channels, out_channels=out_channels)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
            state_dict.pop("loss_function.weight", None)  # Remove the unexpected key
            self.load_state_dict(state_dict)  # loading only weights
        self.lr = lr  # Set the learning rate
        # Set the loss function to CrossEntropyLoss if none is provided
        self.loss_function = loss_function if loss_function else nn.CrossEntropyLoss()
        self.schedule = schedule  # Set the learning rate schedule
        self.log_batches_per_epoch = (
            log_batches_per_epoch  # Set the number of batches to log per epoch
        )
        self.log_samples_per_batch = (
            log_samples_per_batch  # Set the number of samples to log per batch
        )
        self.training_step_outputs = []  # Initialize the list of training step outputs
        self.validation_step_outputs = (
            []
        )  # Initialize the list of validation step outputs

        self.pred_cm = None  # Initialize the confusion matrix
        self.index_to_label_dict = ["Infected", "Uninfected"]

    # Define the forward pass
    def forward(self, x):
        return self.unet_model(x)  # Pass the input through the UNet model

    # Define the optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr
        )  # Use the Adam optimizer
        return optimizer

    # Define the training step
    def training_step(self, batch: Sample, batch_idx: int):
        """
        The training step for the model.
        This method is called for every batch during the training process.
        """
        source = batch["source"]  # Extract the source from the batch
        target = batch["target"]  # Extract the target from the batch
        pred = self.forward(source)  # Make a prediction using the source
        # Convert the target to one-hot encoding
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=3).permute(
            0, 4, 1, 2, 3
        )
        target_one_hot = target_one_hot.float()  # Convert the target to float type
        train_loss = self.loss_function(pred, target_one_hot)  # Calculate the loss
        # Log the training step outputs if the batch index is less than the number of batches to log per epoch
        if batch_idx < self.log_batches_per_epoch:
            self.training_step_outputs.extend(
                self._detach_sample((source, target_one_hot, pred))
            )
        # Log the training loss
        self.log(
            "loss/train",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return train_loss  # Return the training loss

    def validation_step(self, batch: Sample, batch_idx: int):
        """
        The validation step for the model.
        """
        source = batch["source"]  # Extract the source from the batch
        target = batch["target"]  # Extract the target from the batch
        pred = self.forward(source)  # Make a prediction using the source
        # Convert the target to one-hot encoding
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=3).permute(
            0, 4, 1, 2, 3
        )
        target_one_hot = target_one_hot.float()  # Convert the target to float type
        loss = self.loss_function(pred, target_one_hot)  # Calculate the loss
        # Log the validation step outputs if the batch index is less than the number of batches to log per epoch
        if batch_idx < self.log_batches_per_epoch:
            self.validation_step_outputs.extend(
                self._detach_sample((source, target_one_hot, pred))
            )
        # Log the validation loss
        self.log(
            "loss/validate", loss, sync_dist=True, add_dataloader_idx=False, logger=True
        )
        return loss  # Return the validation loss

    def on_predict_start(self):
        """Pad the input shape to be divisible by the downsampling factor.
        The inverse of this transform crops the prediction to original shape.
        """
        down_factor = 2**self.unet_model.num_blocks
        self._predict_pad = DivisiblePad((0, 0, down_factor, down_factor))

    # Define the prediction step
    def predict_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0):

        source = self._predict_pad(batch["source"])  # Pad the source
        logits = self._predict_pad.inverse(
            self.forward(source)
        )  # Predict and remove padding.
        prob_pred = F.softmax(logits, dim=1)  # Calculate the probabilities
        # Go from probabilities/one-hot encoded data to class labels.
        labels_pred = torch.argmax(
            prob_pred, dim=1, keepdim=True
        )  # Calculate the predicted labels

        return labels_pred  # log the class predicted image

    def on_test_start(self):
        self.pred_cm = torch.zeros((2, 2))
        down_factor = 2**self.unet_model.num_blocks
        self._predict_pad = DivisiblePad((0, 0, down_factor, down_factor))

    def test_step(self, batch: Sample):
        source = self._predict_pad(batch["source"])  # Pad the source
        # predict_writer(batch["source"], f"test_source_{self.i_num}.npy")
        logits = self._predict_pad.inverse(self.forward(source))
        prob_pred = F.softmax(logits, dim=1)  # Calculate the probabilities
        labels_pred = torch.argmax(
            prob_pred, dim=1, keepdim=True
        )  # Calculate the predicted labels

        target = self._predict_pad(batch["target"])  # Extract the target from the batch
        pred_cm = confusion_matrix_per_cell(
            target, labels_pred, num_classes=2
        )  # Calculate the confusion matrix per cell
        self.pred_cm += pred_cm  # Append the confusion matrix to pred_cm

        self.logger.experiment.add_figure(
            "Confusion Matrix per Cell",
            plot_confusion_matrix(pred_cm, self.index_to_label_dict),
            self.current_epoch,
        )

    # Accumulate the confusion matrix at the end of test epoch and log.
    def on_test_end(self):
        confusion_matrix_sum = self.pred_cm
        self.logger.experiment.add_figure(
            "Confusion Matrix",
            plot_confusion_matrix(confusion_matrix_sum, self.index_to_label_dict),
            self.current_epoch,
        )

    # Define what happens at the end of a training epoch
    def on_train_epoch_end(self):
        self._log_samples(
            "train_samples", self.training_step_outputs
        )  # Log the training samples
        self.training_step_outputs = []  # Reset the list of training step outputs

    # Define what happens at the end of a validation epoch
    def on_validation_epoch_end(self):
        self._log_samples(
            "val_samples", self.validation_step_outputs
        )  # Log the validation samples
        self.validation_step_outputs = []  # Reset the list of validation step outputs

    # Define a method to detach a sample
    def _detach_sample(self, imgs: Sequence[Tensor]):
        # Detach the images and convert them to numpy arrays
        num_samples = 3
        return [
            [np.squeeze(img[i].detach().cpu().numpy().max(axis=1)) for img in imgs]
            for i in range(num_samples)
        ]

    # Define a method to log samples
    def _log_samples(self, key: str, imgs: Sequence[Sequence[np.ndarray]]):
        images_grid = []  # Initialize the list of image grids
        for sample_images in imgs:  # For each sample image
            images_row = []  # Initialize the list of image rows
            for i, image in enumerate(
                sample_images
            ):  # For each image in the sample images
                cm_name = "gray" if i == 0 else "inferno"  # Set the colormap name
                if image.ndim == 2:  # If the image is 2D
                    image = image[np.newaxis]  # Add a new axis
                for channel in image:  # For each channel in the image
                    channel = rescale_intensity(
                        channel, out_range=(0, 1)
                    )  # Rescale the intensity of the channel
                    render = get_cmap(cm_name)(channel, bytes=True)[
                        ..., :3
                    ]  # Render the channel
                    images_row.append(
                        render
                    )  # Append the render to the list of image rows
            images_grid.append(
                np.concatenate(images_row, axis=1)
            )  # Append the concatenated image rows to the list of image grids
        grid = np.concatenate(images_grid, axis=0)  # Concatenate the image grids
        # Log the image grid
        self.logger.experiment.add_image(
            key, grid, self.current_epoch, dataformats="HWC"
        )


# %%
