import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import numpy as np

from viscy.callbacks.gradcam import GradCAMCallback


class ResNetClassifier(LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load pretrained ResNet18
        self.model = torchvision.models.resnet18(pretrained=True)

        # Replace final layer for CIFAR-10
        self.model.fc = nn.Linear(512, num_classes)

        # Save the target layer for GradCAM
        self.target_layer = self.model.layer4[-1]

        # Ensure gradients are enabled for the target layer
        for param in self.target_layer.parameters():
            param.requires_grad = True

        self.gradients = None
        self.activations = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    # GradCAM methods
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations(self, x):
        return self.activations

    def gradcam(self, x):
        # Store original training mode and switch to eval mode
        was_training = self.training
        self.eval()  # Use eval mode for inference

        try:
            # Register hooks
            h = self.target_layer.register_forward_hook(
                lambda module, input, output: setattr(self, "activations", output)
            )
            h_bp = self.target_layer.register_backward_hook(
                lambda module, grad_in, grad_out: self.activations_hook(grad_out[0])
            )

            # Forward pass
            x = x.unsqueeze(0).to(self.device)  # Add batch dimension

            # Enable gradients for the entire computation
            with torch.enable_grad():
                x = x.requires_grad_(True)
                output = self(x)

                # Get predicted class
                pred = output.argmax(dim=1)

                # Create one hot vector for backward pass
                one_hot = torch.zeros_like(output, device=self.device)
                one_hot[0][pred] = 1

                # Clear gradients
                self.zero_grad(set_to_none=False)

                # Backward pass
                output.backward(gradient=one_hot)

                # Generate GradCAM
                gradients = self.gradients
                activations = self.activations

                # Ensure we have valid gradients
                if gradients is None:
                    raise RuntimeError("No gradients available for GradCAM computation")

                weights = torch.mean(gradients, dim=(2, 3))
                cam = torch.sum(weights[:, :, None, None] * activations, dim=1)
                cam = F.relu(cam)
                cam = (
                    F.interpolate(
                        cam.unsqueeze(0),
                        size=x.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )

                return cam

        finally:
            # Clean up
            h.remove()
            h_bp.remove()
            # Restore original training mode
            self.train(mode=was_training)


def main():
    # Data transforms
    transform = transforms.Compose(
        [
            transforms.Resize(224),  # ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Create small visualization dataset
    vis_dataset = torch.utils.data.Subset(val_dataset, indices=range(10))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    vis_loader = DataLoader(vis_dataset, batch_size=32)  # Added visualization loader

    # Initialize model
    model = ResNetClassifier()

    # Initialize callbacks
    gradcam_callback = GradCAMCallback(
        visual_datasets=[vis_dataset],
        every_n_epochs=1,  # Generate visualizations every epoch
        max_samples=5,  # Visualize 5 samples
        max_height=224,  # Match ResNet input size
    )

    # Initialize trainer with specific logger
    trainer = Trainer(
        max_epochs=5,
        callbacks=[gradcam_callback],
        accelerator="auto",
        devices=1,
        logger=TensorBoardLogger(
            save_dir="/home/eduardo.hirata/repos/viscy/tests/representation/lightning_logs",  # specify your desired log directory
            name="gradcam_cifar",  # experiment name
        ),
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Test GradCAM visualization
    test_gradcam_visualization(model, vis_loader)


def test_gradcam_visualization(model, dataloader):
    """Test GradCAM visualization.

    Parameters
    ----------
    model : LightningModule
        The trained model
    dataloader : DataLoader
        DataLoader containing samples to visualize
    """
    model.eval()
    # Get a sample from validation set
    batch = next(iter(dataloader))
    images, labels = batch

    # Generate GradCAM for first sample
    sample_img = images[0]  # Shape: (C, H, W)
    cam = model.gradcam(sample_img)

    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Original image
    img = images[0].squeeze().cpu().numpy()
    if img.ndim == 3:  # Handle RGB images
        axes[0].imshow(np.transpose(img, (1, 2, 0)))
    else:  # Handle grayscale images
        axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Image")
    plt.colorbar(axes[0].images[0], ax=axes[0])

    # GradCAM visualization
    im = axes[1].imshow(cam, cmap="magma")
    axes[1].set_title("GradCAM")
    plt.colorbar(im, ax=axes[1])

    # Overlay GradCAM on original image
    img = images[0].squeeze().cpu().numpy()
    if img.ndim == 3:  # Handle RGB images
        img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize to [0,1]

    # Create RGB overlay
    if img.ndim == 2:  # Convert grayscale to RGB
        img_rgb = np.stack([img] * 3, axis=-1)
    else:  # Already RGB
        img_rgb = img
    cam_rgb = plt.cm.magma(cam_norm)[..., :3]  # Convert to RGB using magma colormap
    overlay = 0.7 * img_rgb + 0.3 * cam_rgb

    axes[2].imshow(overlay)
    axes[2].set_title("GradCAM Overlay")

    plt.suptitle(f"GradCAM Visualization (Predicted: {labels[0].item()})", y=1.05)
    plt.savefig("./gradcam_cifar.png")
    plt.close()
    # plt.show()


if __name__ == "__main__":
    main()
