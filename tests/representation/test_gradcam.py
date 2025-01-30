import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger

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
            save_dir="your/log/path",  # specify your desired log directory
            name="gradcam_experiment",  # experiment name
        ),
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
