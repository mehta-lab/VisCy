import logging

import lightning.pytorch as pl
import numpy as np
import torch
from cmap import Colormap
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Dummy dataset with synthetic RGB images using cmap."""
    
    def __init__(self, size: int = 500, num_classes: int = 5, image_size: int = 64):
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        
        torch.manual_seed(42)
        self.labels = torch.randint(0, num_classes, (size,))
        
        # NOTE: hardcoding annotation labels: classes 0,1,2 = positive (1), classes 3,4 = negative (0)
        self.annotations = torch.zeros(size, dtype=torch.int)
        for i in range(size):
            class_id = self.labels[i].item()
            self.annotations[i] = 1 if class_id in [0, 1, 2] else 0
        
        self.images = self._generate_synthetic_rgb_images()
        
    def _generate_synthetic_rgb_images(self):
        """Generate synthetic RGB images using single colormaps per class."""
        # Single colormap per class for simplicity
        class_cmaps = [
            'green',    # Class 0
            'magenta',  # Class 1  
            'cyan',     # Class 2
            'orange',   # Class 3
            'blue'      # Class 4
        ]
        
        rgb_images = []
        
        for i in range(self.size):
            class_id = self.labels[i].item()
            cmap_name = class_cmaps[class_id]
            
            # Create single channel pattern
            pattern = np.zeros((self.image_size, self.image_size))
            
            # Class-specific geometric pattern
            if class_id == 0:  # Circles
                y, x = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size), indexing='ij')
                center = self.image_size // 2
                radius = self.image_size // 4
                pattern = ((x - center) ** 2 + (y - center) ** 2 <= radius ** 2).astype(float)
                
            elif class_id == 1:  # Horizontal stripes
                for j in range(0, self.image_size, 8):
                    pattern[j:j+4, :] = 1.0
                    
            elif class_id == 2:  # Vertical stripes  
                for j in range(0, self.image_size, 8):
                    pattern[:, j:j+4] = 1.0
                    
            elif class_id == 3:  # Diagonal pattern
                y, x = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size), indexing='ij')
                pattern = ((x + y) % 10 < 5).astype(float)
                
            else:  # class_id == 4: Checkerboard
                y, x = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size), indexing='ij')
                pattern = ((x // 8 + y // 8) % 2).astype(float)
            
            # Add some Gaussian blobs
            y, x = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size), indexing='ij')
            blob_centers = np.random.RandomState(i).rand(2, 2) * self.image_size
            for center in blob_centers:
                cx, cy = center
                sigma = self.image_size // 6
                gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
                pattern += gaussian * 0.5
            
            # Normalize and add noise
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
            noise = np.random.RandomState(i).randn(*pattern.shape) * 0.1
            pattern = np.clip(pattern + noise, 0, 1)
            
            # Apply colormap to create RGB
            colormap = Colormap(cmap_name)
            rgb_image = colormap(pattern)  # This returns (H, W, 3)
            rgb_images.append(rgb_image)
            
        # Convert to tensor format (N, C, H, W) - required by TensorBoard
        rgb_images = np.stack(rgb_images)  # (N, H, W, 3)
        
        # Ensure we have exactly 3 channels for RGB
        if rgb_images.shape[-1] == 4:
            rgb_images = rgb_images[..., :3]
        elif rgb_images.shape[-1] != 3:
            raise ValueError(f"Expected 3 or 4 channels, got {rgb_images.shape[-1]}")
            
        rgb_images = torch.from_numpy(rgb_images).permute(0, 3, 1, 2).float()  # (N, 3, H, W)
        
        return rgb_images
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DummyEmbeddingModel(pl.LightningModule):    
    def __init__(self, embedding_dim: int = 64, num_samples: int = 500, num_classes: int = 5, dataset=None):
        super().__init__()        
        self.embedding_dim = embedding_dim
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # Store all data for consistent embedding generation
        self.register_buffer('sample_indices', torch.arange(num_samples))
        
        # Store RGB images and metadata for TensorBoard visualization
        if dataset is not None:
            self.rgb_images = dataset.images  # Shape: (N, C, H, W)
            self.register_buffer('class_labels', dataset.labels)
            self.register_buffer('annotations', dataset.annotations)
        else:
            self.rgb_images = None
            self.register_buffer('class_labels', torch.randint(0, num_classes, (num_samples,)))
            self.register_buffer('annotations', torch.zeros(num_samples, dtype=torch.int))
        
    def generate_embeddings(self, epoch: int, indices: torch.Tensor, labels: torch.Tensor):
        """Generate semi-random embeddings with decreasing noise/randomness based on epoch number and class labels."""
        batch_size = len(indices)
        
        torch.manual_seed(42 + epoch)
        
        # Centers for the distribution of the embeddings for each class
        class_centers = torch.tensor([
            [3.0, 3.0] + [0.1] * (self.embedding_dim - 2),    # Class 0: green circles
            [-3.0, 3.0] + [0.2] * (self.embedding_dim - 2),   # Class 1: magenta h-stripes  
            [3.0, -3.0] + [0.3] * (self.embedding_dim - 2),   # Class 2: cyan v-stripes
            [-3.0, -3.0] + [0.4] * (self.embedding_dim - 2),  # Class 3: orange diagonal
            [0.0, 0.0] + [0.5] * (self.embedding_dim - 2)     # Class 4: blue checkerboard
        ], device=self.device, dtype=torch.float32)
        
        embeddings = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        
        if epoch == 0:
            embeddings = torch.randn(batch_size, self.embedding_dim, device=self.device) * 3.0
            
        elif epoch == 1:
            noise = torch.randn(batch_size, self.embedding_dim, device=self.device) * 2.0
            for i, label in enumerate(labels):
                embeddings[i] = noise[i] + class_centers[label] * 0.1
                
        elif epoch == 2:
            noise = torch.randn(batch_size, self.embedding_dim, device=self.device) * 1.2
            for i, label in enumerate(labels):
                embeddings[i] = noise[i] + class_centers[label] * 0.4
                
        else:
            noise = torch.randn(batch_size, self.embedding_dim, device=self.device) * 0.4
            for i, label in enumerate(labels):
                embeddings[i] = noise[i] + class_centers[label] * 1.0
        
        return embeddings
    
    def forward(self, x):
        """Forward pass - not really used."""
        return torch.zeros(x.size(0), self.embedding_dim, device=self.device)
        
    def training_step(self, batch, batch_idx):
        """Training step - dummy loss."""
        inputs, labels = batch
        
        # Generate embeddings for current epoch
        indices = inputs.long().squeeze()
        embeddings = self.generate_embeddings(self.current_epoch, indices, labels)
        
        # Dummy loss (just to have something to optimize)
        loss = torch.tensor(1.0 / (self.current_epoch + 1), device=self.device, requires_grad=True)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
        
    def on_train_epoch_end(self):
        """Log all embeddings with RGB images to TensorBoard at the end of each epoch."""
        all_embeddings = self.generate_embeddings(
            self.current_epoch, 
            self.sample_indices, 
            self.class_labels
        )
        
        # NOTE: Create metadata as list of lists for TensorBoard columns
        metadata_header = ["Class", "Annotation"]
        metadata = []
        for i in range(len(self.class_labels)):
            class_id = str(self.class_labels[i].item())
            annotation_str = "positive" if self.annotations[i].item() == 1 else "negative"
            # Each row is a list with [class, annotation]
            metadata.append([class_id, annotation_str])
    
        # NOTE: Prepare images for TensorBoard (needs to be in range [0, 1])
        images_for_tb = None
        if self.rgb_images is not None:
            images_for_tb = torch.clamp(self.rgb_images, 0, 1)
        
        tag = f"embeddings_epoch_{self.current_epoch}"
        self.logger.experiment.add_embedding(
            all_embeddings.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            label_img=images_for_tb,
            tag=tag,
            global_step=self.current_epoch
        )
        
    def configure_optimizers(self):
        """Configure dummy optimizer."""
        return torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-3)


def main():    
    pl.seed_everything(42)
    
    train_dataset = DummyDataset(size=500, num_classes=5, image_size=64)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=50, 
        shuffle=False,
        num_workers=0
    )
    
    model = DummyEmbeddingModel(
        embedding_dim=64,
        num_samples=500,
        num_classes=5,
        dataset=train_dataset
    )
    
    tb_logger = TensorBoardLogger(
        save_dir="/home/eduardo.hirata/repos/viscy/viscy/scripts/tb_logs",
        name="embedding_demo",
    )
    
    trainer = pl.Trainer(
        max_epochs=4,
        logger=tb_logger,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=5
    )
    # Train the dummy model with decreasing noise/randomness in the pseudo-embeddings
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()