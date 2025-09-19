import logging

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Dummy dataset with synthetic grayscale images."""
    
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
        
        self.images = self._generate_synthetic_grayscale_images()
        
    def _generate_synthetic_grayscale_images(self):
        """Generate synthetic grayscale images with class-specific patterns."""
        grayscale_images = []
        
        for i in range(self.size):
            class_id = self.labels[i].item()
            
            pattern = np.zeros((self.image_size, self.image_size))
            
            # Class-specific geometric pattern with different intensities
            if class_id == 0:  # Circles - bright
                y, x = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size), indexing='ij')
                center = self.image_size // 2
                radius = self.image_size // 4
                pattern = ((x - center) ** 2 + (y - center) ** 2 <= radius ** 2).astype(float) * 0.8
                
            elif class_id == 1:  # Horizontal stripes - medium bright
                for j in range(0, self.image_size, 8):
                    pattern[j:j+4, :] = 0.6
                    
            elif class_id == 2:  # Vertical stripes - medium
                for j in range(0, self.image_size, 8):
                    pattern[:, j:j+4] = 0.5
                    
            elif class_id == 3:  # Diagonal pattern - medium dark
                y, x = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size), indexing='ij')
                pattern = ((x + y) % 10 < 5).astype(float) * 0.4
                
            else:  # class_id == 4: Checkerboard - dark
                y, x = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size), indexing='ij')
                pattern = ((x // 8 + y // 8) % 2).astype(float) * 0.3
            
            # Add some Gaussian blobs
            y, x = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size), indexing='ij')
            blob_centers = np.random.RandomState(i).rand(2, 2) * self.image_size
            for center in blob_centers:
                cx, cy = center
                sigma = self.image_size // 6
                gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
                pattern += gaussian * 0.2
            
            # Normalize and add noise
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
            noise = np.random.RandomState(i).randn(*pattern.shape) * 0.05
            pattern = np.clip(pattern + noise, 0, 1)
            
            grayscale_images.append(pattern)
            
        #NOTE: Convert to tensor format (N, 1, H, W) - single channel grayscale
        grayscale_images = np.stack(grayscale_images)  # (N, H, W)
        grayscale_images = torch.from_numpy(grayscale_images).unsqueeze(1).float()  # (N, 1, H, W)
        
        return grayscale_images
        
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

        self.sample_indices = torch.arange(num_samples)
        
        if dataset is not None:
            self.grayscale_images = dataset.images  # Shape: (N, 1, H, W)
            self.class_labels = dataset.labels
            self.annotations = dataset.annotations
        else:
            self.grayscale_images = None
            self.class_labels = torch.randint(0, num_classes, (num_samples,))
            self.annotations = torch.zeros(num_samples, dtype=torch.int)
        
    def generate_embeddings(self, epoch: int, indices: torch.Tensor, labels: torch.Tensor):
        """Generate semi-random embeddings with decreasing noise/randomness based on epoch number and class labels."""
        batch_size = len(indices)
        
        torch.manual_seed(42 + epoch)
        
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
        
        indices = inputs.long().squeeze()
        embeddings = self.generate_embeddings(self.current_epoch, indices, labels)
        
        # Dummy loss (just to have something to optimize)
        loss = torch.tensor(1.0 / (self.current_epoch + 1), device=self.device, requires_grad=True)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
        
    def on_train_epoch_end(self):
        """Log all embeddings with RGB images to TensorBoard at the end of each epoch."""
        # Move tensors to device for computation
        sample_indices = self.sample_indices.to(self.device)
        class_labels = self.class_labels.to(self.device)
        annotations = self.annotations.to(self.device)
        
        all_embeddings = self.generate_embeddings(
            self.current_epoch, 
            sample_indices, 
            class_labels
        )
        
        all_embeddings_cpu = all_embeddings.detach().cpu()
        class_labels_cpu = class_labels.detach().cpu()
        annotations_cpu = annotations.detach().cpu()
        
        # NOTE: Create metadata as list of lists for TensorBoard columns
        metadata_header = ["Class", "Annotation"]
        metadata = []
        for i in range(len(class_labels_cpu)):
            class_id = str(class_labels_cpu[i].item())
            annotation_str = "positive" if annotations_cpu[i].item() == 1 else "negative"
            # Each row is a list with [class, annotation]
            metadata.append([class_id, annotation_str])
    
        # NOTE: Prepare images for TensorBoard (needs to be in range [0, 1])
        images_for_tb = None
        if self.grayscale_images is not None:
            images_for_tb = torch.clamp(self.grayscale_images, 0, 1)
        
        tag = f"embeddings_epoch_{self.current_epoch}"
        self.logger.experiment.add_embedding(
            all_embeddings_cpu,
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
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()