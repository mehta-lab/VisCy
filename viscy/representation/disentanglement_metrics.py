import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

_logger = logging.getLogger(__name__)


class DisentanglementMetrics:
    """
    Disentanglement metrics for VAE evaluation on microscopy data.

    Implements MIG, SAP, DCI, and Beta-VAE score metrics for evaluating
    how well the VAE learns disentangled representations.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def compute_all_metrics(
        self,
        vae_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_samples: int = 1000,
        n_factors: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute all disentanglement metrics.

        Args:
            vae_model: Trained VAE model
            dataloader: DataLoader with labeled data
            max_samples: Maximum number of samples to use
            n_factors: Number of known generative factors (if available)

        Returns:
            Dictionary of metric scores
        """
        latents, factors = self._extract_latents_and_factors(
            vae_model, dataloader, max_samples
        )

        metrics = {}

        # MIG Score
        metrics["MIG"] = self.compute_mig(latents, factors)

        # SAP Score
        metrics["SAP"] = self.compute_sap(latents, factors)

        # DCI Scores
        dci_scores = self.compute_dci(latents, factors)
        metrics.update(dci_scores)

        # Beta-VAE Score (unsupervised)
        metrics["Beta_VAE_Score"] = self.compute_beta_vae_score(
            vae_model, dataloader, max_samples
        )

        return metrics

    def _extract_latents_and_factors(
        self,
        vae_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract latent representations and generative factors.

        For microscopy data, we'll extract simple visual factors like:
        - Cell size (approximated from pixel intensity)
        - Cell count (approximated from connected components)
        - Brightness (mean intensity)
        - Contrast (std of intensity)
        """
        vae_model.eval()
        latents = []
        factors = []

        samples_collected = 0

        with torch.no_grad():
            for batch in dataloader:
                if samples_collected >= max_samples:
                    break

                x = batch["anchor"].to(self.device)
                batch_size = x.shape[0]

                # Extract latent representations
                model_output = vae_model(x)
                z = (
                    model_output.z
                    if hasattr(model_output, "z")
                    else model_output.embedding
                )
                latents.append(z.cpu().numpy())

                # Extract visual factors from images
                batch_factors = self._extract_visual_factors(x.cpu())
                factors.append(batch_factors)

                samples_collected += batch_size

        latents = np.vstack(latents)[:max_samples]
        factors = np.vstack(factors)[:max_samples]

        return latents, factors

    def _extract_visual_factors(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract visual factors from microscopy images.

        Args:
            images: Batch of images (B, C, D, H, W)

        Returns:
            Array of shape (B, n_factors) with extracted factors
        """
        batch_size = images.shape[0]
        factors = []

        for i in range(batch_size):
            img = images[i].numpy()  # (C, D, H, W)

            # Take middle z-slice for 2D analysis
            mid_z = img.shape[1] // 2
            img_2d = img[:, mid_z, :, :]  # (C, H, W)

            # Factor 1: Brightness (mean intensity)
            brightness = np.mean(img_2d)

            # Factor 2: Contrast (std of intensity)
            contrast = np.std(img_2d)

            # Factor 3: Cell size (approximated by high-intensity regions)
            binary_mask = img_2d[0] > np.percentile(img_2d[0], 75)
            cell_size = np.sum(binary_mask) / (img_2d.shape[1] * img_2d.shape[2])

            # Factor 4: Texture complexity (gradient magnitude)
            grad_x = np.gradient(img_2d[0], axis=1)
            grad_y = np.gradient(img_2d[0], axis=0)
            texture = np.mean(np.sqrt(grad_x**2 + grad_y**2))

            factors.append([brightness, contrast, cell_size, texture])

        return np.array(factors)

    def compute_mig(self, latents: np.ndarray, factors: np.ndarray) -> float:
        """
        Compute Mutual Information Gap (MIG).

        MIG = (1/K) * Σ_k (I(z_j*; v_k) - I(z_j'; v_k))
        where j* = argmax_j I(z_j; v_k) and j' = argmax_{j≠j*} I(z_j; v_k)
        """

        def mutual_info_continuous(x, y):
            """Estimate mutual information between continuous variables."""
            # Discretize continuous variables
            x_discrete = self._discretize(x)
            y_discrete = self._discretize(y)

            # Compute mutual information
            return self._mutual_info_discrete(x_discrete, y_discrete)

        n_factors = factors.shape[1]
        n_latents = latents.shape[1]

        # Compute mutual information matrix
        mi_matrix = np.zeros((n_latents, n_factors))

        for i in range(n_latents):
            for j in range(n_factors):
                mi_matrix[i, j] = mutual_info_continuous(latents[:, i], factors[:, j])

        # Compute MIG
        mig = 0.0
        for j in range(n_factors):
            mi_values = mi_matrix[:, j]
            sorted_indices = np.argsort(mi_values)[::-1]

            if len(sorted_indices) > 1:
                gap = mi_values[sorted_indices[0]] - mi_values[sorted_indices[1]]
                mig += gap / np.max(mi_values) if np.max(mi_values) > 0 else 0

        return mig / n_factors

    def compute_sap(self, latents: np.ndarray, factors: np.ndarray) -> float:
        """
        Compute Attribute Predictability Score (SAP).

        SAP measures how well a simple classifier can predict factors from latents.
        """
        n_factors = factors.shape[1]
        scores = []

        for i in range(n_factors):
            # Discretize factor for classification
            factor_discrete = self._discretize(factors[:, i], n_bins=10)

            # Train classifier
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(latents, factor_discrete)

            # Evaluate
            pred = clf.predict(latents)
            score = accuracy_score(factor_discrete, pred)
            scores.append(score)

        return np.mean(scores)

    def compute_dci(self, latents: np.ndarray, factors: np.ndarray) -> Dict[str, float]:
        """
        Compute Disentanglement, Completeness, and Informativeness (DCI).
        """
        n_factors = factors.shape[1]
        n_latents = latents.shape[1]

        # Train predictors for each factor
        importance_matrix = np.zeros((n_factors, n_latents))

        for i in range(n_factors):
            # Discretize factor
            factor_discrete = self._discretize(factors[:, i], n_bins=10)

            # Train random forest to get feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(latents, factor_discrete)

            importance_matrix[i, :] = rf.feature_importances_

        # Normalize importance matrix
        importance_matrix = importance_matrix / (
            np.sum(importance_matrix, axis=1, keepdims=True) + 1e-8
        )

        # Compute DCI metrics
        disentanglement = self._compute_disentanglement(importance_matrix)
        completeness = self._compute_completeness(importance_matrix)
        informativeness = self._compute_informativeness(importance_matrix)

        return {
            "DCI_Disentanglement": disentanglement,
            "DCI_Completeness": completeness,
            "DCI_Informativeness": informativeness,
        }

    def compute_beta_vae_score(
        self,
        vae_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_samples: int,
    ) -> float:
        """
        Compute Beta-VAE score (unsupervised disentanglement metric).

        Measures how well individual latent dimensions affect reconstruction
        when perturbed independently.
        """
        vae_model.eval()
        scores = []

        samples_collected = 0

        with torch.no_grad():
            for batch in dataloader:
                if samples_collected >= max_samples:
                    break

                x = batch["anchor"].to(self.device)
                batch_size = x.shape[0]

                # Get latent representation
                model_output = vae_model(x)
                z = (
                    model_output.z
                    if hasattr(model_output, "z")
                    else model_output.embedding
                )

                # Compute baseline reconstruction
                baseline_recon = vae_model.decoder(z)
                if hasattr(baseline_recon, "reconstruction"):
                    baseline_recon = baseline_recon.reconstruction

                # Perturb each latent dimension
                for dim in range(z.shape[1]):
                    z_perturbed = z.clone()
                    z_perturbed[:, dim] += torch.randn_like(z_perturbed[:, dim]) * 0.5

                    # Get perturbed reconstruction
                    perturbed_recon = vae_model.decoder(z_perturbed)
                    if hasattr(perturbed_recon, "reconstruction"):
                        perturbed_recon = perturbed_recon.reconstruction

                    # Compute reconstruction difference
                    diff = F.mse_loss(baseline_recon, perturbed_recon, reduction="none")
                    diff = diff.mean(
                        dim=(1, 2, 3, 4)
                    )  # Average over spatial dimensions

                    # Score is inverse of reconstruction change
                    score = 1.0 / (1.0 + diff.mean().item())
                    scores.append(score)

                samples_collected += batch_size

        return np.mean(scores)

    def _discretize(self, x: np.ndarray, n_bins: int = 20) -> np.ndarray:
        """Discretize continuous variable into bins."""
        return np.digitize(x, np.linspace(x.min(), x.max(), n_bins))

    def _mutual_info_discrete(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between discrete variables."""
        # Joint histogram
        xy = np.stack([x, y], axis=1)
        unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
        p_xy = counts_xy / counts_xy.sum()

        # Marginal histograms
        unique_x, counts_x = np.unique(x, return_counts=True)
        p_x = counts_x / counts_x.sum()

        unique_y, counts_y = np.unique(y, return_counts=True)
        p_y = counts_y / counts_y.sum()

        # Compute MI
        mi = 0.0
        for i, (x_val, y_val) in enumerate(unique_xy):
            p_joint = p_xy[i]
            p_x_marginal = p_x[unique_x == x_val][0]
            p_y_marginal = p_y[unique_y == y_val][0]

            if p_joint > 0 and p_x_marginal > 0 and p_y_marginal > 0:
                mi += p_joint * np.log(p_joint / (p_x_marginal * p_y_marginal))

        return mi

    def _compute_disentanglement(self, importance_matrix: np.ndarray) -> float:
        """Compute disentanglement score from importance matrix."""
        disentanglement = 0.0
        for i in range(importance_matrix.shape[0]):
            if np.sum(importance_matrix[i]) > 0:
                disentanglement += 1.0 - stats.entropy(importance_matrix[i])
        return disentanglement / importance_matrix.shape[0]

    def _compute_completeness(self, importance_matrix: np.ndarray) -> float:
        """Compute completeness score from importance matrix."""
        completeness = 0.0
        for j in range(importance_matrix.shape[1]):
            if np.sum(importance_matrix[:, j]) > 0:
                completeness += 1.0 - stats.entropy(importance_matrix[:, j])
        return completeness / importance_matrix.shape[1]

    def _compute_informativeness(self, importance_matrix: np.ndarray) -> float:
        """Compute informativeness score from importance matrix."""
        return np.mean(np.sum(importance_matrix, axis=1))
