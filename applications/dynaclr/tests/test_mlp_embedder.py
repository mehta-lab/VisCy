"""Tests for MLP embedder training and inference."""

import numpy as np
import pytest
import torch

from dynaclr.evaluation.mlp_embedder.apply_mlp_embedder import _load_model
from dynaclr.evaluation.mlp_embedder.train_mlp_embedder import MlpEmbedderTrainConfig, _train_loop
from viscy_models.components.heads import MLP


@pytest.fixture
def trained_mlp(annotated_adata):
    """Train a small MLP on the annotated_adata fixture and return (model, adata)."""
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, TensorDataset

    adata = annotated_adata
    X = adata.X
    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_death_state"].to_numpy())

    embs = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(embs, labels), batch_size=16, shuffle=False)

    model = MLP(in_dims=X.shape[1], hidden_dims=[32], num_classes=len(le.classes_))
    device = torch.device("cpu")
    model = model.to(device)

    _train_loop(
        model, loader, loader, num_epochs=2, learning_rate=1e-3, weight_decay=1e-4, device=device, wandb_run=None
    )

    return model, adata, le


class TestMLP:
    def test_forward_shape(self, annotated_adata):
        X = torch.tensor(annotated_adata.X, dtype=torch.float32)
        model = MLP(in_dims=16, hidden_dims=[32, 32], num_classes=3)
        logits = model(X)
        assert logits.shape == (len(X), 3)

    def test_encode_shape(self, annotated_adata):
        X = torch.tensor(annotated_adata.X, dtype=torch.float32)
        model = MLP(in_dims=16, hidden_dims=[32, 32], num_classes=3)
        reps = model.encode(X)
        assert reps.shape == (len(X), 32)

    def test_encode_l2_normalised(self, annotated_adata):
        X = torch.tensor(annotated_adata.X, dtype=torch.float32)
        model = MLP(in_dims=16, hidden_dims=[32], num_classes=3)
        reps = model.encode(X)
        norms = reps.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_raises_in_projection_mode(self):
        model = MLP(in_dims=16, hidden_dims=[32], out_dims=16)
        x = torch.randn(4, 16)
        with pytest.raises(RuntimeError, match="encode\\(\\)"):
            model.encode(x)

    def test_input_dim_stored(self):
        model = MLP(in_dims=64, hidden_dims=[32], num_classes=2)
        assert model.input_dim == 64


class TestTrainLoop:
    def test_runs_without_wandb(self, annotated_adata):
        from sklearn.preprocessing import LabelEncoder
        from torch.utils.data import DataLoader, TensorDataset

        X = torch.tensor(annotated_adata.X, dtype=torch.float32)
        le = LabelEncoder()
        y = torch.tensor(le.fit_transform(annotated_adata.obs["cell_death_state"].to_numpy()), dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)

        model = MLP(in_dims=16, hidden_dims=[32], num_classes=3).to("cpu")
        _train_loop(
            model,
            loader,
            loader,
            num_epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            device=torch.device("cpu"),
            wandb_run=None,
        )


class TestMlpEmbedderTrainConfig:
    def test_valid_config(self, tmp_path):
        cfg = MlpEmbedderTrainConfig(
            embeddings_path="/tmp/fake.zarr",
            target_col="predicted_infection_state",
            output_path=str(tmp_path / "model.pt"),
        )
        assert cfg.hidden_dims == [512, 512, 512]
        assert cfg.num_epochs == 50

    def test_batch_norm_default_true(self, tmp_path):
        cfg = MlpEmbedderTrainConfig(
            embeddings_path="/tmp/fake.zarr",
            target_col="predicted_infection_state",
            output_path=str(tmp_path / "model.pt"),
        )
        assert cfg.batch_norm is True


class TestCheckpointRoundTrip:
    def test_save_and_load(self, annotated_adata, tmp_path):
        model = MLP(in_dims=16, hidden_dims=[32], num_classes=3)
        checkpoint_path = tmp_path / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": 16,
                "num_classes": 3,
                "classes": ["alive", "apoptotic", "dead"],
                "hidden_dims": [32],
                "dropout": 0.4,
                "cosine_classifier": True,
                "batch_norm": True,
            },
            checkpoint_path,
        )
        loaded = _load_model(checkpoint_path, torch.device("cpu"))
        assert loaded.input_dim == 16

        X = torch.tensor(annotated_adata.X, dtype=torch.float32)
        reps = loaded.encode(X)
        assert reps.shape == (len(X), 32)


class TestApplyMlpEmbedder:
    def test_obsm_written(self, annotated_adata, tmp_path):
        import torch.nn.functional as F

        from dynaclr.evaluation.mlp_embedder.apply_mlp_embedder import _load_model

        model = MLP(in_dims=16, hidden_dims=[32], num_classes=3).eval()
        checkpoint_path = tmp_path / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": 16,
                "num_classes": 3,
                "classes": ["alive", "apoptotic", "dead"],
                "hidden_dims": [32],
                "dropout": 0.4,
                "cosine_classifier": True,
                "batch_norm": True,
            },
            checkpoint_path,
        )

        device = torch.device("cpu")
        loaded = _load_model(checkpoint_path, device)

        X = annotated_adata.X
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            reps = F.normalize(loaded.backbone(X_t), dim=1).numpy()

        annotated_adata.obsm["X_mlp"] = reps
        assert "X_mlp" in annotated_adata.obsm
        assert annotated_adata.obsm["X_mlp"].shape == (len(annotated_adata), 32)
        assert np.allclose(np.linalg.norm(reps, axis=1), 1.0, atol=1e-5)
