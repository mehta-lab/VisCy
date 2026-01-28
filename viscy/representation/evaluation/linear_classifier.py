"""Core functions for training and applying linear classifiers on embeddings."""

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

import anndata as ad
import joblib
import numpy as np
import wandb
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from viscy.representation.evaluation import load_annotation_anndata


class LinearClassifierPipeline:
    """Encapsulates trained classifier with preprocessing transformations.

    Parameters
    ----------
    classifier : LogisticRegression
        Trained logistic regression classifier.
    scaler : Optional[StandardScaler]
        Fitted StandardScaler, if feature scaling was used.
    pca : Optional[PCA]
        Fitted PCA transformer, if dimensionality reduction was used.
    config : dict
        Configuration used for training.
    task : str
        Name of the classification task.
    """

    def __init__(
        self,
        classifier: LogisticRegression,
        scaler: Optional[StandardScaler],
        pca: Optional[PCA],
        config: dict,
        task: str,
    ):
        self.classifier = classifier
        self.scaler = scaler
        self.pca = pca
        self.config = config
        self.task = task

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply preprocessing transformations to features.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed features.
        """
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for features.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        X_transformed = self.transform(X)
        return self.classifier.predict(X_transformed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for features.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class probabilities of shape (n_samples, n_classes).
        """
        X_transformed = self.transform(X)
        return self.classifier.predict_proba(X_transformed)


def load_and_combine_datasets(datasets: list[dict], task: str) -> ad.AnnData:
    """Load and combine multiple datasets with embeddings and annotations.

    Parameters
    ----------
    datasets : list[dict]
        List of dataset dicts with 'embeddings' and 'annotations' paths.
    task : str
        Name of the classification task (column name in annotations).

    Returns
    -------
    ad.AnnData
        Combined AnnData object with embeddings and task annotations.

    Raises
    ------
    ValueError
        If no valid training data is loaded after processing all datasets.
    """
    train_data_list = []

    for i, dataset in enumerate(datasets):
        embeddings_path = Path(dataset["embeddings"])
        annotations_path = Path(dataset["annotations"])

        print(f"\nLoading dataset {i + 1}/{len(datasets)}: {embeddings_path.name}")
        print(f"  Embeddings: {embeddings_path}")
        print(f"  Annotations: {annotations_path}")

        adata = ad.read_zarr(embeddings_path)

        try:
            adata_annotated = load_annotation_anndata(
                adata, str(annotations_path), task
            )
        except KeyError as e:
            print(f"⚠ Skipping dataset - task '{task}' not found in annotations:")
            print(f"  Error: {e}")
            continue

        if task not in adata_annotated.obs.columns:
            print(f"⚠ Skipping dataset - task '{task}' not in columns:")
            print(f"  Available: {list(adata_annotated.obs.columns)}")
            continue

        adata_filtered = adata_annotated[adata_annotated.obs[task] != "unknown"]
        adata_filtered = adata_filtered[adata_filtered.obs[task].notna()]

        if len(adata_filtered) == 0:
            print("⚠ Skipping dataset - no valid samples after filtering")
            continue

        print(f"  ✓ Loaded {adata_filtered.shape[0]} samples")
        print(f"  Class distribution:\n{adata_filtered.obs[task].value_counts()}")
        train_data_list.append(adata_filtered)

    if len(train_data_list) == 0:
        raise ValueError("No training data loaded from any dataset!")

    if len(train_data_list) == 1:
        combined = train_data_list[0]
    else:
        combined = ad.concat(train_data_list, join="outer")

    print(f"\n{'=' * 60}")
    print(f"Total training samples: {combined.shape[0]}")
    print(f"Overall class distribution:\n{combined.obs[task].value_counts()}")
    print("=" * 60)

    return combined


def train_linear_classifier(
    adata: ad.AnnData,
    task: str,
    use_scaling: bool = True,
    use_pca: bool = False,
    n_pca_components: Optional[int] = None,
    classifier_params: Optional[dict[str, Any]] = None,
    split_train_data: float = 0.8,
    random_seed: int = 42,
) -> tuple[LinearClassifierPipeline, dict[str, float]]:
    """Train a linear classifier on embeddings with preprocessing and evaluation.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing embeddings in .X and labels in .obs[task].
    task : str
        Name of the classification task (column in .obs).
    use_scaling : bool
        Whether to apply StandardScaler normalization.
    use_pca : bool
        Whether to apply PCA dimensionality reduction.
    n_pca_components : Optional[int]
        Number of PCA components (required if use_pca=True).
    classifier_params : Optional[dict]
        Parameters for LogisticRegression classifier.
    split_train_data : float
        Fraction of data to use for training (rest for validation).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    LinearClassifierPipeline
        Trained classifier pipeline with preprocessing.
    dict
        Dictionary of evaluation metrics (train and validation if split).
    """
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFIER")
    print("=" * 60)

    if classifier_params is None:
        classifier_params = {}

    X_full = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    y_full = adata.obs[task].values

    scaler = None
    pca = None

    if use_scaling:
        scaler = StandardScaler()
        X_full_scaled = scaler.fit_transform(X_full)
        print("\n✓ Features scaled with StandardScaler")
    else:
        X_full_scaled = X_full
        print("\n✓ Using raw embeddings (no scaling)")

    if use_pca:
        pca = PCA(n_components=n_pca_components)
        X_full_transformed = pca.fit_transform(X_full_scaled)
        print(f"\n✓ PCA applied with {n_pca_components} components")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        X_full_transformed = X_full_scaled
        print("\n✓ Using full feature space (no PCA)")

    if split_train_data < 1.0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_full_transformed,
            y_full,
            train_size=split_train_data,
            random_state=random_seed,
            stratify=y_full,
            shuffle=True,
        )
        print(f"\n✓ Split data: train ({len(X_train)}) / validation ({len(X_val)})")
    else:
        X_train = X_full_transformed
        y_train = y_full
        X_val = None
        y_val = None
        print("\n✓ Using all data for training (no split)")

    classifier = LogisticRegression(**classifier_params)
    classifier.fit(X_train, y_train)
    print("✓ Classifier trained")

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    y_train_pred = classifier.predict(X_train)
    train_report = classification_report(
        y_train, y_train_pred, digits=3, output_dict=True
    )
    print("\nTraining Set:")
    print(classification_report(y_train, y_train_pred, digits=3))

    train_metrics = {
        "train_accuracy": train_report["accuracy"],
        "train_weighted_precision": train_report["weighted avg"]["precision"],
        "train_weighted_recall": train_report["weighted avg"]["recall"],
        "train_weighted_f1": train_report["weighted avg"]["f1-score"],
    }

    for class_name in classifier.classes_:
        if class_name in train_report:
            train_metrics[f"train_{class_name}_precision"] = train_report[class_name][
                "precision"
            ]
            train_metrics[f"train_{class_name}_recall"] = train_report[class_name][
                "recall"
            ]
            train_metrics[f"train_{class_name}_f1"] = train_report[class_name][
                "f1-score"
            ]

    val_metrics = {}
    if X_val is not None and y_val is not None:
        y_val_pred = classifier.predict(X_val)
        val_report = classification_report(
            y_val, y_val_pred, digits=3, output_dict=True
        )
        print("\nValidation Set:")
        print(classification_report(y_val, y_val_pred, digits=3))

        val_metrics = {
            "val_accuracy": val_report["accuracy"],
            "val_weighted_precision": val_report["weighted avg"]["precision"],
            "val_weighted_recall": val_report["weighted avg"]["recall"],
            "val_weighted_f1": val_report["weighted avg"]["f1-score"],
        }

        for class_name in classifier.classes_:
            if class_name in val_report:
                val_metrics[f"val_{class_name}_precision"] = val_report[class_name][
                    "precision"
                ]
                val_metrics[f"val_{class_name}_recall"] = val_report[class_name][
                    "recall"
                ]
                val_metrics[f"val_{class_name}_f1"] = val_report[class_name]["f1-score"]

    all_metrics = {**train_metrics, **val_metrics}

    config_dict = {
        "task": task,
        "use_scaling": use_scaling,
        "use_pca": use_pca,
        "n_pca_components": n_pca_components,
        "classifier_params": classifier_params,
        "split_train_data": split_train_data,
        "random_seed": random_seed,
    }

    pipeline = LinearClassifierPipeline(
        classifier=classifier,
        scaler=scaler,
        pca=pca,
        config=config_dict,
        task=task,
    )

    return pipeline, all_metrics


def predict_with_classifier(
    adata: ad.AnnData,
    pipeline: LinearClassifierPipeline,
    task: str,
) -> ad.AnnData:
    """Apply trained classifier to make predictions on new data.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing embeddings in .X.
    pipeline : LinearClassifierPipeline
        Trained classifier pipeline with preprocessing.
    task : str
        Name of the classification task.

    Returns
    -------
    ad.AnnData
        AnnData with predictions added to .obs[f"predicted_{task}"],
        probabilities in .obsm[f"predicted_{task}_proba"],
        and class labels in .uns[f"predicted_{task}_classes"].
    """
    print("\nApplying preprocessing and making predictions...")
    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()

    predictions = pipeline.predict(X)
    prediction_proba = pipeline.predict_proba(X)

    adata.obs[f"predicted_{task}"] = predictions
    adata.obsm[f"predicted_{task}_proba"] = prediction_proba
    adata.uns[f"predicted_{task}_classes"] = pipeline.classifier.classes_.tolist()

    print("✓ Predictions complete")
    print("  Predicted class distribution:")
    print(adata.obs[f"predicted_{task}"].value_counts())
    print(f"  Probability matrix shape: {prediction_proba.shape}")
    print(f"  Classes: {pipeline.classifier.classes_.tolist()}")

    return adata


def save_pipeline_to_wandb(
    pipeline: LinearClassifierPipeline,
    metrics: dict[str, float],
    config: dict[str, Any],
    wandb_project: str,
    wandb_entity: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
    """Save trained pipeline and metrics to Weights & Biases.

    Parameters
    ----------
    pipeline : LinearClassifierPipeline
        Trained classifier pipeline.
    metrics : dict
        Dictionary of evaluation metrics.
    config : dict
        Full training configuration.
    wandb_project : str
        W&B project name.
    wandb_entity : Optional[str]
        W&B entity (username or team).
    tags : Optional[list[str]]
        Tags to add to the run.

    Returns
    -------
    str
        Name of the created W&B artifact.
    """
    print("\n" + "=" * 60)
    print("SAVING MODEL AND LOGGING TO WANDB")
    print("=" * 60)

    task = config["task"]
    input_channel = config["input_channel"]
    use_pca = config.get("preprocessing", {}).get("use_pca", False)
    n_pca = config.get("preprocessing", {}).get("n_pca_components")

    model_name = f"linear-classifier-{task}-{input_channel}"
    if use_pca:
        model_name += f"-pca{n_pca}"

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        job_type=f"linear-classifier-{task}-{input_channel}",
        name=model_name,
        config=config,
        tags=tags or [],
    )

    wandb.log(metrics)
    print("\n✓ Logged metrics to wandb:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.3f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        model_filename = tmpdir_path / f"{model_name}.joblib"
        joblib.dump(pipeline.classifier, model_filename)

        config_filename = tmpdir_path / f"{model_name}_config.json"
        with open(config_filename, "w") as f:
            json.dump(config, f, indent=2)

        artifact = wandb.Artifact(model_name, type="model")
        artifact.add_file(str(model_filename))
        artifact.add_file(str(config_filename))

        if pipeline.scaler is not None:
            scaler_filename = tmpdir_path / f"{model_name}_scaler.joblib"
            joblib.dump(pipeline.scaler, scaler_filename)
            artifact.add_file(str(scaler_filename))
            print("✓ Scaler saved to artifact")

        if pipeline.pca is not None:
            pca_filename = tmpdir_path / f"{model_name}_pca.joblib"
            joblib.dump(pipeline.pca, pca_filename)
            artifact.add_file(str(pca_filename))
            print("✓ PCA saved to artifact")

        run.log_artifact(artifact)

    run.finish()

    print(f"✓ Model logged to wandb: {model_name}")
    print("=" * 60)

    return model_name


def load_pipeline_from_wandb(
    wandb_project: str,
    model_name: str,
    version: str = "latest",
    wandb_entity: Optional[str] = None,
) -> tuple[LinearClassifierPipeline, dict]:
    """Load trained pipeline and config from Weights & Biases.

    Parameters
    ----------
    wandb_project : str
        W&B project name.
    model_name : str
        Name of the model artifact.
    version : str
        Version of the artifact (default: 'latest').
    wandb_entity : Optional[str]
        W&B entity (username or team).

    Returns
    -------
    LinearClassifierPipeline
        Loaded classifier pipeline.
    dict
        Configuration used for training.
    """
    print("\n" + "=" * 60)
    print("LOADING MODEL FROM WANDB")
    print("=" * 60)

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        job_type="inference",
    )

    artifact = run.use_artifact(f"{model_name}:{version}")
    artifact_dir = Path(artifact.download())

    config_path = artifact_dir / f"{model_name}_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"✓ Loaded config: {config_path.name}")
    print(f"  Task: {config['task']}")
    print(f"  Input channel: {config.get('input_channel', 'N/A')}")

    model_path = artifact_dir / f"{model_name}.joblib"
    classifier = joblib.load(model_path)
    print(f"✓ Loaded classifier: {model_path.name}")

    scaler = None
    scaler_path = artifact_dir / f"{model_name}_scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"✓ Loaded scaler: {scaler_path.name}")

    pca = None
    pca_path = artifact_dir / f"{model_name}_pca.joblib"
    if pca_path.exists():
        pca = joblib.load(pca_path)
        print(f"✓ Loaded PCA: {pca_path.name}")

    print("=" * 60)

    pipeline = LinearClassifierPipeline(
        classifier=classifier,
        scaler=scaler,
        pca=pca,
        config=config,
        task=config["task"],
    )

    run.finish()

    return pipeline, config
