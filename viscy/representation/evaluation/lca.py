"""Linear probing of trained encoder based on cell state labels."""

from typing import Mapping

import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, Occlusion
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from xarray import DataArray

from viscy.representation.contrastive import ContrastiveEncoder


def fit_logistic_regression(
    features: DataArray,
    annotations: pd.Series,
    train_fovs: list[str] | None = None,
    train_ratio: float = 0.8,
    remove_background_class: bool = True,
    scale_features: bool = False,
    class_weight: Mapping | str | None = "balanced",
    random_state: int | None = None,
    solver="liblinear",
) -> tuple[
    LogisticRegression,
    tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]],
]:
    """Fit a binary logistic regression classifier.

    Parameters
    ----------
    features : DataArray
        Xarray of features.
    annotations : pd.Series
        Categorical class annotations with label values starting from 0.
        Must have 3 classes (when remove background is True) or 2 classes.
    train_fovs : list[str] | None, optional
        List of FOVs to use for training. The rest will be used for testing.
        If None, uses stratified sampling based on train_ratio.
    train_ratio : float, optional
        Proportion of samples to use for training (0.0 to 1.0).
        Used when train_fovs is None.
        Uses stratified sampling to ensure balanced class representation.
        Default is 0.8 (80% training, 20% testing).
    remove_background_class : bool, optional
        Remove background class (0), by default True
    scale_features : bool, optional
        Scale features, by default False
    class_weight : Mapping | str | None, optional
        Class weight for balancing, by default "balanced"
    random_state : int | None, optional
        Random state or seed, by default None
    solver : str, optional
        Solver for the regression problem, by default "liblinear"

    Returns
    -------
    tuple[LogisticRegression, tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]]
        Trained classifier and data split [[X_train, y_train], [X_test, y_test]].
    """
    annotations = annotations.cat.codes.values.copy()

    # Handle background class removal before splitting for stratification
    if remove_background_class:
        valid_indices = annotations != 0
        features_filtered = features[valid_indices]
        annotations_filtered = annotations[valid_indices] - 1
    else:
        features_filtered = features
        annotations_filtered = annotations

    # Determine train FOVs
    if train_fovs is None:
        unique_fovs = features_filtered["fov_name"].unique()

        fov_class_dist = []
        for fov in unique_fovs:
            fov_mask = features_filtered["fov_name"] == fov
            fov_classes = annotations_filtered[fov_mask]
            # Use majority class for stratification or class distribution
            majority_class = pd.Series(fov_classes).mode()[0]
            fov_class_dist.append(majority_class)

        # Split FOVs, not individual samples
        train_fovs, test_fovs = train_test_split(
            unique_fovs,
            test_size=1 - train_ratio,
            stratify=fov_class_dist,
            random_state=random_state,
        )

    # Create train/test selections
    train_selection = features_filtered["fov_name"].isin(train_fovs)
    test_selection = ~train_selection
    train_features = features_filtered.values[train_selection]
    test_features = features_filtered.values[test_selection]
    train_annotations = annotations_filtered[train_selection]
    test_annotations = annotations_filtered[test_selection]

    if scale_features:
        train_features = StandardScaler().fit_transform(train_features)
        test_features = StandardScaler().fit_transform(test_features)
    logistic_regression = LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        solver=solver,
    )
    logistic_regression.fit(train_features, train_annotations)
    prediction = logistic_regression.predict(test_features)
    print("Trained logistic regression classifier.")
    print(
        "Training set accuracy:\n"
        + classification_report(
            logistic_regression.predict(train_features), train_annotations, digits=3
        )
    )
    print(
        "Test set accuracy:\n"
        + classification_report(prediction, test_annotations, digits=3)
    )
    return logistic_regression, (
        (train_features, train_annotations),
        (test_features, test_annotations),
    )


def linear_from_binary_logistic_regression(
    logistic_regression: LogisticRegression,
) -> nn.Linear:
    """Convert a binary logistic regression model to a ``torch.nn.Linear`` layer.

    Parameters
    ----------
    logistic_regression : LogisticRegression
        Trained logistic regression model.

    Returns
    -------
    nn.Linear
        Converted linear model.
    """
    weights = torch.from_numpy(logistic_regression.coef_).float()
    bias = torch.from_numpy(logistic_regression.intercept_).float()
    model = nn.Linear(in_features=weights.shape[1], out_features=1)
    model.weight.data = weights
    model.bias.data = bias
    model.eval()
    return model


class AssembledClassifier(torch.nn.Module):
    """Assemble a contrastive encoder with a linear classifier.

    Parameters
    ----------
    backbone : ContrastiveEncoder
        Encoder backbone.
    classifier : nn.Linear
        Classifier head.
    """

    def __init__(self, backbone: ContrastiveEncoder, classifier: nn.Linear) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    @staticmethod
    def scale_features(x: Tensor) -> Tensor:
        m = x.mean(-2, keepdim=True)
        s = x.std(-2, unbiased=False, keepdim=True)
        return (x - m) / s

    def forward(self, x: Tensor, scale_features: bool = False) -> Tensor:
        x = self.backbone.stem(x)
        x = self.backbone.encoder(x)
        if scale_features:
            x = self.scale_features(x)
        x = self.classifier(x)
        return x

    def attribute_integrated_gradients(self, img: Tensor, **kwargs) -> Tensor:
        """Compute integrated gradients for a binary classification task.

        Parameters
        ----------
        img : Tensor
            input image
        **kwargs : Any
            Keyword arguments for ``IntegratedGradients()``.

        Returns
        -------
        attribution : Tensor
            Integrated gradients attribution map.
        """
        self.zero_grad()
        ig = IntegratedGradients(self, **kwargs)
        attribution = ig.attribute(img)
        return attribution

    def attribute_occlusion(self, img: Tensor, **kwargs) -> Tensor:
        """Compute occlusion-based attribution for a binary classification task.

        Parameters
        ----------
        img : Tensor
            input image
        **kwargs : Any
            Keyword arguments for the ``Occlusion.attribute()``.

        Returns
        -------
        attribution : Tensor
            Occlusion attribution map.
        """
        oc = Occlusion(self)
        return oc.attribute(img, **kwargs)
