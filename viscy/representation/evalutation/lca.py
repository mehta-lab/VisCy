"""Linear probing of trained encoder based on cell state labels."""

from typing import Mapping

import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, Occlusion
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from xarray import DataArray

from viscy.representation.contrastive import ContrastiveEncoder


def fit_logistic_regression(
    features: DataArray,
    annotations: pd.Series,
    train_fovs: list[str],
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
    train_fovs : list[str]
        List of FOVs to use for training. The rest will be used for testing.
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
    fov_selection = features["fov_name"].isin(train_fovs)
    train_selection = fov_selection
    test_selection = ~fov_selection
    annotations = annotations.cat.codes.values.copy()
    if remove_background_class:
        label_selection = annotations != 0
        train_selection &= label_selection
        test_selection &= label_selection
        annotations -= 1
    train_features = features.values[train_selection]
    test_features = features.values[test_selection]
    if scale_features:
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.fit_transform(test_features)
    train_annotations = annotations[train_selection]
    test_annotations = annotations[test_selection]
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
