"""Linear probing of trained encoder based on cell state labels."""

import logging
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
]:
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
    weights = torch.from_numpy(logistic_regression.coef_).float()
    bias = torch.from_numpy(logistic_regression.intercept_).float()
    model = nn.Linear(in_features=weights.shape[1], out_features=1)
    model.weight.data = weights
    model.bias.data = bias
    model.eval()
    return model


class AssembledClassifier(torch.nn.Module):
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
