import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from viscy.representation.evaluation.lca import linear_from_binary_logistic_regression


def test_linear_from_logistic_regression():
    """
    Test ``linear_from_logistic_regression``.
    Check that the logits from the logistic regression
    and the linear model are almost equal.
    """
    rand_data = np.random.rand(100, 8)
    rand_labels = np.random.randint(0, 2, size=(100))
    logistic_regression = LogisticRegression().fit(rand_data, rand_labels)
    linear_model = linear_from_binary_logistic_regression(logistic_regression)
    logistic_logits = logistic_regression.decision_function(rand_data)
    with torch.inference_mode():
        torch_logits = (
            linear_model(torch.from_numpy(rand_data).float()).squeeze().numpy()
        )
    np.testing.assert_allclose(logistic_logits, torch_logits, rtol=1e-3)
