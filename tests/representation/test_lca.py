import numpy as np

from viscy.representation.lca import Trainer, train_and_test_linear_classifier


def test_train_and_test_linear_classifier(caplog):
    """Test ``train_and_test_linear_classifier``."""
    embeddings = np.random.rand(10, 8)
    labels = np.random.randint(0, 2, 10)
    with caplog.at_level("INFO"):
        train_and_test_linear_classifier(
            embeddings,
            labels,
            num_classes=3,
            trainer=Trainer(fast_dev_run=True),
            batch_size=4,
        )
    assert "accuracy_macro" in caplog.text
    assert "f1_weighted" in caplog.text
