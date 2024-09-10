# FIXME: this is a method from previous version at (viscy.representatin.evaluation)
# and needs to be turned into lightning module.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset


def linear_classifier_accuracy(self, batch_size=32, learning_rate=0.01, epochs=10):
    """
    Evaluate the accuracy of a single-layer neural network trained on the
    embeddings.

    Parameters
    ----------
    batch_size : int, optional
        Batch size for training. Default is 32.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.01.
    epochs : int, optional
        Number of training epochs. Default is 10.

    Returns
    -------
    float
        Accuracy of the neural network classifier.
    """

    class SingleLayerNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SingleLayerNN, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.fc(x)

    # Convert numpy arrays to PyTorch tensors
    inputs = torch.tensor(self.embeddings, dtype=torch.float32)
    labels = torch.tensor(self.annotations, dtype=torch.long)

    # Create a dataset and data loader
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the neural network, loss function, and optimizer
    input_dim = self.embeddings.shape[1]
    output_dim = len(np.unique(self.annotations))
    model = SingleLayerNN(input_dim, output_dim)
    criterion = (
        nn.CrossEntropyLoss()
    )  # Works with logits, so no softmax in the last layer

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_inputs, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        accuracy = accuracy_score(labels.numpy(), predictions.numpy())

    return accuracy
