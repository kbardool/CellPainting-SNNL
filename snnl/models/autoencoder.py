# Soft Nearest Neighbor Loss
# Copyright (C) 2020  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Implementation of a feed-forward neural network-based autoencoder"""
import torch


class Autoencoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=kwargs["input_shape"], out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=2000),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2000, out_features=kwargs["code_dim"]),
                torch.nn.Sigmoid(),
                torch.nn.Linear(in_features=kwargs["code_dim"], out_features=2000),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2000, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=kwargs["input_shape"]),
            ]
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=kwargs["learning_rate"])
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, features):
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features : torch.Tensor
            The input features.

        Returns
        -------
        reconstruction : torch.Tensor
            The model output.
        """
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations[index - 1])
        reconstruction = activations[len(activations) - 1]
        return reconstruction

    def fit(self, data_loader, epochs):
        """
        Trains the autoencoder model.

        Parameters
        ----------
        data_loader : torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        """
        train_loss = []
        for epoch in range(epochs):
            epoch_loss = epoch_train(self, data_loader)
            train_loss.append(epoch_loss)
            print(f"epoch {epoch + 1}/{epochs} : mean loss = {train_loss[-1]:.6f}")
        self.train_loss = train_loss


def epoch_train(model, data_loader):
    """
    Trains a model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    data_loader : torch.utils.dataloader.DataLoader
        The data loader object that consists of the data pipeline.

    Returns
    -------
    epoch_loss : float
        The epoch loss.
    """
    epoch_loss = 0
    for batch_features, _ in data_loader:
        batch_features = batch_features.view(batch_features.shape[0], -1)
        model.optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = model.criterion(outputs, batch_features)
        train_loss.backward()
        model.optimizer.step()
        epoch_loss += train_loss.item()
    epoch_loss /= len(data_loader)
    return epoch_loss
