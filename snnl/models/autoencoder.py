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

from snnl.loss import composite_loss

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        model_device: torch.device = torch.device("cpu"),
        input_shape: int,
        code_dim: int,
        learning_rate: float = 1e-3,
    ):
        """
        Constructs the autoencoder model with the following units,
        <input_shape>-500-500-2000-<code_dim>-2000-500-500-<input_shape>

        Parameters
        ----------
        model_device: torch.device
            The device to use for the model computations.
        input_shape: int
            The dimensionality of the input features.
        code_dim: int
            The dimensionality of the latent code.
        learning_rate: float
            The learning rate to use for optimization.
        """
        super().__init__()
        self.model_device = model_device
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=input_shape, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=2000),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2000, out_features=code_dim),
                torch.nn.Sigmoid(),
                torch.nn.Linear(in_features=code_dim, out_features=2000),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2000, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=500),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=500, out_features=input_shape),
            ]
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.model_device)
        self.train_loss = []

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

    def fit(self, data_loader, epochs, use_snnl=False, factor=None):
        """
        Trains the autoencoder model.

        Parameters
        ----------
        data_loader : torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        use_snnl : bool
            Whether to use soft nearest neighbor loss or not.
        factor : float
            The soft nearest neighbor loss factor.
        """
        self.to(self.model_device)

        if use_snnl:
            assert factor is not None, "[factor] must not be None if use_snnl == True"
            train_snn_loss = []
            train_recon_loss = []

        for epoch in range(epochs):
            epoch_loss = epoch_train(self, data_loader, epoch, use_snnl, factor)

            if "cuda" in self.model_device.type:
                torch.cuda.empty_cache()

            if type(epoch_loss) is tuple:
                self.train_loss.append(epoch_loss[0])
                train_snn_loss.append(epoch_loss[1])
                train_recon_loss.append(epoch_loss[2])
                print(
                    f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                )
                print(
                    f"\trecon loss = {train_recon_loss[-1]:.6f}\t|\tsnn loss = {train_snn_loss[-1]:.6f}"
                )
            else:
                self.train_loss.append(epoch_loss)
                print(
                    f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                )


def epoch_train(model, data_loader, epoch=None, use_snnl=False, factor=None):
    """
    Trains a model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    data_loader : torch.utils.dataloader.DataLoader
        The data loader object that consists of the data pipeline.
    epoch : int
        The epoch number of the training.
    use_snnl : bool
        Whether to use soft nearest neighbor loss or not.
    factor : float
        The soft nearest neighbor loss factor.

    Returns
    -------
    epoch_loss : float
        The epoch loss.
    """
    if use_snnl:
        assert epoch is not None, "[epoch] must not be None if use_snnl == True"
        assert factor is not None, "[factor] must not be None if use_snnl == True"
        epoch_recon_loss = 0
        epoch_snn_loss = 0
    epoch_loss = 0
    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.view(batch_features.shape[0], -1)
        batch_features = batch_features.to(model.model_device)
        batch_labels = batch_labels.to(model.model_device)
        if use_snnl:
            outputs = model(batch_features)
            train_loss, snn_loss, recon_loss = composite_loss(
                model=model,
                outputs=outputs,
                features=batch_features,
                labels=batch_labels,
                epoch=epoch,
                factor=factor,
                unsupervised=True,
            )
            epoch_loss += train_loss.item()
            epoch_snn_loss += snn_loss.item()
            epoch_recon_loss += recon_loss.item()
        else:
            model.optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = model.criterion(outputs, batch_features)
            train_loss.backward()
            model.optimizer.step()
            epoch_loss += train_loss.item()
    epoch_loss /= len(data_loader)
    if use_snnl:
        epoch_snn_loss /= len(data_loader)
        epoch_recon_loss /= len(data_loader)
        return epoch_loss, epoch_snn_loss, epoch_recon_loss
    else:
        return epoch_loss
