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
"""Implementation of a feed-forward neural network"""
import torch

from snnl.loss import composite_loss


class DNN(torch.nn.Module):
    """
    A feed-forward neural network that optimizes
    softmax cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the softmax cross entropy.
    """

    def __init__(
        self,
        units: list or tuple,
        model_device: torch.device = torch.device("cpu"),
        learning_rate: float = 1e-3,
    ):
        """
        Constructs a feed-forward neural network classifier.

        Parameters
        ----------
        model_device: torch.device
            The device to use for model computations.
        units: list or tuple
            An iterable that consists of the number of units in each hidden layer.
        learning_rate: float
            The learning rate to use for optimization.
        """
        super().__init__()
        self.model_device = model_device
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=in_features, out_features=out_features)
                for in_features, out_features in units
            ]
        )
        self.train_loss = []
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.model_device)

    def forward(self, features):
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features : torch.Tensor
            The input features.

        Returns
        -------
        logits : torch.Tensor
            The model output.
        """
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = torch.relu(layer(features))
            elif index == len(self.layers) - 1:
                activations[index] = layer(activations[index - 1])
            else:
                activations[index] = torch.relu(layer(activations[index - 1]))
        logits = activations[len(activations) - 1]
        return logits

    def fit(self, data_loader, epochs, use_snnl=False, factor=None):
        """
        Trains the dnn model.

        Parameters
        ----------
        data_loader : torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        """
        self.to(self.model_device)

        if use_snnl:
            assert factor is not None, "[factor] must not be None if use_snnl == True"
            train_snn_loss = []
            train_xent_loss = []

        for epoch in range(epochs):
            epoch_loss = epoch_train(self, data_loader, epoch, use_snnl, factor)

            if "cuda" in self.model_device.type:
                torch.cuda.empty_cache()

            if type(epoch_loss) is tuple:
                self.train_loss.append(epoch_loss[0])
                train_snn_loss.append(epoch_loss[1])
                train_xent_loss.append(epoch_loss[2])
                print(
                    f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                )
                print(
                    f"\txent loss = {train_xent_loss[-1]:.6f}\t|\tsnn loss = {train_snn_loss[-1]:.6f}"
                )
            else:
                self.train_loss.append(epoch_loss)
                print(
                    f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                )

    def predict(self, features, return_likelihoods=False):
        """
        Returns model classifications

        Parameters
        ----------
        features: torch.Tensor
            The input features to classify.
        return_likelihoods: bool
            Whether to return classes with likelihoods or not.

        Returns
        -------
        predictions: torch.Tensor
            The class likelihood output by the model.
        classes: torch.Tensor
            The class prediction by the model.
        """
        outputs = self.forward(features)
        predictions, classes = torch.max(outputs.data, dim=1)
        return (predictions, classes) if return_likelihoods else classes


def epoch_train(model, data_loader, epoch=None, use_snnl=False, factor=None):
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
    if use_snnl:
        assert epoch is not None, "[epoch] must not be None if use_snnl == True"
        epoch_xent_loss = 0
        epoch_snn_loss = 0
    epoch_loss = 0
    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.view(batch_features.shape[0], -1)
        batch_features = batch_features.to(model.model_device)
        batch_labels = batch_labels.to(model.model_device)
        if use_snnl:
            outputs = model(batch_features)
            train_loss, snn_loss, xent_loss = composite_loss(
                model=model,
                outputs=outputs,
                features=batch_features,
                labels=batch_labels,
                epoch=epoch,
                factor=factor,
            )
            epoch_loss += train_loss.item()
            epoch_snn_loss += snn_loss.item()
            epoch_xent_loss += xent_loss.item()
        else:
            model.optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = model.criterion(outputs, batch_labels)
            train_loss.backward()
            model.optimizer.step()
            epoch_loss += train_loss.item()

    epoch_loss /= len(data_loader)

    if use_snnl:
        epoch_snn_loss /= len(data_loader)
        epoch_xent_loss /= len(data_loader)
        return epoch_loss, epoch_snn_loss, epoch_xent_loss
    else:
        return epoch_loss
