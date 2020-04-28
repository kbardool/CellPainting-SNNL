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
"""Implementation of ResNet18 and ResNet34 models"""
from typing import Tuple
import torch
import torchvision

from snnl import SNNL

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class ResNet(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train_loss = []

    def forward(self, features):
        pass

    def fit(self, data_loader, epochs, use_snnl=False, factor=None, temperature=None):
        """
        Finetunes the ResNet18 model.

        Parameters
        ---------
        data_loader : torch.utils.data.DataLoader
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        use_snnl : bool
            Whether to use soft nearest neighbor loss or not. Default: [False].
        factor : float
            The soft nearest neighbor loss scaling factor.
        temperature : int
            The temperature to use for soft nearest neighbor loss.
            If None, annealing temperature will be used.
        """
        if use_snnl:
            assert factor is not None, "[factor] must not be None if use_snnl == True"
            train_snn_loss = []
            train_xent_loss = []

        for epoch in range(epochs):
            epoch_loss = epoch_train(
                self, data_loader, epoch, use_snnl, factor, temperature
            )
            if "cuda" in self.device.type:
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


def composite_loss(
    model: torch.nn.Module,
    outputs: torch.Tensor,
    batch_features: torch.Tensor,
    batch_labels: torch.Tensor,
    epoch: int,
    temperature: int = None,
    factor: float = 100.0,
) -> Tuple[float, float, float]:
    """
    Returns the composite loss with soft nearest neighbor loss.
    If the objective is unsupervised learning, the primary loss
    is reconstruction loss.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    outputs : torch.Tensor
        The model outputs.
    batch_features : torch.Tensor
        The input features.
    batch_labels : torch.Tensor
        The input labels.
    epoch : int
        The training epoch.
    temperature : int
        Use fixed temperature if not None.
    factor : float
        The SNNL factor.

    Returns
    -------
    train_loss : torch.Tensor
        The total reconstruction and soft nearest neighbor loss.
    snn_loss : torch.Tensor
        The soft nearest neighbor loss.
    primary_loss : torch.Tensor
        The loss on primary objective of the model.
    """
    model.optimizer.zero_grad()
    primary_loss = model.criterion(outputs, batch_labels)
    activations = {}
    for index, (_, layer) in enumerate(list(model.resnet.named_children())):
        if index == 0:
            activations[index] = layer(batch_features)
        elif index == 9:
            value = activations[index - 1].view(activations[index - 1].shape[0], -1)
            activations[index] = layer(value)
        else:
            activations[index] = layer(activations[index - 1])
    layers_snnl = []
    if temperature is None:
        temperature = 1.0 / ((1.0 + epoch) ** 0.55)
    for key, value in activations.items():
        if key > 6:
            if len(value.shape) > 2:
                value = value.view(value.shape[0], -1)
            layer_snnl = SNNL(
                features=value, labels=batch_labels, temperature=temperature
            )
            layers_snnl.append(layer_snnl)
    del activations
    layers_snnl = torch.FloatTensor(layers_snnl)
    layers_snnl = layers_snnl.to(model.device)
    snn_loss = sum(layers_snnl)
    train_loss = primary_loss + (factor * snn_loss)
    train_loss.backward(snn_loss)
    model.optimizer.step()
    return train_loss, snn_loss, primary_loss


def epoch_train(
    model, data_loader, epoch=None, use_snnl=False, factor=None, temperature=None
):
    """
    Trains a model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    data_loader : torch.utils.dataloader.DataLoader
        The data loader object that consists of the data pipeline.
    use_snnl : bool
        Whether to use soft nearest neighbor loss or not. Default: [False].
    factor : float
        The soft nearest neighbor loss scaling factor.
    temperature : int
        The temperature to use for soft nearest neighbor loss.
        If None, annealing temperature will be used.

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
        batch_features = batch_features.to(model.device)
        batch_labels = batch_labels.to(model.device)
        if use_snnl:
            outputs = model(batch_features)
            train_loss, snn_loss, xent_loss = composite_loss(
                model=model,
                outputs=outputs,
                batch_features=batch_features,
                batch_labels=batch_labels,
                epoch=epoch,
                factor=factor,
            )
            del outputs
            epoch_loss += train_loss.item()
            epoch_snn_loss = snn_loss.item()
            epoch_xent_loss = xent_loss.item()
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
