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
"""Implementation of a convolutional neural network"""
import torch

from snnl.loss import composite_loss

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class CNN(torch.nn.Module):
    """
    A convolutional neural network that optimizes
    softmax cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the softmax cross entropy.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        model_device: torch.device = torch.device("cpu"),
        learning_rate: float = 1e-4,
    ):
        """
        Constructs a convolutional neural network classifier.

        Parameters
        ----------
        model_device: torch.device
            The device to use for model computations.
        input_dim: int
            The dimensionality of the input features.
        num_classes: int
            The number of classes in the dataset.
        learning_rate: float
            The learning rate to use for optimization.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    in_channels=input_dim,
                    out_channels=64,
                    kernel_size=8,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=1
                ),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(in_features=(128 * 5 * 5), out_features=1024),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=1024, out_features=1024),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=1024, out_features=512),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=512, out_features=num_classes),
            ]
        )
        self.model_device = model_device
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.model_device)
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
        logits : torch.Tensor
            The model output.
        """
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations[index - 1])
        logits = activations[len(activations) - 1]
        del activations
        return logits

    def fit(self, data_loader, epochs, use_snnl=False, factor=None, temperature=None):
        """
        Trains the cnn model.

        Parameters
        ----------
        data_loader : torch.utils.dataloader.DataLoader
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
        self.to(self.model_device)

        if use_snnl:
            assert factor is not None, "[factor] must not be None if use_snnl == True"
            train_snn_loss = []
            train_xent_loss = []

        for epoch in range(epochs):
            epoch_loss = epoch_train(
                self, data_loader, epoch, use_snnl, factor, temperature
            )

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
            del outputs
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
