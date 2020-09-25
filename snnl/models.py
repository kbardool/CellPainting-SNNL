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
"""Implementation of models"""
from typing import List, Tuple

import torch
import torchvision

from snnl import composite_loss, SNNL, SNNLoss

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class Autoencoder(torch.nn.Module):
    """
    A feed-forward autoencoder neural network that optimizes
    binary cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the binary cross entropy.
    """

    _supported_modes = ["autoencoding", "latent_code"]

    def __init__(
        self,
        input_shape: int,
        code_dim: int,
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 1e-3,
        use_snnl: bool = False,
        factor: float = 100.0,
        temperature: int = None,
        mode: str = "autoencoding",
        code_units: int = None,
        stability_epsilon: float = 1e-5,
    ):
        """
        Constructs the autoencoder model with the following units,
        <input_shape>-500-500-2000-<code_dim>-2000-500-500-<input_shape>

        Parameters
        ----------
        input_shape: int
            The dimensionality of the input features.
        code_dim: int
            The dimensionality of the latent code.
        device: torch.device
            The device to use for the model computations.
        learning_rate: float
            The learning rate to use for optimization.
        use_snnl: bool
            Whether to use soft nearest neighbor loss or not.
        factor: float
            The balance factor between SNNL and the primary loss.
            A positive factor implies SNNL minimization, while a negative
            factor implies SNNL maximization.
        temperature: int
            The SNNL temperature.
        mode: str
            The mode in which the soft nearest neighbor loss
            will be used.
        code_units: int
            The number of units in which the SNNL will be applied.
        stability_epsilon: float
            A constant for helping SNNL computation stability.
        """
        super().__init__()
        mode = mode.lower()
        if mode not in Autoencoder._supported_modes:
            raise ValueError(f"Mode {mode} is not supported.")
        if (mode == "latent_code") and (code_units <= 0):
            raise ValueError(
                "[code_units] must be greater than 0 when mode == 'latent_code'."
            )
        assert factor is not None, "[factor] must not be None if use_snnl == True."
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
                torch.nn.Sigmoid(),
            ]
        )

        for index, layer in enumerate(self.layers):
            if (index == 6 or index == 14) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                pass

        self.device = device
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.BCELoss().to(self.device)
        self.train_loss = []
        self.to(self.device)
        self.use_snnl = use_snnl
        self.factor = factor
        self.code_units = code_units
        self.temperature = temperature
        self.mode = mode
        self.stability_epsilon = stability_epsilon
        if self.use_snnl:
            self.snnl_criterion = SNNLoss(
                mode=self.mode,
                factor=self.factor,
                temperature=self.temperature,
                code_units=self.code_units,
                stability_epsilon=self.stability_epsilon,
            )

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

    def fit(self, data_loader, epochs, show_every=2):
        """
        Trains the autoencoder model.

        Parameters
        ----------
        data_loader : torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        show_every : int
            The interval in terms of epoch on displaying training progress.
        """
        if self.use_snnl:
            self.train_snn_loss = []
            self.train_recon_loss = []

        for epoch in range(epochs):
            epoch_loss = self.epoch_train(data_loader, epoch)
            if type(epoch_loss) is tuple:
                self.train_loss.append(epoch_loss[0])
                self.train_snn_loss.append(epoch_loss[1])
                self.train_recon_loss.append(epoch_loss[2])
                if (epoch + 1) % show_every == 0:
                    print(
                        f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                    )
                    print(
                        f"\trecon loss = {self.train_recon_loss[-1]:.6f}\t|\tsnn loss = {self.train_snn_loss[-1]:.6f}"
                    )
            else:
                self.train_loss.append(epoch_loss)
                if (epoch + 1) % show_every == 0:
                    print(
                        f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                    )

    def epoch_train(self, data_loader, epoch=None):
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

        Returns
        -------
        epoch_loss : float
            The epoch loss.
        epoch_snn_loss : float
            The soft nearest neighbor loss for an epoch.
        epoch_recon_loss : float
            The reconstruction loss for an epoch.
        """
        if self.use_snnl:
            assert epoch is not None, "[epoch] must not be None if use_snnl == True"
            epoch_recon_loss = 0
            epoch_snn_loss = 0
        epoch_loss = 0
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.view(batch_features.shape[0], -1)
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            if self.use_snnl:
                outputs = self(batch_features)
                train_loss, recon_loss, snn_loss = self.snnl_criterion(
                    model=self,
                    features=batch_features,
                    labels=batch_labels,
                    outputs=outputs,
                    epoch=epoch,
                )
                epoch_loss += train_loss.item()
                epoch_snn_loss += snn_loss.item()
                epoch_recon_loss += recon_loss.item()
                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                self.optimizer.zero_grad()
                outputs = self(batch_features)
                train_loss = self.criterion(outputs, batch_features)
                train_loss.backward()
                self.optimizer.step()
                epoch_loss += train_loss.item()
        epoch_loss /= len(data_loader)
        if self.use_snnl:
            epoch_snn_loss /= len(data_loader)
            epoch_recon_loss /= len(data_loader)
            return epoch_loss, epoch_snn_loss, epoch_recon_loss
        else:
            return epoch_loss


class DNN(torch.nn.Module):
    """
    A feed-forward neural network that optimizes
    softmax cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the softmax cross entropy.
    """

    def __init__(
        self,
        units: List or Tuple = [(784, 500), (500, 500), (500, 10)],
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 1e-3,
        use_snnl: bool = False,
        factor: float = 100.0,
        temperature: int = None,
        stability_epsilon: float = 1e-5,
    ):
        """
        Constructs a feed-forward neural network classifier.

        Parameters
        ----------
        units: list or tuple
            An iterable that consists of the number of units in each hidden layer.
        device: torch.device
            The device to use for model computations.
        learning_rate: float
            The learning rate to use for optimization.
        use_snnl: bool
            Whether to use soft nearest neighbor loss or not.
        factor: float
            The balance between SNNL and the primary loss.
            A positive factor implies SNNL minimization,
            while a negative factor implies SNNL maximization.
        temperature: int
            The SNNL temperature.
        stability_epsilon: float
            A constant for helping SNNL computation stability
        """
        super().__init__()
        self.device = device
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=in_features, out_features=out_features)
                for in_features, out_features in units
            ]
        )

        for index, layer in enumerate(self.layers):
            if index < (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif index == (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            else:
                pass

        self.train_loss = []
        self.train_accuracy = []
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.to(self.device)
        self.use_snnl = use_snnl
        self.factor = factor
        self.temperature = temperature
        self.stability_epsilon = stability_epsilon
        if self.use_snnl:
            self.snnl_criterion = SNNLoss(
                mode="classifier",
                factor=self.factor,
                temperature=self.temperature,
                stability_epsilon=self.stability_epsilon,
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
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

    def fit(self, data_loader, epochs, show_every=2):
        """
        Trains the dnn model.

        Parameters
        ----------
        data_loader : torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        show_every : int
            The interval in terms of epoch on displaying training progress.
        """
        if self.use_snnl:
            assert (
                self.factor is not None
            ), "[factor] must not be None if use_snnl == True"
            self.train_snn_loss = []
            self.train_xent_loss = []

        for epoch in range(epochs):
            if self.use_snnl:
                *epoch_loss, epoch_accuracy = self.epoch_train(data_loader, epoch)
                self.train_loss.append(epoch_loss[0])
                self.train_snn_loss.append(epoch_loss[1])
                self.train_xent_loss.append(epoch_loss[2])
                self.train_accuracy.append(epoch_accuracy)
                if (epoch + 1) % show_every == 0:
                    print(f"epoch {epoch + 1}/{epochs}")
                    print(
                        f"\tmean loss = {self.train_loss[-1]:.6f}\t|\tmean acc = {self.train_accuracy[-1]:.6f}"
                    )
                    print(
                        f"\txent loss = {self.train_xent_loss[-1]:.6f}\t|\tsnn loss = {self.train_snn_loss[-1]:.6f}"
                    )
            else:
                epoch_loss, epoch_accuracy = self.epoch_train(data_loader)
                self.train_loss.append(epoch_loss)
                if (epoch + 1) % show_every == 0:
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

    @staticmethod
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
        epoch_snn_loss : float
            The soft nearest neighbor loss for an epoch.
        epoch_xent_loss : float
            The cross entropy loss for an epoch.
        """
        if use_snnl:
            assert epoch is not None, "[epoch] must not be None if use_snnl == True"
            epoch_xent_loss = 0
            epoch_snn_loss = 0
        epoch_loss = 0
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.view(batch_features.shape[0], -1)
            batch_features = batch_features.to(model.device)
            batch_labels = batch_labels.to(model.device)
            if use_snnl:
                outputs = model(batch_features)
                train_loss, snn_loss, xent_loss = composite_loss(
                    model=model,
                    outputs=outputs,
                    features=batch_features,
                    labels=batch_labels,
                    epoch=epoch,
                    temperature=temperature,
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
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 1e-4,
    ):
        """
        Constructs a convolutional neural network classifier.

        Parameters
        ----------
        device: torch.device
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
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.train_loss = []
        self.to(self.device)

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

    def fit(
        self,
        data_loader,
        epochs,
        use_snnl=False,
        factor=None,
        temperature=None,
        show_every=2,
    ):
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
        show_every : int
            The interval in terms of epoch on displaying training progress.
        """
        if use_snnl:
            assert factor is not None, "[factor] must not be None if use_snnl == True"
            self.train_snn_loss = []
            self.train_xent_loss = []

        for epoch in range(epochs):
            epoch_loss = self.epoch_train(
                self, data_loader, epoch, use_snnl, factor, temperature
            )

            if type(epoch_loss) is tuple:
                self.train_loss.append(epoch_loss[0])
                self.train_snn_loss.append(epoch_loss[1])
                self.train_xent_loss.append(epoch_loss[2])
                if (epoch + 1) % show_every == 0:
                    print(
                        f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                    )
                    print(
                        f"\txent loss = {self.train_xent_loss[-1]:.6f}\t|\tsnn loss = {self.train_snn_loss[-1]:.6f}"
                    )
            else:
                self.train_loss.append(epoch_loss)
                if (epoch + 1) % show_every == 0:
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

    @staticmethod
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
        epoch_snn_loss : float
            The soft nearest neighbor loss for an epoch.
        epoch_xent_loss : float
            The cross entropy loss for an epoch.
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


class ResNet(torch.nn.Module):
    """
    A residual neural network that optimizes
    softmax cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the softmax cross entropy.
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Constructs a residual neural network classifier.

        Parameters
        ----------
        num_classes: int
            The number of classes in the dataset.
        learning_rate: float
            The learning rate to use for optimization.
        device: torch.device
            The device to use for model computations.
        """
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = []

    def forward(self, features):
        pass

    def fit(
        self,
        data_loader,
        epochs,
        use_snnl=False,
        factor=None,
        temperature=None,
        show_every=2,
    ):
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
        show_every : int
            The interval in terms of epoch on displaying training progress.
        """
        self.to(self.device)
        if use_snnl:
            assert factor is not None, "[factor] must not be None if use_snnl == True"
            self.train_snn_loss = []
            self.train_xent_loss = []

        for epoch in range(epochs):
            epoch_loss = self.epoch_train(
                self, data_loader, epoch, use_snnl, factor, temperature
            )

            if type(epoch_loss) is tuple:
                self.train_loss.append(epoch_loss[0])
                self.train_snn_loss.append(epoch_loss[1])
                self.train_xent_loss.append(epoch_loss[2])
                if (epoch + 1) % show_every == 0:
                    print(
                        f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                    )
                    print(
                        f"\txent loss = {self.train_xent_loss[-1]:.6f}\t|\tsnn loss = {self.train_snn_loss[-1]:.6f}"
                    )
            else:
                self.train_loss.append(epoch_loss)
                if (epoch + 1) % show_every == 0:
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

    @staticmethod
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
        epoch_snn_loss : float
            The soft nearest neighbor loss for an epoch.
        epoch_xent_loss : float
            The cross entropy loss for an epoch.
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
                train_loss, snn_loss, xent_loss = ResNet.composite_loss(
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

    @staticmethod
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


class ResNet18(ResNet):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Loads a pretrained ResNet18 classifier.

        Parameters
        ----------
        num_classes: int
            The number of classes in the dataset.
        learning_rate: float
            The learning rate to use for optimization.
        device:
            The device to use for computations.
        """
        super().__init__(
            num_classes=num_classes, learning_rate=learning_rate, device=device
        )
        self.device = device
        self.resnet = torchvision.models.resnet.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=num_classes
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, features):
        return self.resnet.forward(features)


class ResNet34(ResNet):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Loads a pretrained ResNet34 classifier.

        Parameters
        ----------
        num_classes: int
            The number of classes in the dataset.
        learning_rate: float
            The learning rate to use for optimization.
        device:
            The device to use for computations.
        """
        super().__init__(
            num_classes=num_classes, learning_rate=learning_rate, device=device
        )
        self.device = device
        self.resnet = torchvision.models.resnet.resnet34(pretrained=True)
        self.resnet.fc = torch.nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=num_classes
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, features):
        return self.resnet.forward(features)
