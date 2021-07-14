# Soft Nearest Neighbor Loss
# Copyright (C) 2020-2021  Abien Fred Agarap
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
"""Feed-forward Neural Network"""
from typing import List, Tuple

import torch

from snnl.models import Model


class DNN(Model):
    """
    A feed-forward neural network that optimizes
    softmax cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the softmax cross entropy.
    """

    _criterion = torch.nn.CrossEntropyLoss()

    def __init__(
        self,
        units: List or Tuple = [(784, 500), (500, 500), (500, 10)],
        learning_rate: float = 1e-3,
        use_snnl: bool = False,
        factor: float = 100.0,
        temperature: int = None,
        use_annealing: bool = True,
        use_sum: bool = False,
        stability_epsilon: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Constructs a feed-forward neural network classifier.

        Parameters
        ----------
        units: list or tuple
            An iterable that consists of the number of units in each hidden layer.
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
        use_annealing: bool
            Whether to use annealing temperature or not.
        use_sum: bool
            Use summation of SNNL across hidden layers if True,
            otherwise get the minimum SNNL.
        stability_epsilon: float
            A constant for helping SNNL computation stability
        device: torch.device
            The device to use for model computations.
        """
        super().__init__(
            mode="classifier",
            criterion=DNN._criterion.to(device),
            device=device,
            use_snnl=use_snnl,
            factor=factor,
            temperature=temperature,
            use_annealing=use_annealing,
            use_sum=use_sum,
            stability_epsilon=stability_epsilon,
        )
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

        self.name = "DNN"
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        if not use_snnl:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.train_accuracy = []
        self.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        logits: torch.Tensor
            The model output.
        """
        features = features.view(features.shape[0], -1)
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = torch.relu(layer(features))
            elif index == len(self.layers) - 1:
                activations[index] = layer(activations.get(index - 1))
            else:
                activations[index] = torch.relu(layer(activations.get(index - 1)))
        logits = activations.get(len(activations) - 1)
        return logits

    def fit(
        self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 2
    ) -> None:
        """
        Trains the DNN model.

        Parameters
        ----------
        data_loader: torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs: int
            The number of epochs to train the model.
        show_every: int
            The interval in terms of epoch on displaying training progress.
        """
        if self.use_snnl:
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
                self.train_accuracy.append(epoch_accuracy)
                if (epoch + 1) % show_every == 0:
                    print(f"epoch {epoch + 1}/{epochs}")
                    print(
                        f"\tmean loss = {self.train_loss[-1]:.6f}\t|\tmean acc = {self.train_accuracy[-1]:.6f}"
                    )

    def predict(
        self, features: torch.Tensor, return_likelihoods: bool = False
    ) -> torch.Tensor:
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
