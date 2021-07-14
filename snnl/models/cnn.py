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
"""Convolutional Neural Network class"""
from typing import Dict

import torch

from snnl.models import Model


class CNN(Model):
    """
    A convolutional neural network that optimizes
    softmax cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the softmax cross entropy.
    """

    _criterion = torch.nn.CrossEntropyLoss()

    _conv1_params = {"out_channels": 64, "kernel_size": 8, "padding": 1, "stride": 2}
    _conv2_params = {"out_channels": 128, "kernel_size": 6, "padding": 1, "stride": 2}

    def __init__(
        self,
        dim: int,
        input_dim: int,
        num_classes: int,
        learning_rate: float = 1e-4,
        use_snnl: bool = False,
        factor: float = 100.0,
        temperature: float = 100.0,
        use_annealing: bool = False,
        use_sum: bool = False,
        stability_epsilon: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
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
        use_snnl: bool
            Whether to use soft nearest neighbor loss or not.
        factor: float
            The balance between SNNL and the primary loss.
            A positive factor implies SNNL minimization,
            while a negative factor implies SNNL maximization.
        temperature: float
            The SNNL temperature.
        use_annealing: bool
            Whether to use annealing temperature or not.
        use_sum: bool
            Use summation of SNNL across hidden layers if True,
            otherwise get the minimum SNNL.
        stability_epsilon: float
            A constant for helping SNNL computation stability.
        """
        super().__init__(
            mode="classifier",
            criterion=CNN._criterion.to(device),
            device=device,
            use_snnl=use_snnl,
            factor=factor,
            temperature=temperature,
            use_annealing=use_annealing,
            use_sum=use_sum,
            stability_epsilon=stability_epsilon,
        )
        conv1_out = self.compute_conv_out(dim, CNN._conv1_params)
        conv2_out = self.compute_conv_out(conv1_out, CNN._conv2_params)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    in_channels=input_dim,
                    out_channels=CNN._conv1_params.get("out_channels"),
                    kernel_size=CNN._conv1_params.get("kernel_size"),
                    stride=CNN._conv1_params.get("stride"),
                    padding=CNN._conv1_params.get("padding"),
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(
                    in_channels=CNN._conv1_params.get("out_channels"),
                    out_channels=CNN._conv2_params.get("out_channels"),
                    kernel_size=CNN._conv2_params.get("kernel_size"),
                    stride=CNN._conv2_params.get("stride"),
                    padding=CNN._conv2_params.get("padding"),
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.Flatten(),
                torch.nn.Linear(
                    in_features=int(
                        CNN._conv2_params.get("out_channels") * conv2_out * conv2_out
                    ),
                    out_features=50,
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=50, out_features=512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=512, out_features=num_classes),
            ]
        )

        for index, layer in enumerate(self.layers):
            if index < (len(self.layers) - 1) and (
                isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d)
            ):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif index == (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            else:
                pass

        self.name = "CNN"
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        if not use_snnl:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.train_accuracy = []
        self.to(self.device)

    @staticmethod
    def compute_conv_out(dim: int, params: Dict) -> int:
        """
        Computes the convolutional layer output size.

        Parameters
        ----------
        dim: int
            The dimensionality of the input to the convolutional layer.
        params: Dict
            The parameters of the convolutional layer.

        Returns
        -------
        int
            The output size of the convolutional layer.
        """
        return (
            dim - params.get("kernel_size") + 2 * params.get("padding")
        ) / params.get("stride") + 1

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
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations.get(index - 1))
        logits = activations.get(len(activations) - 1)
        del activations
        return logits

    def fit(
        self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 2
    ) -> None:
        """
        Trains the cnn model.

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
