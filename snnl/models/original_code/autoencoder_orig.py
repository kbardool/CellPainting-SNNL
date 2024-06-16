# Soft Nearest Neighbor Loss
# Copyright (C) 2020-2024  Abien Fred Agarap
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
# """Implementation of models"""
from typing import Dict, List, Tuple

import torch
import torchvision
from pt_datasets import create_dataloader

from snnl.models import Model
from snnl import SNNLoss

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class Autoencoder(Model):
    """
    A feed-forward autoencoder neural network that optimizes
    binary cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the binary cross entropy.
    """

    _supported_modes = ["autoencoding", "latent_code"]
    _criterion = torch.nn.BCELoss()

    def __init__(
        self,
        input_shape: int,
        code_dim: int,
        learning_rate: float = 1e-3,
        use_snnl: bool = False,
        factor: float = 100.0,
        temperature: int = None,
        use_annealing: bool = True,
        use_sum: bool = False,
        mode: str = "autoencoding",
        code_units: int = 0,
        stability_epsilon: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
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
        use_annealing: bool
            Whether to use annealing temperature or not.
        use_sum: bool
            Use summation of SNNL across hidden layers if True,
            otherwise get the minimum SNNL.
        mode: str
            The mode in which the soft nearest neighbor loss
            will be used.
        code_units: int
            The number of units in which the SNNL will be applied.
        stability_epsilon: float
            A constant for helping SNNL computation stability.
        device: torch.device
            The device to use for the model computations.
        """
        super().__init__(
            mode=mode,
            criterion=Autoencoder._criterion.to(device),
            device=device,
            use_snnl=use_snnl,
            factor=factor,
            code_units=code_units,
            temperature=temperature,
            use_annealing=use_annealing,
            use_sum=use_sum,
            stability_epsilon=stability_epsilon,
        )
        if mode not in Autoencoder._supported_modes:
            raise ValueError(f"Mode {mode} is not supported.")
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=input_shape, out_features=500),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=500, out_features=500),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=500, out_features=2000),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=2000, out_features=code_dim),
                torch.nn.Sigmoid(),
                torch.nn.Linear(in_features=code_dim, out_features=2000),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=2000, out_features=500),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=500, out_features=500),
                torch.nn.ReLU(inplace=True),
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

        self.name = "Autoencoder"
        self.to(self.device)
        if not use_snnl:
            self.criterion = Autoencoder._criterion.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        reconstruction: torch.Tensor
            The model output.
        """
        features = features.view(features.shape[0], -1)
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations.get(index - 1))
        reconstruction = activations.get(len(activations) - 1)
        return reconstruction

    def compute_latent_code(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes the latent code representation for the features
        using a trained autoencoder network.

        Parameters
        ----------
        features: torch.Tensor
            The features to represent in latent space.

        Returns
        -------
        latent_code: np.ndarray
            The latent code representation for the features.
        """
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features)
        activations = {}
        for index, layer in enumerate(self.layers[:8]):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations.get(index - 1))
        latent_code = activations.get(len(activations) - 1)
        latent_code = latent_code.detach().numpy()
        return latent_code

    def fit(
        self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 2
    ) -> None:
        """
        Trains the autoencoder model.

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
