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
