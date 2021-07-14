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
"""ResNet"""
from typing import Tuple

import torch
import torchvision

from snnl import SNNLoss


class ResNet(torch.nn.Module):
    """
    A residual neural network that optimizes
    softmax cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the softmax cross entropy.
    """

    def __init__(
        self,
        use_snnl: bool = False,
        factor: float = 100.0,
        mode: str = "resnet",
        stability_epsilon: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Constructs a residual neural network classifier.

        Parameters
        ----------
        use_snnl: bool
            Whether to use soft nearest neighbor loss or not.
        factor: float
            The balance factor between SNNL and the primary loss.
            A positive factor implies SNNL minimization,
            while a negative factor implies SNNL maximization.
        mode: str
            The mode in which the soft nearest neighbor loss
            will be used.
        stability_epsilon: float
            A constant for helping SNNL computation stability.
        device: torch.device
            The device to use for model computations.
        """
        super().__init__()
        self.name = "ResNet"
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = []
        self.train_accuracy = []
        self.device = device
        self.use_snnl = use_snnl
        self.factor = factor
        self.mode = mode
        self.stability_epsilon = stability_epsilon
        if self.use_snnl:
            self.snnl_criterion = SNNLoss(
                mode=self.mode,
                factor=self.factor,
                stability_epsilon=self.stability_epsilon,
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pass

    def fit(
        self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 2
    ) -> None:
        """
        Finetunes the ResNet18 model.

        Parameters
        ---------
        data_loader : torch.utils.data.DataLoader
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        show_every : int
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

    def epoch_train(
        self, data_loader: torch.utils.data.DataLoader, epoch: int = None
    ) -> Tuple:
        """
        Trains a model for one epoch.

        Parameters
        ----------
        model: torch.nn.Module
            The model to train.
        data_loader: torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epoch: int
            The epoch training index.

        Returns
        -------
        epoch_loss: float
            The epoch loss.
        epoch_snn_loss: float
            The soft nearest neighbor loss for an epoch.
        epoch_xent_loss: float
            The cross entropy loss for an epoch.
        epoch_accuracy: float
            The epoch accuracy.
        """
        if self.use_snnl:
            epoch_xent_loss = 0
            epoch_snn_loss = 0
        epoch_loss = 0
        epoch_accuracy = 0
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            if self.use_snnl:
                self.optimizer.zero_grad()
                outputs = self.forward(batch_features)
                train_loss, xent_loss, snn_loss = self.snnl_criterion(
                    model=self,
                    outputs=outputs,
                    features=batch_features,
                    labels=batch_labels,
                    epoch=epoch,
                )
                epoch_loss += train_loss.item()
                epoch_snn_loss = snn_loss.item()
                epoch_xent_loss = xent_loss.item()
                train_loss.backward()
                self.optimizer.step()
                train_accuracy = (outputs.argmax(1) == batch_labels).sum().item() / len(
                    batch_labels
                )
                epoch_accuracy += train_accuracy
            else:
                self.optimizer.zero_grad()
                outputs = self(batch_features)
                train_loss = self.criterion(outputs, batch_labels)
                train_loss.backward()
                self.optimizer.step()
                epoch_loss += train_loss.item()
                train_accuracy = (outputs.argmax(1) == batch_labels).sum().item() / len(
                    batch_labels
                )
                epoch_accuracy += train_accuracy
        epoch_loss /= len(data_loader)
        epoch_accuracy /= len(data_loader)
        if self.use_snnl:
            epoch_snn_loss /= len(data_loader)
            epoch_xent_loss /= len(data_loader)
            return epoch_loss, epoch_snn_loss, epoch_xent_loss, epoch_accuracy
        else:
            return epoch_loss, epoch_accuracy


class ResNet18(ResNet):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        use_snnl: bool = False,
        factor: float = 100.0,
        mode: str = "resnet",
        stability_epsilon: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Loads a pretrained ResNet18 classifier.

        Parameters
        ----------
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
        mode: str
            The mode in which the soft nearest neighbor loss
            will be used.
        stability_epsilon: float
            A constant for helping SNNL computation stability.
        device: torch.device
            The device to use for computations.
        """
        super().__init__(
            use_snnl=use_snnl,
            factor=factor,
            mode=mode,
            stability_epsilon=stability_epsilon,
            device=device,
        )
        self.resnet = torchvision.models.resnet.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=num_classes
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.resnet.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        torch.Tensor
            The model output.
        """
        return self.resnet.forward(features)


class ResNet34(ResNet):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        use_snnl: bool = False,
        factor: float = 100.0,
        mode: str = "resnet",
        stability_epsilon: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Loads a pretrained ResNet34 classifier.

        Parameters
        ----------
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
        mode: str
            The mode in which the soft nearest neighbor loss
            will be used.
        stability_epsilon: float
            A constant for helping SNNL computation stability.
        device: torch.device
            The device to use for computations.
        """
        super().__init__(
            use_snnl=use_snnl,
            factor=factor,
            mode=mode,
            stability_epsilon=stability_epsilon,
            device=device,
        )
        self.resnet = torchvision.models.resnet.resnet34(pretrained=True)
        self.resnet.fc = torch.nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=num_classes
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.resnet.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        torch.Tensor
            The model output.
        """
        return self.resnet.forward(features)
