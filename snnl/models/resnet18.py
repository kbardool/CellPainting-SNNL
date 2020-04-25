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
"""Implementation of ResNet18 model"""
import torch
import torchvision


__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class ResNet18(torch.nn.Module):
    """
    A pre-trained ResNet18 model that optimizes
    softmax corss entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularized can be used with the softmax cross entropy.
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Constructs and loads a pre-trained ResNet18 classifier.

        Parameters
        ----------
        num_classes: int
            The number of classes in a dataset.
        learning_rate: float
            The learning rate to use for optimization.
        device: torch.device
            The device to use for model computations.
        """
        super().__init__()
        self.resnet = torchvision.models.resnet.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=num_classes
        )
        self.device = device
        self.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, features):
        """
        The forward pass of the model.

        Parameter
        ---------
        features : torch.Tensor
            The input features.

        Returns
        -------
        logits : torch.Tensor
            The model output.
        """
        return self.resnet.forward(features)

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
                self.resnet, data_loader, epoch, use_snnl, factor, temperature
            )
            if "cuda" in self.device.type:
                torch.cuda.empty_cache()

            if type(epoch_loss) is tuple:
                self.train_loss.append(epoch_loss[0])
                train_snn_loss.append(epoch_loss[1])
                train_xent_loss.append(epoch_loss[2])
                print(
                    f"epoch {epoch + 1}/{epoch} : mean loss = {self.train_loss[-1]:.6f}"
                )
                print(
                    f"\txent loss = {train_xent_loss[-1]:.6f}\t|\tsnn loss = {train_snn_loss[-1]:.6f}"
                )
            else:
                self.train_loss.append(epoch_loss)
                print(
                    f"epoch {epoch + 1}/{epoch} : mean loss = {self.train_loss[-1]:.6f}"
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
    if use_snnl:
        assert epoch is not None, "[epoch] must not be None if use_snnl == True"
        epoch_xent_loss = 0
        epoch_snn_loss = 0
    epoch_loss = 0
    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.to(model.device)
        batch_labels = batch_labels.to(model.device)
        if use_snnl:
            pass
        else:
            model.optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = model.criterion(outputs, batch_labels)
            train_loss.backward()
            model.optimizer.step()
            epoch_loss += train_loss.item()
    epoch_loss /= len(data_loader)
    return epoch_loss
