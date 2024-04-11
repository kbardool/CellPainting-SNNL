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
from typing import Tuple

import torch
from pt_datasets import create_dataloader

from snnl import SNNLoss


class Model(torch.nn.Module):
    def __init__(
        self,
        mode: str,
        criterion: object,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        use_snnl: bool = False,
        factor: float = 100.0,
        temperature: float = 100.0,
        use_annealing: bool = False,
        use_sum: bool = False,
        code_units: int = 0,
        stability_epsilon: float = 1e-5,
    ):
        super().__init__()
        print(f" Building Base Model from models/base.py")
        mode = mode.lower()
        self.mode = mode
        self.device = device
        self.train_loss = []
        self.use_snnl = use_snnl
        self.factor = factor
        self.code_units = code_units
        self.stability_epsilon = stability_epsilon
        if self.use_snnl:
            if temperature is not None:
                self.temperature = torch.nn.Parameter(
                    data=torch.tensor([temperature]), requires_grad=True
                )
                self.register_parameter(name="temperature", param=self.temperature)
            else:
                self.temperature = temperature
            self.use_annealing = use_annealing
            self.use_sum = use_sum
            self.snnl_criterion = SNNLoss(
                mode=self.mode,
                criterion=criterion,
                factor=self.factor,
                temperature=self.temperature,
                use_annealing=self.use_annealing,
                use_sum=self.use_sum,
                code_units=self.code_units,
                stability_epsilon=self.stability_epsilon,
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

    def sanity_check(
        self,
        data_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        show_every: int = 2,
    ):
        """
        Trains the model on a subset of the dataset.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
            The data loader that consists of the data pipeline.
        epochs: int
            The number of epochs to train the model.
        show_every:
            The epoch interval between progress displays.
        """
        batch_size = data_loader.batch_size
        subset = len(data_loader.dataset.data) * 0.10
        subset = int(subset)
        assert subset > batch_size, "[subset] must be greater than [batch_size]."
        features = data_loader.dataset.data[:subset] / 255.0
        labels = data_loader.dataset.targets[:subset]
        dataset = torch.utils.data.TensorDataset(features, labels)
        data_loader = create_dataloader(
            dataset=dataset, batch_size=batch_size, num_workers=0
        )
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_features, batch_labels in data_loader:
                if self.name in ["Autoencoder", "DNN"]:
                    batch_features = batch_features.view(batch_features.shape[0], -1)
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(features=batch_features)
                train_loss = self.criterion(
                    outputs,
                    batch_labels if self.name in ["CNN", "DNN"] else batch_features,
                )
                epoch_loss += train_loss.item()
                train_loss.backward()
                self.optimizer.step()
            epoch_loss /= len(data_loader)
            if (epoch + 1) % show_every == 0:
                print(f"epoch {epoch + 1}/{epochs}")
                print(f"mean loss = {epoch_loss:4f}")

    def epoch_train(
        self, data_loader: torch.utils.data.DataLoader, epoch: int = None, verbose: bool = False,
    ) -> Tuple:
        """
        Trains a model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epoch: int
            The current epoch training index.

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
            epoch_primary_loss = 0
            epoch_snn_loss = 0
        if self.name == "DNN" or self.name == "CNN":
            epoch_accuracy = 0
        epoch_loss = 0
        batch_count = 0
        for batch_features, batch_labels in data_loader:
            batch_count +=1
            if self.name in ["Autoencoder", "DNN"]:
                batch_features = batch_features.view(batch_features.shape[0], -1)
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(features=batch_features)
            if self.use_snnl:
                train_loss, primary_loss, snn_loss = self.snnl_criterion(
                    model=self,
                    outputs=outputs,
                    features=batch_features,
                    labels=batch_labels,
                    epoch=epoch,
                )
                epoch_loss += train_loss.item()
                epoch_snn_loss += snn_loss.item()
                epoch_primary_loss += primary_loss.item()
            else:
                train_loss = self.criterion(
                    outputs,
                    batch_labels
                    if self.name == "DNN" or self.name == "CNN"
                    else batch_features,
                )
                epoch_loss += train_loss.item()
            if self.name == "DNN" or self.name == "CNN":
                train_accuracy = (outputs.argmax(1) == batch_labels).sum().item() / len(
                    batch_labels
                )
                epoch_accuracy += train_accuracy
            train_loss.backward()
            self.optimizer.step()
            
            # print(f" batch:{batch_count:3d} - ttl loss:  {train_loss:10.6f}  XEntropy: {primary_loss:10.6f}    SNN: {snn_loss*self.snnl_criterion.factor:10.6f}" 
            #       f" (loss: {snn_loss:10.6f} * {self.snnl_criterion.factor})   temp: {self.temperature.item():16.12f}   temp.grad: {self.temperature.grad.item():16.12f}")    
            
            if self.use_snnl and self.temperature is not None:
                self.optimize_temperature()
        epoch_loss /= len(data_loader)
        if self.name in ["DNN", "CNN"]:
            epoch_accuracy /= len(data_loader)
        if self.use_snnl:
            epoch_snn_loss /= len(data_loader)
            epoch_primary_loss /= len(data_loader)
            if self.name == "DNN" or self.name == "CNN":
                return epoch_loss, epoch_snn_loss, epoch_primary_loss, epoch_accuracy
            else:
                return epoch_loss, epoch_snn_loss, epoch_primary_loss
        else:
            if self.name == "DNN" or self.name == "CNN":
                return epoch_loss, epoch_accuracy
            else:
                return epoch_loss

    def optimize_temperature(self):
        """
        Learns an optimized temperature parameter.
        """
        temperature_gradients = self.temperature.grad
        updated_temperature = self.temperature - (1e-1 * temperature_gradients)
        self.temperature.data = updated_temperature
