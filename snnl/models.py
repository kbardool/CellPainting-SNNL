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
from typing import Dict, List, Tuple

from pt_datasets import create_dataloader
import torch
import torchvision

from snnl import SNNLoss


__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


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
        code_units: int = 0,
        stability_epsilon: float = 1e-5,
    ):
        super().__init__()
        mode = mode.lower()
        self.mode = mode
        self.device = device
        self.train_loss = []
        self.use_snnl = use_snnl
        self.factor = factor
        self.code_units = code_units
        self.temperature = torch.nn.Parameter(
            data=torch.tensor([temperature]), requires_grad=True
        )
        self.register_parameter(name="temperature", param=self.temperature)
        self.stability_epsilon = stability_epsilon
        if self.use_snnl:
            self.snnl_criterion = SNNLoss(
                mode=self.mode,
                criterion=criterion,
                factor=self.factor,
                temperature=self.temperature,
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
        self, data_loader: torch.utils.data.DataLoader, epoch: int = None
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
        for batch_features, batch_labels in data_loader:
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
        temperature_gradients = self.temperature.grad
        updated_temperature = self.temperature - (1e-1 * temperature_gradients)
        self.temperature.data = updated_temperature


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
                activations[index] = layer(activations[index - 1])
        reconstruction = activations[len(activations) - 1]
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
                activations[index] = layer(activations[index - 1])
            else:
                activations[index] = torch.relu(layer(activations[index - 1]))
        logits = activations[len(activations) - 1]
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
                activations[index] = layer(activations[index - 1])
        logits = activations[len(activations) - 1]
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
                train_loss, snn_loss, xent_loss = self.snnl_criterion(
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
