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
"""Implementation of loss functions"""
from typing import Tuple
import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class SNNLoss(torch.nn.Module):
    """
    A composite loss of the Soft Nearest Neighbor Loss
    computed at each hidden layer, and a softmax
    cross entropy (for classification) loss or binary
    cross entropy (for reconstruction) loss.

    Presented in
    "Improving k-Means Clustering Performance with Disentangled Internal
    Representations" by Abien Fred Agarap and Arnulfo P. Azcarraga (2020),
    and in
    "Analyzing and Improving Representations with the Soft Nearest Neighbor
    Loss" by Nicholas Frosst, Nicolas Papernot, and Geoffrey Hinton (2019).

    https://arxiv.org/abs/2006.04535/
    https://arxiv.org/abs/1902.01889/
    """

    _supported_modes = {
        "classifier": False,
        "resnet": False,
        "autoencoding": True,
        "latent_code": True,
    }

    def __init__(
        self,
        mode: str = "classifier",
        criterion: object = torch.nn.CrossEntropyLoss(),
        factor: float = 100.0,
        temperature: int = None,
        code_units: int = 30,
        stability_epsilon: float = 1e-5,
    ):
        """
        Constructs the Soft Nearest Neighbor Loss.

        Parameters
        ----------
        mode: str
            The mode in which the soft nearest neighbor loss
            will be used. Default: [classifier]
        factor: float
            The balance factor between SNNL and the primary loss.
            A positive factor implies SNNL minimization, while a negative
            factor implies SNNL maximization.
        temperature: int
            The SNNL temperature.
        code_units: int
            The number of units in which the SNNL will be applied.
        stability_epsilon: float
            A constant for helping SNNL computation stability.
        """
        super().__init__()
        mode = mode.lower()
        if mode not in SNNLoss._supported_modes:
            raise ValueError(f"Mode {mode} is not supported.")
        if (mode == "latent_code") and (code_units <= 0):
            raise ValueError(
                "[code_units] must be greater than 0 when mode == 'latent_code'."
            )
        assert isinstance(
            code_units, int
        ), f"Expected dtype for [code_units]: int, but {code_units} is {type(code_units)}"
        self.mode = mode
        self.primary_criterion = criterion
        self.unsupervised = SNNLoss._supported_modes.get(self.mode)
        self.factor = factor
        self.temperature = temperature
        self.code_units = code_units
        self.stability_epsilon = stability_epsilon

    def forward(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        epoch: int,
    ):
        """
        Defines the forward pass for the Soft Nearest Neighbor Loss.

        Parameters
        ----------
        model: torch.nn.Module
            The model whose parameters will be optimized.
        features: torch.Tensor
            The input features.
        labels: torch.Tensor
            The corresponding labels for the input features.
        outputs: torch.Tensor
            The model outputs.
        epoch: int
            The current training epoch.
        """
        temperature = (
            (1.0 / ((1.0 + epoch) ** 0.55))
            if self.temperature is None
            else self.temperature
        )

        if self.unsupervised:
            primary_loss = model.criterion(outputs, features)
        else:
            primary_loss = model.criterion(outputs, labels)

        activations = dict()
        if self.mode == "classifier":
            for index, layer in enumerate(model.layers[:-1]):
                if index == 0:
                    activations[index] = layer(features)
                else:
                    activations[index] = layer(activations[index - 1])
        elif self.mode == "resnet":
            for index, (name, layer) in enumerate(list(model.resnet.named_children())):
                if index == 0:
                    activations[index] = layer(features)
                elif index == 9:
                    value = activations[index - 1].view(
                        activations[index - 1].shape[0], -1
                    )
                    activations[index] = layer(value)
                else:
                    activations[index] = layer(activations[index - 1])
        else:
            for index, layer in enumerate(model.layers):
                if index == 0:
                    activations[index] = layer(features)
                else:
                    activations[index] = layer(activations[index - 1])

        layers_snnl = []
        for key, value in activations.items():
            if len(value.shape) > 2:
                value = value.view(value.shape[0], -1)
            if key == 7 and self.mode == "latent_code":
                value = value[:, : self.code_units]
            a = value.clone()
            b = value.clone()
            normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
            normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
            normalized_b = torch.conj(normalized_b).T
            product = torch.matmul(normalized_a, normalized_b)
            distance_matrix = torch.sub(torch.tensor(1.0), product)
            pairwise_distance_matrix = torch.exp(
                -(distance_matrix / temperature)
            ) - torch.eye(value.shape[0]).to(model.device)
            pick_probability = pairwise_distance_matrix / (
                self.stability_epsilon
                + torch.sum(pairwise_distance_matrix, 1).view(-1, 1)
            )
            masking_matrix = torch.squeeze(
                torch.eq(labels, labels.unsqueeze(1)).float()
            )
            masked_pick_probability = pick_probability * masking_matrix
            summed_masked_pick_probability = torch.sum(masked_pick_probability, dim=1)
            snnl = torch.mean(
                -torch.log(self.stability_epsilon + summed_masked_pick_probability)
            )
            if self.mode == "latent_code":
                if key == 7:
                    layers_snnl.append(snnl)
                    break
            elif self.mode == "resnet":
                if key > 6:
                    layers_snnl.append(snnl)
            else:
                layers_snnl.append(snnl)
        snn_loss = torch.stack(layers_snnl).sum()
        train_loss = torch.add(primary_loss, torch.mul(self.factor, snn_loss))
        return train_loss, primary_loss, snn_loss
