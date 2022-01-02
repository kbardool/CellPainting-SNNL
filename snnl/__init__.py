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
"""Implementation of loss functions"""
from typing import Dict, Tuple

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
        "sae": True,
        "custom": False,
        "moe": False,
    }

    def __init__(
        self,
        mode: str = "classifier",
        criterion: object = torch.nn.CrossEntropyLoss(),
        factor: float = 100.0,
        temperature: float = None,
        use_annealing: bool = True,
        use_sum: bool = False,
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
        criterion: object
            The primary loss to use.
            Default: [torch.nn.CrossEntropyLoss()]
        factor: float
            The balance factor between SNNL and the primary loss.
            A positive factor implies SNNL minimization, while a negative
            factor implies SNNL maximization.
        temperature: float
            The SNNL temperature.
        use_annealing: bool
            Whether to use annealing temperature or not.
        use_sum: bool
            If true, the sum of SNNL across all hidden layers are used.
            Otherwise, the minimum SNNL will be obtained.
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
        self.use_annealing = use_annealing
        self.use_sum = use_sum
        self.code_units = code_units
        self.stability_epsilon = stability_epsilon

    def forward(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        epoch: int,
    ) -> Tuple:
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

        Returns
        -------
        train_loss: float
            The composite training loss.
        primary_loss: float
            The primary loss function value.
        snn_loss: float
            The soft nearest neighbor loss value.
        """
        if self.use_annealing:
            self.temperature = 1.0 / ((1.0 + epoch) ** 0.55)

        if self.mode == "sae":
            (
                reconstruction_criterion,
                classification_criterion,
            ) = self.primary_criterion
            reconstruction_loss = reconstruction_criterion(outputs, features)
            classification_loss = classification_criterion(outputs, features)
            primary_loss = reconstruction_loss + classification_loss

        primary_loss = self.primary_criterion(
            outputs, features if self.unsupervised else labels
        )

        activations = self.compute_activations(model=model, features=features)

        layers_snnl = []
        for key, value in activations.items():
            if len(value.shape) > 2:
                value = value.view(value.shape[0], -1)
            if (key == 7 and self.mode == "latent_code") or (
                key == 9 and self.mode == "sae"
            ):
                value = value[:, : self.code_units]
            distance_matrix = self.pairwise_cosine_distance(features=value)
            pairwise_distance_matrix = self.normalize_distance_matrix(
                features=value, distance_matrix=distance_matrix, device=model.device
            )
            pick_probability = self.compute_sampling_probability(
                pairwise_distance_matrix
            )
            summed_masked_pick_probability = self.mask_sampling_probability(
                labels, pick_probability
            )
            snnl = torch.mean(
                -torch.log(self.stability_epsilon + summed_masked_pick_probability)
            )
            if self.mode == "latent_code":
                if key == 7:
                    layers_snnl.append(snnl)
                    break
            if self.mode == "sae":
                if key == 9:
                    layers_snnl.append(snnl)
                    break
            elif self.mode == "resnet":
                if key > 6:
                    layers_snnl.append(snnl)
            else:
                layers_snnl.append(snnl)
        if self.use_sum:
            snn_loss = torch.stack(layers_snnl).sum()
        else:
            snn_loss = torch.stack(layers_snnl)
            snn_loss = torch.min(snn_loss)
        if self.mode != "moe":
            train_loss = torch.add(primary_loss, torch.mul(self.factor, snn_loss))
            return train_loss, primary_loss, snn_loss
        elif self.mode == "sae":
            train_loss = torch.add(primary_loss, torch.mul(self.factor, snn_loss))
            return train_loss, reconstruction_loss, classification_loss, snn_loss
        else:
            return primary_loss, snn_loss

    def compute_activations(
        self, model: torch.nn.Module, features: torch.Tensor
    ) -> Dict:
        """
        Returns the hidden layer activations of a model.

        Parameters
        ----------
        model: torch.nn.Module
            The model whose hidden layer representations shall be computed.
        features: torch.Tensor
            The input features.

        Returns
        -------
        activations: Dict
            The hidden layer activations of the model.
        """
        activations = dict()
        if self.mode in ["classifier", "autoencoding", "latent_code"]:
            # layers = model.layers[:-1] if self.mode == "classifier" else model.layers
            layers = model.layers
            for index, layer in enumerate(layers):
                if index == 0:
                    activations[index] = layer(features)
                else:
                    activations[index] = layer(activations[index - 1])
        elif self.mode == "sae":
            for index, layer in enumerate(model.encoder):
                activations[index] = layer(
                    features if index == 0 else activations[index - 1]
                )
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
        elif self.mode == "custom":
            for index, layer in enumerate(list(model.children())):
                activations[index] = (
                    layer(features) if index == 0 else layer(activations[index - 1])
                )
        elif self.mode == "moe":
            layers = dict(model.named_children())
            layers = layers.get("feature_extractor")
            if isinstance(layers[0], torch.nn.Linear) and len(features.shape) > 2:
                features = features.view(features.shape[0], -1)
            for index, layer in enumerate(layers):
                activations[index] = (
                    layer(features) if index == 0 else layer(activations.get(index - 1))
                )
        return activations

    def pairwise_cosine_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns the pairwise cosine distance between two copies
        of the features matrix.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        distance_matrix: torch.Tensor
            The pairwise cosine distance matrix.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> snnl.pairwise_cosine_distance(a)
        tensor([[1.1921e-07, 7.4125e-02, 1.8179e-02, 1.0152e-01],
                [7.4125e-02, 1.1921e-07, 1.9241e-02, 2.2473e-03],
                [1.8179e-02, 1.9241e-02, 1.1921e-07, 3.4526e-02],
                [1.0152e-01, 2.2473e-03, 3.4526e-02, 0.0000e+00]])
        """
        a, b = features.clone(), features.clone()
        normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
        normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
        normalized_b = torch.conj(normalized_b).T
        product = torch.matmul(normalized_a, normalized_b)
        distance_matrix = torch.sub(torch.tensor(1.0), product)
        return distance_matrix

    def normalize_distance_matrix(
        self,
        features: torch.Tensor,
        distance_matrix: torch.Tensor,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ) -> torch.Tensor:
        """
        Normalizes the pairwise distance matrix.

        Parameters
        ----------
        features: torch.Tensor
            The input features.
        distance_matrix: torch.Tensor
            The pairwise distance matrix to normalize.
        device: torch.device
            The device to use for computation.

        Returns
        -------
        pairwise_distance_matrix: torch.Tensor
            The normalized pairwise distance matrix.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> distance_matrix = snnl.pairwise_cosine_distance(a)
        >>> snnl.normalize_distance_matrix(a, distance_matrix, device=torch.device("cpu"))
        tensor([[-1.1921e-07,  9.2856e-01,  9.8199e-01,  9.0346e-01],
                [ 9.2856e-01, -1.1921e-07,  9.8094e-01,  9.9776e-01 ],
                [ 9.8199e-01,  9.8094e-01, -1.1921e-07,  9.6606e-01 ],
                [ 9.0346e-01,  9.9776e-01,  9.6606e-01,  0.0000e+00 ]])
        """
        pairwise_distance_matrix = torch.exp(
            -(distance_matrix / self.temperature)
        ) - torch.eye(features.shape[0]).to(device)
        return pairwise_distance_matrix

    def compute_sampling_probability(
        self, pairwise_distance_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the probability of sampling `j` based
        on distance between points `i` and `j`.

        Parameter
        ---------
        pairwise_distance_matrix: torch.Tensor
            The normalized pairwise distance matrix.

        Returns
        -------
        pick_probability: torch.Tensor
            The probability matrix for selecting neighbors.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> distance_matrix = snnl.pairwise_cosine_distance(a)
        >>> distance_matrix = snnl.normalize_distance_matrix(a, distance_matrix)
        >>> snnl.compute_sampling_probability(distance_matrix)
        tensor([[-4.2363e-08,  3.2998e-01,  3.4896e-01,  3.2106e-01],
                [ 3.1939e-01, -4.1004e-08,  3.3741e-01,  3.4319e-01 ],
                [ 3.3526e-01,  3.3491e-01, -4.0700e-08,  3.2983e-01 ],
                [ 3.1509e-01,  3.4798e-01,  3.3693e-01,  0.0000e+00 ]])
        """
        pick_probability = pairwise_distance_matrix / (
            self.stability_epsilon + torch.sum(pairwise_distance_matrix, 1).view(-1, 1)
        )
        return pick_probability

    def mask_sampling_probability(
        self, labels: torch.Tensor, sampling_probability: torch.Tensor
    ) -> torch.Tensor:
        """
        Masks the sampling probability, to zero out diagonal
        of sampling probability, and returns the sum per row.

        Parameters
        ----------
        labels: torch.Tensor
            The labels of the input features.
        sampling_probability: torch.Tensor
            The probability matrix of picking neighboring points.

        Returns
        -------
        summed_masked_pick_probability: torch.Tensor
            The probability matrix of selecting a
            class-similar data points.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> distance_matrix = snnl.pairwise_cosine_distance(a)
        >>> distance_matrix = snnl.normalize_distance_matrix(a, distance_matrix)
        >>> pick_probability = snnl.compute_sampling_probability(distance_matrix)
        >>> snnl.mask_sampling_probability(labels, pick_probability)
        tensor([0.3490, 0.3432, 0.3353, 0.3480])
        """
        masking_matrix = torch.squeeze(torch.eq(labels, labels.unsqueeze(1)).float())
        masked_pick_probability = sampling_probability * masking_matrix
        summed_masked_pick_probability = torch.sum(masked_pick_probability, dim=1)
        return summed_masked_pick_probability
