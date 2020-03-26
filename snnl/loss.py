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
import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def composite_loss(model, features, labels, outputs, epoch, factor=100.0, unsupervised=False):

    model.optimizer.zero_grad()

    if unsupervised:
        primary_loss = model.criterion(outputs, features)
    elif not unsupervised:
        primary_loss = model.criterion(outputs, labels)

    activations = {}

    for index, layer in enumerate(model.layers):
        if index == 0:
            activations[index] = layer(features)
        else:
            activations[index] = layer(activations[index - 1])

    layers_snnl = []

    temperature = 1.0 / ((1.0 + epoch) ** 0.55)

    for key, value in activations.items():
        layer_snnl = SNNL(features=value, labels=labels, temperature=temperature)
        layers_snnl.append(layer_snnl)

    del activations

    snn_loss = torch.min(torch.Tensor(layers_snnl))

    train_loss = [primary_loss, (factor * snn_loss)]
    train_loss = sum(train_loss)

    train_loss.backward()
    model.optimizer.step()

    return train_loss, snn_loss, primary_loss


def softmax_crossentropy(model, outputs, features, labels, epoch, factor=100.0):
    """
    Returns the entropy loss (in softmax cross entropy) with SNNL.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    outputs : torch.Tensor
        The model outputs.
    features : torch.Tensor
        The input features.
    labels : torch.Tensor
        The input labels.
    epoch : int
        The training epoch.
    factor : float
        The SNNL factor.

    Returns
    -------
    train_loss : torch.Tensor
        The total reconstruction and soft nearest neighbor loss.
    snn_loss : torch.Tensor
        The soft nearest neighbor loss.
    xent_loss : torch.Tensor
        The softmax cross entropy loss for classification.
    """

    model.optimizer.zero_grad()

    xent_loss = model.criterion(outputs, labels)

    activations = {}

    for index, layer in enumerate(model.layers):
        if index == 0:
            activations[index] = layer(features)
        else:
            activations[index] = layer(activations[index - 1])

    layers_snnl = []

    temperature = 1.0 / ((1.0 + epoch) ** 0.55)

    for key, value in activations.items():
        layer_snnl = SNNL(features=value, labels=labels, temperature=temperature)
        layers_snnl.append(layer_snnl)

    del activations

    snn_loss = torch.min(torch.Tensor(layers_snnl))

    train_loss = [xent_loss, (factor * snn_loss)]
    train_loss = sum(train_loss)

    train_loss.backward()
    model.optimizer.step()

    return train_loss, snn_loss, xent_loss


def binary_crossentropy(model, outputs, features, labels, epoch, factor=100.0):
    """
    Returns the reconstruction loss (in binary cross entropy) with SNNL.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    outputs : torch.Tensor
        The model outputs.
    features : torch.Tensor
        The input features.
    labels : torch.Tensor
        The input labels.
    epoch : int
        The training epoch.
    factor : float
        The SNNL factor.

    Returns
    -------
    train_loss : torch.Tensor
        The total reconstruction and soft nearest neighbor loss.
    snn_loss : torch.Tensor
        The soft nearest neighbor loss.
    bce_loss : torch.Tensor
        The binary cross entropy loss for reconstruction.
    """
    model.optimizer.zero_grad()

    bce_loss = model.criterion(outputs, features)

    activations = {}

    for index, layer in enumerate(model.layers):
        if index == 0:
            activations[index] = layer(features)
        else:
            activations[index] = layer(activations[index - 1])

    layers_snnl = []

    temperature = 1.0 / ((1.0 + epoch) ** 0.55)

    for key, value in activations.items():
        layer_snnl = SNNL(features=value, labels=labels, temperature=temperature)
        layers_snnl.append(layer_snnl)

    del activations

    snn_loss = torch.min(torch.Tensor(layers_snnl))

    train_loss = [bce_loss, (factor * snn_loss)]
    train_loss = sum(train_loss)

    train_loss.backward()
    model.optimizer.step()

    return train_loss, snn_loss, bce_loss


def SNNL(
    features: object,
    labels: object,
    distance: str = "cosine",
    temperature: int = 100.0,
) -> float:
    """
    Computes the Soft Nearest Neighbors Loss (Fross, Papernot, & Hinton, 2019)
    https://arxiv.org/abs/1902.01889/
    Parameters
    ----------
    features : array-like object
        The input features.
    labels : array-like object
        The input labels.
    distance : str
        The distance metric to use.
    temperature : int
        The temperature factor.
    Returns
    -------
    float
        The soft nearest neighbors loss across all layers of a model.
    """
    distance = distance.lower()

    stability_epsilon = 1e-5
    if distance == "cosine":
        summed_masked_pick_probability = torch.sum(
            masked_pick_probability(
                features, labels, temperature, True, stability_epsilon
            ),
            dim=1,
        )
    elif distance == "euclidean":
        summed_masked_pick_probability = torch.sum(
            masked_pick_probability(
                features, labels, temperature, False, stability_epsilon
            ),
            dim=1,
        )
    snnl = torch.mean(-torch.log(stability_epsilon + summed_masked_pick_probability))
    return snnl


def masked_pick_probability(
    features, labels, temperature, cosine_distance, stability_epsilon
):
    """
    Returns the pairwise sampling probabilities for the `feature` elements
    for neighbor points sharing the same labels.
    Parameters
    ----------
    features : array-like object
        The input features, it may be the raw features or hidden activations.
    labels : array-like object
        The input labels.
    temperature : float
        The temperature constant.
    cosine_distance : bool
        Boolean whether to use cosine or Euclidean distance.
    stability_epsilon : float
        A constant for making the calculation for SNNL more stable.
    Returns
    -------
    tensor
        A tensor for pairwise sampling probabilities.
    """
    return pick_probability(
        features, temperature, cosine_distance, stability_epsilon
    ) * same_label_mask(labels, labels)


def pick_probability(features, temperature, cosine_distance, stability_epsilon=1e-5):
    """
    Returns a row normalized pairwise distance between all elements of `features`.
    Parameters
    ----------
    features : matrix
        The input features.
    temperature : float
        The temperature constant.
    cosine_distance : bool
        Boolean whether to use cosine or Euclidean distance.
    stability_epsilon : float
        The stability constant for SNNL.
    Returns
    -------
    normalized_pairwise_distance : matrix
        The normalized pairwise distance among `features`.
    """
    pairwise_distance = compute_pairwise_distance(
        features, features, temperature, cosine_distance
    )
    pairwise_distance -= torch.eye(features.shape[0])
    normalized_pairwise_distance = pairwise_distance / (
        stability_epsilon + torch.sum(pairwise_distance, 1).view(-1, 1)
    )
    return normalized_pairwise_distance


def same_label_mask(labels_a: torch.Tensor, labels_b: torch.Tensor) -> torch.Tensor:
    """
    Returns a masking matrix such that element (i, j) is 1
    iff labels[i] == labels_2[i].
    Parameters
    ----------
    labels_a, labels_b : array-like object
        The input labels.
    Returns
    -------
    masking_matrix : tensor
        The masking matrix, indicates whether labels are equal.
    """
    masking_matrix = torch.squeeze(torch.eq(labels_a, labels_b.view(-1, 1)).float())
    return masking_matrix


def compute_pairwise_distance(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    temperature: int,
    cosine_distance: bool,
) -> torch.Tensor:
    """
    Returns the exponentiated pairwise distance between each element of
    `features_a` and all those of `features_b`.

    Parameters
    ----------
    features_a, features_b : tensor
        The input features.
    temperature : float
        The temperature constant.
    cosine_distance : bool
        Boolean whether to use cosine or Euclidean distance.

    Returns
    -------
    tensor
        The exponentiated pairwise distance between `features_a` and `features_b`.
    """
    if cosine_distance:
        distance_matrix = pairwise_cosine_distance(features_a, features_b)
    else:
        distance_matrix = pairwise_euclidean_distance(features_a, features_b)
    return torch.exp(-(distance_matrix / temperature))


def pairwise_euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Returns the pairwise Euclidean distance between matrices `a` and `b`.

    Parameters
    ----------
    a, b : torch.Tensor
        The input features.

    Returns
    -------
    torch.Tensor
        The pairwise Euclidean distance between matrices `a` and `b`.

    Example
    -------
    >>> import torch
    >>> _ = torch.manual_seed(42)
    >>> a = torch.rand((4, 2))
    >>> b = torch.rand((4, 2))
    >>> pairwise_euclidean_distance(a, b)
    tensor([[0.6147, 0.9937, 0.5216, 0.9043],
            [0.1061, 0.4382, 0.2962, 0.4997],
            [0.1208, 0.3901, 0.2305, 0.4266],
            [0.2557, 0.4091, 0.1524, 0.3674]])
    """
    a = torch.FloatTensor(a)
    b = torch.FloatTensor(b)
    batch_size_a = a.size()[0]
    batch_size_b = b.size()[0]
    squared_norm_a = torch.sum(torch.pow(a, 2), dim=1)
    squared_norm_a = torch.reshape(squared_norm_a, [1, batch_size_a])
    squared_norm_b = torch.sum(torch.pow(b, 2), dim=1)
    squared_norm_b = torch.reshape(squared_norm_b, [batch_size_b, 1])

    a = a.T
    inner_product = torch.matmul(b, a)

    tile_a = (
        squared_norm_a.view(-1, 1)
        .repeat(batch_size_b, 1)
        .view((batch_size_b * squared_norm_a.size()[0]), -1)
    )
    tile_b = squared_norm_b.view(-1, 1).repeat(1, batch_size_a)
    return tile_a + tile_b - 2.0 * inner_product


def pairwise_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Returns the pairwise cosine distance between matrices `a` and `b`.

    Parameters
    ----------
    a, b : matrix
        The input features.

    Returns
    -------
    matrix
        The pairwise cosine distance between `a` and `b`.

    Example
    -------
    >>> import torch
    >>> _ = torch.manual_seed(42)
    >>> a = torch.rand((4, 2))
    >>> b = torch.rand((4, 2))
    >>> pairwise_cosine_distance(a, b)
    tensor([[0.2118, 0.0281, 0.0252, 0.0385],
            [0.5028, 0.1892, 0.1818, 0.2136],
            [0.3430, 0.0905, 0.0853, 0.1082],
            [0.5621, 0.2302, 0.2222, 0.2568]])
    """
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
    normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
    normalized_b = torch.conj(normalized_b).T
    product = torch.matmul(normalized_a, normalized_b)
    return 1.0 - product
