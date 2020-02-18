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


def pick_probability(
        features, temperature, cosine_distance, stability_epsilon=1e-5
        ):
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
    pairwise_distance -= torch.eye(features.size()[0])
    normalized_pairwise_distance = pairwise_distance / (
            stability_epsilon
            + torch.sum(pairwise_distance, 1).view(-1, 1)
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
    masking_matrix = torch.tensor(
            torch.squeeze(
                torch.eq(labels_a, labels_b.view(-1, 1))
                ),
            torch.float32
            )
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
    """
    batch_size_a = a.size()[0]
    batch_size_b = b.size()[0]
    squared_norm_a = torch.sum(torch.pow(a, 2), dim=1)
    squared_norm_a = torch.reshape(squared_norm_a, [1, batch_size_a])
    squared_norm_b = torch.sum(torch.pow(b, 2), dim=1)
    squared_norm_b = torch.reshape(squared_norm_b, [batch_size_b, 1])

    a = torch.t(a)
    inner_product = torch.matmul(b, a)

    tile_a = (
        squared_norm_a.view(-1, 1)
        .repeat(batch_size_b, 1)
        .view((batch_size_b * squared_norm_a.size()[0]), -1)
    )
    tile_b = (
        squared_norm_b.view(-1, 1)
        .repeat(1, batch_size_a)
    )
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
