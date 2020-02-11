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
        .view((batch_size_a * squared_norm_b.size()[0]), -1)
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
    tensor([[0.7569, 0.6507, 0.6715, 0.7327],
            [0.8754, 0.7632, 0.7759, 0.8224],
            [0.8858, 0.8157, 0.8262, 0.8602],
            [0.9114, 0.8185, 0.8280, 0.8644]])
    """
    normalized_a = torch.Tensor(a) / torch.norm(torch.Tensor(a), p=2)
    normalized_b = torch.Tensor(b) / torch.norm(torch.Tensor(b), p=2)
    normalized_b = torch.t(normalized_b)
    product = torch.matmul(normalized_a, normalized_b)
    return 1.0 - product
