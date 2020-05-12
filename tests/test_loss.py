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
"""Test module"""
import torch
from snnl import compute_pairwise_distance
from snnl import pairwise_euclidean_distance
from snnl import pairwise_cosine_distance

torch.manual_seed(42)


def test_compute_pairwise_distance():
    a = torch.rand((4, 2))
    b = torch.rand((4, 2))
    temperature = 10
    pairwise_distance = compute_pairwise_distance(a, b, temperature, True)
    assert pairwise_distance.size() == (4, 4)
    assert type(pairwise_distance) is torch.Tensor


def test_pairwise_euclidean_distance():
    a = torch.rand((4, 2))
    b = torch.rand((4, 2))
    distance = pairwise_euclidean_distance(a, b)
    assert distance.size() == (4, 4)
    assert type(distance) is torch.Tensor


def test_pairwise_cosine_distance():
    a = torch.rand((4, 2))
    b = torch.rand((4, 2))
    distance = pairwise_cosine_distance(a, b)
    assert distance.size() == (4, 4)
    assert type(distance) is torch.Tensor
