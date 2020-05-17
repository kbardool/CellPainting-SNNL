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
from snnl import masked_pick_probability
from snnl import same_label_mask
from snnl import SNNL
from snnl import pairwise_euclidean_distance
from snnl import pairwise_cosine_distance
from snnl import pick_probability


torch.manual_seed(42)


def test_SNNL():
    features = torch.rand((4, 2))
    labels = torch.ones((4, 1))
    snn_loss = SNNL(
        features=features, labels=labels, distance="cosine", temperature=100
    )
    assert isinstance(snn_loss, float)
    snn_loss = SNNL(
        features=features, labels=labels, distance="euclidean", temperature=100
    )
    assert isinstance(snn_loss, float)


def test_masked_pick_probability():
    features = torch.rand((4, 2))
    labels = torch.ones((4, 1))
    masked_probability_matrix = masked_pick_probability(
        features=features,
        labels=labels,
        temperature=10.0,
        cosine_distance=True,
        stability_epsilon=1e-5,
    )
    assert masked_probability_matrix.size() == (4, 4)
    assert isinstance(masked_probability_matrix, torch.Tensor)


def test_pick_probability():
    a = torch.rand((4, 2))
    probability_matrix = pick_probability(
        features=a, temperature=10, cosine_distance=True
    )
    assert probability_matrix.size() == (4, 4)
    assert isinstance(probability_matrix, torch.Tensor)


def test_same_label_mask():
    labels = torch.ones((4, 1))
    masking_matrix = same_label_mask(labels_a=labels, labels_b=labels)
    assert masking_matrix.size() == (4, 4)
    assert isinstance(masking_matrix, torch.Tensor)


def test_compute_pairwise_distance():
    a = torch.rand((4, 2))
    b = torch.rand((4, 2))
    temperature = 10
    pairwise_distance = compute_pairwise_distance(a, b, temperature, True)
    assert pairwise_distance.size() == (4, 4)
    assert isinstance(pairwise_distance, torch.Tensor)
    pairwise_distance = compute_pairwise_distance(a, b, temperature, False)
    assert pairwise_distance.size() == (4, 4)
    assert isinstance(pairwise_distance, torch.Tensor)


def test_pairwise_euclidean_distance():
    a = torch.rand((4, 2))
    b = torch.rand((4, 2))
    distance = pairwise_euclidean_distance(a, b)
    assert distance.size() == (4, 4)
    assert isinstance(distance, torch.Tensor)


def test_pairwise_cosine_distance():
    a = torch.rand((4, 2))
    b = torch.rand((4, 2))
    distance = pairwise_cosine_distance(a, b)
    assert distance.size() == (4, 4)
    assert isinstance(distance, torch.Tensor)
