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
"""Module for metrics"""
import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def accuracy(y_true, y_pred) -> float:
    """
    Returns the classification accuracy of the model.

    Parameters
    ----------
    model: torch.nn.Module
        The classification model to test.
    dataset: torch.utils.dataloader.DataLoader
        The dataset to use for inference.

    Returns
    -------
    float
        The classification accuracy of the model.
    """
    correct = (y_pred == y_true).sum().item()
    accuracy = correct / len(y_true)
    accuracy *= 100.0
    return accuracy
