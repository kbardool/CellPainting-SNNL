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
"""Implementation of ResNet18 and ResNet34 models"""
from typing import Tuple
import torch
import torchvision

from snnl import SNNL

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class ResNet(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train_loss = []
