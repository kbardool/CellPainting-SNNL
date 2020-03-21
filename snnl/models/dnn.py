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
"""Implementation of a feed-forward neural network"""
import torch


class DNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=in_features, out_features=out_features)
                for in_features, out_features in kwargs["units"]
            ]
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(),
                lr=kwargs["learning_rate"])
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, features):
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = torch.relu(layer(features))
            elif index == len(self.layers) - 1:
                activations[index] = layer(activations[index - 1])
            else:
                activations[index] = torch.nn.relu(layer(activations[index - 1]))
        logits = activations[len(activations) - 1]
        return logits


def epoch_train(model, data_loader):
    epoch_loss = 0
    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.view(batch_features.shape[0], -1)
        model.optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = model.criterion(outputs, batch_labels)
        train_loss.backward()
        model.optimizer.step()
        epoch_loss += train_loss.item()
    epoch_loss /= len(data_loader)
    return epoch_loss
