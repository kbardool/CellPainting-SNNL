from typing import Tuple

import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=8, padding=1, stride=2
        )
        self.activation1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=6, padding=1, stride=2
        )
        self.activation2 = torch.nn.ReLU(inplace=True)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=(128 * 6 * 6), out_features=1024)
        self.activation3 = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=1024)
        self.activation4 = torch.nn.ReLU(inplace=True)
        self.fc3 = torch.nn.Linear(in_features=1024, out_features=512)
        self.activation5 = torch.nn.ReLU(inplace=True)
        self.output_layer = torch.nn.Linear(in_features=512, out_features=10)

        for layer in list(self.children()[-1]):
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        activations = dict()
        for index, layer in enumerate(self.children()):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations.get(index - 1))
        logits = activations.get(len(activations) - 1)
        return logits
