import torch
import torchvision


__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class ResNet18(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.resnet = torchvision.models.resnet.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=num_classes
        )
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, features):
        return self.resnet.forward(features)
