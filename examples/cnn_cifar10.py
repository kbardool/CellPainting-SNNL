from typing import Tuple

import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pass

    def fit(
        self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 2
    ):
        pass

    def epoch_train(
        self, data_loader: torch.utils.data.DataLoader, epoch: int = None
    ) -> Tuple:
        pass

    def predict(
        self, features: torch.Tensor
    ) -> torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
        pass
