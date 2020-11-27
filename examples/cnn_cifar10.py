from typing import Tuple

import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pass
