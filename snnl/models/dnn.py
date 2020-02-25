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

    def forward(self, features):
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = torch.nn.ReLU(layer(features))
            elif index == len(activations) - 1:
                activations[index] = layer(activations[index - 1])
            else:
                activations[index] = torch.nn.ReLU(layer(activations[index - 1]))
        logits = activations[len(activations) - 1]
        return logits
