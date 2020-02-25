import torch


class Autoencoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=kwargs["input_shape"], out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=2000),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2000, out_features=kwargs["code_dim"]),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=kwargs["code_dim"], out_features=2000),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2000, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=kwargs["input_shape"]),
            torch.nn.Sigmoid()
            ])

    def forward(self, features):
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations[index - 1])
        reconstruction = activations[len(activations) - 1]
        return reconstruction
