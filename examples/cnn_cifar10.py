from typing import Tuple

from pt_datasets import create_dataloader, load_dataset
import torch

from snnl import SNNLoss


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

        for layer in list(self.children())[:-1]:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        logits: torch.Tensor
            The output of the model.
        """
        activations = dict()
        for index, layer in enumerate(self.children()):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations.get(index - 1))
        logits = activations.get(len(activations) - 1)
        return logits


train_data, test_data = load_dataset("cifar10")
train_loader = create_dataloader(train_data, batch_size=256)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN()
model = model.to(device)
model.device = device
optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
snnl_criterion = SNNLoss()
epochs = 10

for epoch in range(epochs):
    epoch_loss, epoch_xent, epoch_snnl, epoch_accuracy = 0, 0, 0, 0
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss, xent_loss, snn_loss = snnl_criterion(
            outputs=outputs,
            model=model,
            features=batch_features,
            labels=batch_labels,
            epoch=epoch,
        )
        train_loss.backward()
        optimizer.step()
        epoch_loss += train_loss.item()
        epoch_xent += xent_loss.item()
        epoch_snnl += snn_loss.item()
        train_accuracy = (outputs.argmax(1) == batch_labels).sum().item()
        train_accuracy /= len(batch_labels)
        epoch_accuracy += train_accuracy
    epoch_loss /= len(train_loader)
    epoch_xent /= len(train_loader)
    epoch_snnl /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    if (epoch + 1) % 2 == 0:
        print(f"epoch {epoch + 1}/{epochs}")
        print(f"\tmean loss = {epoch_loss:.4f}\t|\tmean acc = {epoch_accuracy:.4f}")
        print(f"\tmean xent = {epoch_xent:.4f}\t|\tmean snnl = {epoch_snnl:.4f}")
