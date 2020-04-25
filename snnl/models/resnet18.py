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
        self.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, features):
        return self.resnet.forward(features)

    def fit(self, data_loader, epochs, use_snnl=False, factor=None, temperature=None):
        if use_snnl:
            assert factor is not None, "[factor] must not be None if use_snnl == True"
            train_snn_loss = []
            train_xent_loss = []

        for epoch in range(epochs):
            epoch_loss = epoch_train(
                self.resnet, data_loader, epoch, use_snnl, factor, temperature
            )
            if "cuda" in self.device.type:
                torch.cuda.empty_cache()

            if type(epoch_loss) is tuple:
                self.train_loss.append(epoch_loss[0])
                train_snn_loss.append(epoch_loss[1])
                train_xent_loss.append(epoch_loss[2])
                print(
                    f"epoch {epoch + 1}/{epoch} : mean loss = {self.train_loss[-1]:.6f}"
                )
                print(
                    f"\txent loss = {train_xent_loss[-1]:.6f}\t|\tsnn loss = {train_snn_loss[-1]:.6f}"
                )
            else:
                self.train_loss.append(epoch_loss)
                print(
                    f"epoch {epoch + 1}/{epoch} : mean loss = {self.train_loss[-1]:.6f}"
                )
