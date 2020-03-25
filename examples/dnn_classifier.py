"""Sample module for using DNN classifier with SNNL"""
from snnl.models.dnn import DNN
from snnl.utils.data import create_dataloader, load_dataset
from snnl.utils.metrics import accuracy

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def main():
    units = ([784, 512], [512, 10])
    learning_rate = 1e-2
    batch_size = 512
    epochs = 40

    train_dataset, test_dataset = load_dataset(name="mnist")
    train_loader = create_dataloader(dataset=train_dataset, batch_size=batch_size)
    test_loader = create_dataloader(dataset=test_dataset, batch_size=batch_size)

    model = DNN(units=units, learning_rate=learning_rate)
    model.fit(data_loader=train_loader, epochs=epochs, use_snnl=True, factor=10)
    acc = accuracy(model, test_loader)
    print(f"accuracy: {acc * 100.}%")


if __name__ == "__main__":
    main()
