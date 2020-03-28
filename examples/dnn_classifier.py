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
