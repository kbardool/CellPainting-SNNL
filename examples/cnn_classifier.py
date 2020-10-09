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
"""Sample module for using CNN classifier with SNNL"""
import argparse
import torch

from snnl.models import CNN
from snnl.utils import get_hyperparameters, set_global_seed
from snnl.utils.data import create_dataloader, load_dataset
from snnl.utils.metrics import accuracy

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def parse_args():
    parser = argparse.ArgumentParser(description="CNN classifier with SNNL")
    group = parser.add_argument_group("Parameters")
    group.add_argument(
        "-s",
        "--seed",
        required=False,
        default=1234,
        type=int,
        help="the random seed value to use, default: [1234]",
    )
    group.add_argument(
        "-m",
        "--model",
        required=False,
        default="baseline",
        type=str,
        help="the model to use, options: [baseline (default) | snnl]",
    )
    group.add_argument(
        "-c",
        "--configuration",
        required=False,
        default="examples/hyperparameters/cnn.json",
        type=str,
        help="the path to the JSON file containing the hyperparameters to use",
    )
    arguments = parser.parse_args()
    return arguments


def main(args):
    (
        dataset,
        batch_size,
        epochs,
        learning_rate,
        input_dim,
        num_classes,
        snnl_factor,
        temperature,
    ) = get_hyperparameters(args.configuration)

    set_global_seed(args.seed)

    if args.device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    train_dataset, test_dataset = load_dataset(name=dataset)
    train_loader = create_dataloader(dataset=train_dataset, batch_size=batch_size)

    model = CNN(
        input_dim=input_dim,
        num_classes=num_classes,
        learning_rate=learning_rate,
        device=device,
    )
    if args.model.lower() == "baseline":
        model.fit(data_loader=train_loader, epochs=epochs)
    elif args.model.lower() == "snnl":
        model.fit(
            data_loader=train_loader,
            epochs=epochs,
            use_snnl=True,
            factor=snnl_factor,
            temperature=temperature,
        )
    else:
        raise ValueError("Choose between [baseline] and [snnl] only.")
    test_features = test_dataset.data.reshape(-1, 1, 28, 28) / 255.0
    model.eval()
    model = model.cpu()
    predictions = model.predict(test_features)
    test_accuracy = accuracy(y_true=test_dataset.targets, y_pred=predictions)
    print(f"accuracy: {test_accuracy}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)
