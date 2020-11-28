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
import argparse

from pt_datasets import create_dataloader, load_dataset
from snnl.models import DNN
from snnl.utils import export_results, get_hyperparameters, set_global_seed

# from snnl.utils.data import create_dataloader, load_dataset
from snnl.utils.metrics import accuracy

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def parse_args():
    parser = argparse.ArgumentParser(description="DNN classifier with SNNL")
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
        default="examples/hyperparameters/dnn.json",
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
        units,
        snnl_factor,
        temperature,
    ) = get_hyperparameters(args.configuration)

    set_global_seed(args.seed)

    train_dataset, test_dataset = load_dataset(name=dataset)
    train_loader = create_dataloader(dataset=train_dataset, batch_size=batch_size)

    if args.model.lower() == "baseline":
        model = DNN(units=units, learning_rate=learning_rate)
    elif args.model.lower() == "snnl":
        model = DNN(
            units=units, learning_rate=learning_rate, use_snnl=True, factor=snnl_factor
        )
    else:
        raise ValueError("Choose between [baseline] and [snnl] only.")

    model.fit(data_loader=train_loader, epochs=epochs)

    test_features = test_dataset.data.reshape(-1, 784) / 255.0
    model.eval()
    model = model.cpu()
    predictions = model.predict(test_features)
    model.test_accuracy = accuracy(y_true=test_dataset.targets, y_pred=predictions)
    print(f"accuracy: {model.test_accuracy}%")
    filename = f"DNN-{args.model.lower()}-{args.seed}.json"
    export_results(model=model, filename=filename)


if __name__ == "__main__":
    args = parse_args()
    main(args)
