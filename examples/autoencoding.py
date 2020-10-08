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
"""Sample module for using Autoencoder with SNNL"""
import argparse
import torch

from snnl.models import Autoencoder
from snnl.utils import get_hyperparameters, set_global_seed
from snnl.utils.data import load_dataset, create_dataloader

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder with SNNL")
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
        default="examples/hyperparameters/autoencoder.json",
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
        input_shape,
        code_dim,
        snnl_factor,
        temperature,
    ) = get_hyperparameters(args.configuration)

    set_global_seed(args.seed)

    train_dataset, test_dataset = load_dataset(name=dataset)
    train_features = train_dataset.data.numpy().astype("float32") / 255.0
    train_features = train_features[:10000]
    train_features = torch.from_numpy(train_features)
    train_labels = train_dataset.targets
    train_labels = train_labels[:10000]
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = create_dataloader(
        dataset=train_dataset, batch_size=batch_size, num_workers=1
    )

    if args.model.lower() == "baseline":
        model = Autoencoder(
            input_shape=input_shape, code_dim=code_dim, learning_rate=learning_rate
        )
    elif args.model.lower() == "snnl":
        model = Autoencoder(
            input_shape=input_shape,
            code_dim=code_dim,
            learning_rate=learning_rate,
            use_snnl=True,
            factor=snnl_factor,
            temperature=temperature,
            mode="latent_code",
            code_units=30,
        )
    else:
        raise ValueError("Choose between [baseline] and [snnl] only.")
    model.fit(data_loader=train_loader, epochs=epochs, show_every=5)


if __name__ == "__main__":
    args = parse_args()
    main(args)
