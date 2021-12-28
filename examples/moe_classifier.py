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
"""Sample MoE classifier module"""
import argparse

import torch
from disentangling_moe.models import DNN, MoE
from disentangling_moe.utils import set_global_seed
from pt_datasets import create_dataloader, load_dataset

__author__ = "Abien Fred Agarap"


def parse_args():
    parser = argparse.ArgumentParser(description="MoE classifier with SNNL")
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
        "-e",
        "--num_experts",
        required=False,
        default=3,
        type=int,
        help="the number of experts to use, default: [3]",
    )
    group.add_argument(
        "-f",
        "--factor",
        required=False,
        default=100.0,
        type=float,
        help="the SNNL balance factor, default: [100]",
    )
    group.add_argument(
        "-t",
        "--temperature",
        required=False,
        default=50.0,
        type=float,
        help="the SNNL temperature value, default: [50]",
    )
    group.add_argument(
        "-i",
        "--epochs",
        required=False,
        default=15,
        type=int,
        help="the number of epochs to train the model, default: [15]",
    )
    group.add_argument(
        "-d",
        "--dataset",
        required=False,
        default="mnist",
        type=str,
        help="the dataset to use",
    )
    group.add_argument(
        "-x",
        "--feature_extractor",
        required=False,
        dest="feature_extractor",
        action="store_true",
    )
    group.add_argument(
        "-v",
        "--show_interval",
        required=False,
        default=2,
        type=int,
        help="the interval between displays of training progress, default: [2]",
    )
    group.set_defaults(feature_extractor=False)
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    set_global_seed(arguments.seed)
    train_data, test_data = load_dataset(arguments.dataset)
    train_loader = create_dataloader(train_data, batch_size=256)
    test_loader = create_dataloader(test_data, batch_size=10000)

    if not arguments.feature_extractor:
        expert_model = DNN(units=((784, 500), (500, 500), (500, 10)))
    else:
        expert_model = DNN(units=((30, 512), (512, 10)))
    gating_model = DNN(units=((784, 512), (512, arguments.num_experts)))
    if arguments.model == "snnl":
        model = MoE(
            input_shape=784,
            expert_model=expert_model,
            gating_model=gating_model,
            num_experts=arguments.num_experts,
            output_dim=10,
            use_snnl=True,
            factor=arguments.factor,
            temperature=arguments.temperature,
            learning_rate=1e-4,
        )
    elif arguments.model == "baseline":
        model = MoE(
            input_shape=784,
            expert_model=expert_model,
            gating_model=gating_model,
            num_experts=arguments.num_experts,
            use_feature_extractor=arguments.feature_extractor,
            output_dim=10,
            use_snnl=False,
            learning_rate=1e-4,
        )
    model.fit(
        data_loader=train_loader,
        epochs=arguments.epochs,
        show_every=arguments.show_interval,
    )
    with torch.no_grad():
        model = model.cpu()
        model = model.eval()
        for test_features, test_labels in test_loader:
            outputs = model(test_features)
            correct = (outputs.argmax(1) == test_labels).sum().item()
            accuracy = correct / len(test_labels)
    print(f"Test accuracy = {accuracy:.4f}")


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
