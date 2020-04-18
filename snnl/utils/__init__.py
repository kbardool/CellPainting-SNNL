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
"""Utility functions"""
import json
import os
from typing import Tuple

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def get_hyperparameters(hyperparameters_path: str) -> Tuple:
    """
    Returns hyperparameters from JSON file.

    Parameters
    ----------
    hyperparameters_path : str
        The path to the hyperparameters JSON file.

    Returns
    -------
    Tuple
        dataset : str
            The name of the dataset to use.
        batch_size : int
            The mini-batch size.
        epochs : int
            The number of training epochs.
        learning_rate : float
            The learning rate to use for optimization.
        units : list
            The list of units per hidden layer if using [dnn].
        input_dim : int
            The dimensionality of the input feature channel.
        num_classes : int
            The number of classes in a dataset.
        snnl_factor : int or float
            The SNNL factor.
        temperature_mode : str
            Use annealing or fixed temperature.
    """
    with open(hyperparameters_path, "r") as file:
        config = json.load(file)

    dataset = config["dataset"]
    assert isinstance(dataset, str), "[dataset] must be [str]."

    batch_size = config["batch_size"]
    assert isinstance(batch_size, int), "[batch_size] must be [int]."

    epochs = config["epochs"]
    assert isinstance(epochs, int), "[epochs] must be [int]."

    learning_rate = config["learning_rate"]
    assert isinstance(learning_rate, float), "[learning_rate] must be [float]."

    snnl_factor = config["snnl_factor"]
    assert isinstance(snnl_factor, float) or isinstance(
        snnl_factor, int
    ), "[snnl_factor] must be either [float] or [int]."

    temperature_mode = config["temperature_mode"]
    assert isinstance(temperature_mode, str), "[temperature_mode] must be [str]."

    hyperparameters_filename = os.path.basename(hyperparameters_path)
    hyperparameters_filename = hyperparameters.lower()
    if "dnn" in hyperparameters_filename:
        units = config["units"]
        assert isinstance(units, list), "[units] must be [list]."
        assert len(units) >= 2, "len(units) must be >= 2."
        return (
            dataset,
            batch_size,
            epochs,
            learning_rate,
            units,
            snnl_factor,
            temperature_mode,
        )
    elif "cnn" in hyperparameters_filename:
        input_shape = config["input_dim"]
        assert isinstance(input_shape, int), "[input_shape] must be [int]."

        num_classes = config["num_classes"]
        assert isinstance(num_classes, int), "[num_classes] must be [int]."

        return (
            dataset,
            batch_size,
            epochs,
            learning_rate,
            input_shape,
            num_classes,
            snnl_factor,
            temperature_mode,
        )
    elif "autoencoder" in hyperparameters_filename:
        input_shape = config["input_shape"]
        assert isinstance(input_shape, int), "[input_shape] must be [int]."

        code_dim = config["code_dim"]
        assert isinstance(code_dim, int), "[code_dim] must be [int]."

        return (
            dataset,
            batch_size,
            epochs,
            learning_rate,
            input_shape,
            code_dim,
            snnl_factor,
            temperature_mode,
        )
