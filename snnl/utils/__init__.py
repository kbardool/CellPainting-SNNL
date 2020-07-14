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
import random
from typing import Tuple

import numpy as np
import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def set_global_seed(seed: int) -> None:
    """
    Sets the seed value for random number generators.

    Parameter
    ---------
    seed : int
        The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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
        input_shape : int
            The dimensionality of flattened input features.
        code_dim : int
            The dimensionality of the latent code.
        snnl_factor : int or float
            The SNNL factor.
        temperature : int
            The soft nearest neighbor loss temperature factor.
            If temperature == 0, use annealing temperature.
    """
    with open(hyperparameters_path, "r") as file:
        config = json.load(file)

    dataset = config.get("dataset")
    assert isinstance(dataset, str), "[dataset] must be [str]."

    batch_size = config.get("batch_size")
    assert isinstance(batch_size, int), "[batch_size] must be [int]."

    epochs = config.get("epochs")
    assert isinstance(epochs, int), "[epochs] must be [int]."

    learning_rate = config.get("learning_rate")
    assert isinstance(learning_rate, float), "[learning_rate] must be [float]."

    snnl_factor = config.get("snnl_factor")
    assert isinstance(snnl_factor, float) or isinstance(
        snnl_factor, int
    ), "[snnl_factor] must be either [float] or [int]."

    temperature = config.get("temperature")
    assert isinstance(temperature, int), "[temperature] must be [int]."
    if temperature == 0:
        temperature = None

    hyperparameters_filename = os.path.basename(hyperparameters_path)
    hyperparameters_filename = hyperparameters_filename.lower()
    if "dnn" in hyperparameters_filename:
        units = config.get("units")
        assert isinstance(units, list), "[units] must be [list]."
        assert len(units) >= 2, "len(units) must be >= 2."
        return (
            dataset,
            batch_size,
            epochs,
            learning_rate,
            units,
            snnl_factor,
            temperature,
        )
    elif "cnn" in hyperparameters_filename:
        input_dim = config.get("input_dim")
        assert isinstance(input_dim, int), "[input_dim] must be [int]."

        num_classes = config.get("num_classes")
        assert isinstance(num_classes, int), "[num_classes] must be [int]."

        return (
            dataset,
            batch_size,
            epochs,
            learning_rate,
            input_dim,
            num_classes,
            snnl_factor,
            temperature,
        )
    elif "autoencoder" in hyperparameters_filename:
        input_shape = config.get("input_shape")
        assert isinstance(input_shape, int), "[input_shape] must be [int]."

        code_dim = config.get("code_dim")
        assert isinstance(code_dim, int), "[code_dim] must be [int]."

        return (
            dataset,
            batch_size,
            epochs,
            learning_rate,
            input_shape,
            code_dim,
            snnl_factor,
            temperature,
        )
    elif "resnet" in hyperparameters_filename:
        return (dataset, batch_size, epochs, learning_rate, snnl_factor, temperature)


def export_results(model, filename):
    """
    Exports the training results stored in model class to a JSON file.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model object.
    filename : str
        The filename of the JSON file to write.
    """
    model_attributes = model.__dict__
    results = dict()
    for key, value in model_attributes.items():
        if isinstance(value, list):
            results[key] = value
    with open(filename, "w") as file:
        json.dump(results, file)


def save_model(model: torch.nn.Module, filename: str):
    pass
