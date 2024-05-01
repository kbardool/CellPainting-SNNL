# Soft Nearest Neighbor Lossdisply_
# Copyright (C) 2020-2024  Abien Fred Agarap
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
"""Utility functions / Metrics """
import sys
import json
import os
import random
import argparse
from typing import List, Tuple
from torchinfo import summary
import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score
__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"



def get_device():
    devices = torch.cuda.device_count()
    gb = 2**30
    for i in range(devices):
        free, total = torch.cuda.mem_get_info(i)
        print(f" device: {i}   {torch.cuda.get_device_name(i):30s} :  free: {free:,d} B   ({free/gb:,.2f} GB)    total: {total:,d} B   ({total/gb:,.2f} GB)")
    # device = 
    device = f"{'cuda' if torch.cuda.is_available() else 'cpu'}:{torch.cuda.current_device()}"
    print(" Current CUDA Device is:", device, torch.cuda.get_device_name(), torch.cuda.current_device())
    return device

def set_device(device_id):
    print(" Running on:",  torch.cuda.get_device_name(), torch.cuda.current_device())
    devices = torch.cuda.device_count()
    assert device_id < devices, f"Invalid device id, must be less than {devices}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = f"{device}:{device_id}"
    print(f" Switch to {device} ")
    torch.cuda.set_device(device_id)
    print(" Running on:",  torch.cuda.get_device_name(), torch.cuda.current_device())
    return device
    
def parse_args(input = None):
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
    arguments = parser.parse_args(input)
    return arguments


def set_global_seed(seed: int) -> None:
    """
    Sets the seed value for random number generators.

    Parameter
    ---------
    seed: int
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
    hyperparameters_path: str
        The path to the hyperparameters JSON file.

    Returns
    -------
    Tuple
        dataset: str
            The name of the dataset to use.
        batch_size: int
            The mini-batch size.
        epochs: int
            The number of training epochs.
        learning_rate: float
            The learning rate to use for optimization.
        units: List
            The list of units per hidden layer if using [dnn].
        image_dim: int
            The dimensionality of the image feature [W, H]
            such that W == H.
        input_dim: int
            The dimensionality of the input feature channel.
        num_classes: int
            The number of classes in a dataset.
        input_shape: int
            The dimensionality of flattened input features.
        code_dim: int
            The dimensionality of the latent code.
        snnl_factor: int or float
            The SNNL factor.
        temperature: int
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
    assert isinstance(temperature, float), "[temperature] must be [float]."
    if temperature == 0:
        temperature = None

    hyperparameters_filename = os.path.basename(hyperparameters_path)
    hyperparameters_filename = hyperparameters_filename.lower()

    
    if "cellpainting" in hyperparameters_filename:
        print("common cellpainting hyperparameters")
        cellpainting_args = config.get("cellpainting_args")
        assert isinstance(cellpainting_args, dict), "[cellpainting_args] must be [dict]."
    
    if "dnn" in hyperparameters_filename:
        print("loading dnn hyperparameters")
        units = config.get("units")
        assert isinstance(units, List), "[units] must be [List]."
        assert len(units) >= 2, "len(units) must be >= 2."
        activations = config.get("activations")
        assert isinstance(activations, List), "[activations] must be [List]."
        assert len(activations) >= 2, "len(activations) must be >= 2."
        
        if "dnn_cellpainting" in hyperparameters_filename:
            print("loading dnn cellpainting hyperparameters")
            # cellpainting_args = config.get("cellpainting_args")
            # assert isinstance(cellpainting_args, dict), "[cellpainting_args] must be [dict]." 
            return (
                dataset,
                batch_size,
                epochs,
                learning_rate,
                units,
                activations,
                snnl_factor,
                temperature,
                cellpainting_args
            )
        else:
            print("load other non-dnn hyperparms")
            return (
                dataset,
                batch_size,
                epochs,
                learning_rate,
                units,
                activations,
                snnl_factor,
                temperature,
            )
    elif "cnn" in hyperparameters_filename:
        image_dim = config.get("image_dim")
        assert isinstance(image_dim, int), "[image_dim] must be [int]."

        input_dim = config.get("input_dim")
        assert isinstance(input_dim, int), "[input_dim] must be [int]."

        num_classes = config.get("num_classes")
        assert isinstance(num_classes, int), "[num_classes] must be [int]."

        return (
            dataset,
            batch_size,
            epochs,
            learning_rate,
            image_dim,
            input_dim,
            num_classes,
            snnl_factor,
            temperature,
        )
    elif "autoencoder" in hyperparameters_filename:
        print("loading autoencoder hyperparameters")

        input_shape = config.get("input_shape")
        assert isinstance(input_shape, int), "[input_shape] must be [int]."

        code_units = config.get("code_units")
        assert isinstance(code_units, int), "[code_units] must be [int]."
        
        units = config.get("units")
        assert isinstance(units, List), "[units] must be [List]."
        assert len(units) >= 2, "len(units) must be >= 2."

        activations = config.get("activations")
        assert isinstance(activations, List), "[activations] must be [List]."
        assert len(activations) >= 2, "len(activations) must be >= 2."
        assert len(activations) == len(units), "len(activations) must be equal to len(units) - use none if corresponding layer has no non-linearity"
        
        if "cellpainting" in hyperparameters_filename:
            print("loading autoencoder_cellpainting hyperparms")
            return (
                dataset,
                batch_size,
                epochs,
                learning_rate,
                input_shape,
                code_units,
                units,
                activations,
                snnl_factor,
                temperature,
                cellpainting_args
            )
        else:
            return (
                dataset,
                batch_size,
                epochs,
                learning_rate,
                input_shape,
                code_units,
                units,
                snnl_factor,
                temperature,
            )        
    elif "resnet" in hyperparameters_filename:
        return (dataset, batch_size, epochs, learning_rate, snnl_factor, temperature)


#-------------------------------------------------------------------------------------------------------------------
#  Import and Export routines
#-------------------------------------------------------------------------------------------------------------------     

def export_results(model: torch.nn.Module, filename: str):
    """
    Exports the training results stored in model class to a JSON file.

    Parameters
    ----------
    model: torch.nn.Module
        The trained model object.
    filename: str
        The filename of the JSON file to write.
    """
    model_attributes = model.__dict__
    results = dict()
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    filename = os.path.join(results_dir, f"{filename}.json")
    for key, value in model_attributes.items():
        if isinstance(value, List) or "test_accuracy" in key:
            results[key] = value
    with open(filename, "w") as file:
        json.dump(results, file)
    print(f"[INFO] Model Results exported to {filename}.")


def import_results(filename: str):
    """
    Exports the training results stored in model class to a JSON file.

    Parameters
    ----------
    model: torch.nn.Module
        The trained model object.
    filename: str
        The filename of the JSON file to write.
    """
    # model_attributes = model.__dict__
    # results = dict()
    results_dir = "results"
    # if not os.path.exists(results_dir):
        # os.mkdir(results_dir)
    filename = os.path.join(results_dir, f"{filename}")
    # for key, value in model_attributes.items():
        # if isinstance(value, List) or "test_accuracy" in key:
            # results[key] = value
    with open(filename, "r") as file:
        results = json.load(file)
    return results


def save_model(model: torch.nn.Module, filename: str):
    """
    Exports the input model to the examples/export directory.

    Parameters
    ----------
    model: torch.nn.Module
        The (presumably) trained model object.
    filename: str
        The filename for the trained model to export.
    """
    path = os.path.join("examples", "export")
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, f"{filename}.pt")
    torch.save(model, path)
    print(f"[INFO] Model exported to {path}.")


def save_checkpoint(epoch, model, filename, update_latest=False, update_best=False):
    model_checkpoints_folder = os.path.join("ckpts")
    if not os.path.exists(model_checkpoints_folder):
        print(f"path {model_checkpoints_folder} doesn't exist")
    checkpoint = {'epoch': epoch,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': model.optimizer.state_dict()}
    
    if hasattr(model, 'scheduler'):
        checkpoint['scheduler']: model.scheduler
        
    if update_latest:
        filename = os.path.join(model_checkpoints_folder, f"{filename}_model_latest.pt")
    elif update_best:
        filename = os.path.join(model_checkpoints_folder, f"{filename}_model_best.pt")
    else:
        filename = os.path.join(model_checkpoints_folder, f"{filename}.pt")
    torch.save(checkpoint, filename) 
    print(f"[INFO] Model exported to {filename}.")


def load_model(filename: str) -> torch.nn.Module:
    """
    Exports the input model to the examples/export directory.

    Parameters
    ----------
    model: torch.nn.Module
        The (presumably) trained model object.
    filename: str
        The filename for the trained model to export.
    """
    path = os.path.join("examples", "export")
    if not os.path.exists(path):
        print(f"path {path} doesn't exist")
    path = os.path.join(path, filename)
    print(f"[INFO] Model imported from {path}.")
    return torch.load(path)


def load_checkpoint(model, filename ):
    epoch = 9999
    try:
        checkpoints_folder = os.path.join("ckpts")
        checkpoint = torch.load(os.path.join(checkpoints_folder, filename))
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if "scheduler" in checkpoint and (hasattr(model, 'scheduler')):
            model.scheduler = checkpoint['scheduler']
        epoch = checkpoint.get('epoch',0)
        print(f"\n ==> Loaded from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}\n")
         
    # except FileNotFoundError:
    #     Exception("Previous state checkpoint not found.")
    except :
        print(sys.exc_info())

    return model, epoch

#-------------------------------------------------------------------------------------------------------------------
#  Plotting routines
#-------------------------------------------------------------------------------------------------------------------     

def plot_train_history(model, epochs= None, n_bins = 25):
 
    if epochs is None:
        epochs = len(model.training_history['train']['loss'])
     
    fig, axs = plt.subplots(1, 5, sharey=False, tight_layout=True, figsize=(5*4,4) )
    labelsize = 6
    # We can set the number of bins with the *bins* keyword argument.
    i = 0
    _ = axs[i].plot(model.training_history['train']['xent_loss'][:epochs],label='Training');
    _ = axs[i].plot(model.training_history['val']['xent_loss'][:epochs],label='Validation');
    _ = axs[i].set_title(f'Cross Entropy loss - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    axs[i].legend()
    i +=1
    _ = axs[i].plot(model.training_history['train']['snn_loss'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['snn_loss'][:epochs]);
    _ = axs[i].set_title(f'SNN Loss - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    i +=1
    _ = axs[i].plot(model.training_history['train']['temp_hist'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['temp_hist'][:epochs]);
    _ = axs[i].set_title(f'train_temp_hist - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    i +=1
    batches = (len(model.training_history['train']['temp_grads']) // len(model.training_history['train']['loss'])) *epochs
    _ = axs[i].plot(model.training_history['train']['temp_grads'][:batches])
    _ = axs[i].set_title(f'Temperature Gradients - {epochs} epochs', fontsize= 10)
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    # i +=1
    # _ = axs[i].plot(model.training_history['train']['temp_grad_hist'][:epochs]);
    # _ = axs[i].set_title(f"Temperature Grad at end of epochs - {epochs} epochs", fontsize= 10);
    # _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    i +=1
    _ = axs[i].plot(model.training_history['train']['layer_grads'][:epochs]);
    _ = axs[i].set_title(f"Monitored layer gradient - {epochs} epochs", fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)    
    # i +=1
    # _ = axs[i].plot(model.training_history['train']['accuracy'][:epochs]);
    # _ = axs[i].plot(model.training_history['val']['accuracy'][:epochs]);
    # _ = axs[i].set_title(f'Accuracy - {epochs} epochs', fontsize= 10);
    # _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    # i +=1
    # _ = axs[i].plot(model.training_history['train']['f1'][:epochs]);
    # _ = axs[i].plot(model.training_history['val']['f1'][:epochs]);
    # _ = axs[i].set_title(f'F1 Score - {epochs} epochs', fontsize= 10);
    # _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    plt.show()

def plot_train_metrics(model, epochs= None, n_bins = 25):
 
    if epochs is None:
        epochs = len(model.training_history['train']['loss'])    
    fig, axs = plt.subplots(1, 5, sharey=False, tight_layout=True, figsize=(5*4,4) )
    i = 0
    _ = axs[i].plot(model.training_history['train']['accuracy'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['accuracy'][:epochs]);
    _ = axs[i].set_title(f'Accuracy - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    i +=1
    _ = axs[i].plot(model.training_history['train']['f1'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['f1'][:epochs]);
    _ = axs[i].set_title(f'F1 Score - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    i += 1
    _ = axs[i].plot(model.training_history['train']['precision'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['precision'][:epochs]);
    _ = axs[i].set_title(f' Precision - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    i +=1
    _ = axs[i].plot(model.training_history['train']['roc_auc'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['roc_auc'][:epochs]);
    _ = axs[i].set_title(f' ROC AUC Score - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    i +=1
    _ = axs[i].plot(model.training_history['train']['recall'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['recall'][:epochs]);
    _ = axs[i].set_title(f'Recall - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    i +=1
    plt.show()    
    
def plot_model_parms(model, epochs= None, n_bins = 25):
    
    weights = dict()
    biases = dict()
    grads = dict()
    layer_id = dict()
    i = 0
    for k, layer in enumerate(model.layers):
        if type(layer) == torch.nn.modules.linear.Linear:
            layer_id[i] =k
            weights[i] = layer.weight.cpu().detach().numpy()
            biases[i] = layer.bias.cpu().detach().numpy()
            grads[i] = layer.weight.grad.cpu().detach().numpy()
            i+=1
    num_layers = i
 
    
    print(f" +------+----------------------------------------------------+-------------------------------------------+------------------------------------+")
    print(f" |      | Weights:                                           |  Biases:                                  |   Gradients:                       |")
    print(f" | layr |                      min         max        stdev  |             min         max        stdev  |      min        max         stdev  |")
    print(f" +------+----------------------------------------------------+-------------------------------------------+------------------------------------+")
    for k in layer_id.keys():
        print(f" | {k:4d} | {str(weights[k].shape):15s}  {weights[k].min():9.6f}   {weights[k].max():9.6f}   {weights[k].std():9.6f}"
              f" |  {biases[k].shape[0]:5d}  {biases[k].min():9.6f}   {biases[k].max():9.6f}   {biases[k].std():9.6f}"
              f" |  {grads[k].min():9.6f}   {grads[k].max():9.6f}   {grads[k].std():9.6f} |")
    print(f" +------+----------------------------------------------------+-------------------------------------------+------------------------------------+")
    print('\n\n')
    
    fig, axs = plt.subplots(3, num_layers, sharey=False, tight_layout=True, figsize=(num_layers*4,13) )
    
    # print("Weights:")
    for k, weight in weights.items():
        _ = axs[0,k].hist(weight.ravel(), bins=n_bins)
        _ = axs[0,k].set_title(f" layer{layer_id[k]} {weight.shape} weights - ep:{epochs}", fontsize=9);
        _ = axs[0,k].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    
    # print("Biases:")
    for k, bias in biases.items():
        _ = axs[1,k].hist(bias.ravel(), bins=n_bins)
        _ = axs[1,k].set_title(f" layer{layer_id[k]} {bias.shape} biases - ep:{epochs}", fontsize= 9);
        _ = axs[1,k].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    
    # print("Gradients:")
    for k, grad in grads.items():
        _ = axs[2,k].hist(grad.ravel(), bins=n_bins)
        _ = axs[2,k].set_title(f" layer{layer_id[k]} {grad.shape} gradients - ep:{epochs}", fontsize= 9);
        _ = axs[2,k].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
        _ = axs[2,k].tick_params(axis='both', which='minor', labelsize=4)
    plt.show()


def plot_TSNE(prj, lbl, cmp, key = None, end = None, epoch = 0):
    if end is None:
        end = len(lbl)
    fig, axs = plt.subplots(1, 5, sharey=False, tight_layout=True, figsize=(20,4) )
    for layer in [0,1,2,3,4]:
        df = pd.DataFrame(dict(
            x=prj[layer][:end,0],
            y=prj[layer][:end,1],
            tpsa=lbl[0:end],
            compound = cmp[0:end]
        ))
        # print(key, np.bincount(lbl), df[key].unique() , palette_count)
        palette_count = len(df[key].unique())
        legend = True if layer in [0,4] else False
        lp=sb.scatterplot( data=df, x ="x", y = "y", hue=key, palette=sb.color_palette(n_colors=palette_count), ax=axs[layer], legend = legend) #, size=6)
        _=lp.set_title(f'Epoch: {epoch} layer {layer}', fontsize = 10)
    
    plt.show()


#-------------------------------------------------------------------------------------------------------------------
#  Display routines
#-------------------------------------------------------------------------------------------------------------------     

def display_model_summary(model, dataset = 'cellpainting', batch_size = 300 ):
    col_names = [ "input_size", "output_size", "num_params", "params_percent", "mult_adds", "trainable"]  # "kernel_size"
    if dataset =="cellpainting":
        summary_input_size = (batch_size, 1471)
    else:
        summary_input_size = (batch_size, 28, 28)
    print(summary(model, input_size=summary_input_size, col_names = col_names))
    
    # print()
    # for index, layer in enumerate(model.layers,1): 
    #     if isinstance(layer, torch.nn.Linear):
    #         if index == 1:
    #             print(f"a- layer {index} activations[{index}] :   Input: {layer.in_features}   out: {layer.out_features}")   
    #         elif index == len(model.layers):
    #             print(f"c- layer {index} activations[{index}] :   Input: {layer.in_features}   out: {layer.out_features}")   
    #         else:
    #             print(f"b- layer {index} activations[{index}] :   Input: {layer.in_features}   out: {layer.out_features}")   
    
    # print()
    # for param in model.parameters():
    #     print(type(param), param.size())
    
    print()
    for index, layer in enumerate(model.layers,1):
        print(f" {index:3d}:  {str(type(layer)):55s} -- {layer}", end ='')
        if isinstance(layer, torch.nn.Linear):
            print(f"\t\t { layer.weight.shape}")
        else:
            print()
    # print(f"\nModel Optimizer :\n{model.optimizer}") 



def display_cellpainting_batch(batch_id, batch):
    data, labels, plates, compounds, cmphash,  = batch
    print("-"*80)
    print(f" Batch Id: {batch_id}   {type(batch)}  Rows returned {len(batch[0])} features: {len(data[0])}  ")
    print("-"*80)
    print(f" idx |  batch[0] |              |                    | batch[3] |[4]|     batch[5]")
    print(f"     |           |              |                    |          |   |             ")
    print(f"{len(labels)}")
    for i in range(len(labels)):
        print(f" {i:3d} | {plates[i,0]:9s} | {compounds[i]:12s} | {cmphash[i,0]} | {cmphash[i,1]:2d} | {labels[i]:1d} | {data[i,:3]}")


def display_model_history(model, key, epoch, epochs):
    ttl = 'train' if key =='train' else "valid"
    print(f"{datetime.now().strftime('%Y%m%d_%H%M%S')} epoch {epoch + 1:3d} of {epochs:4d} - {ttl} |"
          f" mean loss = {model.training_history[key]['loss'][-1]:.6f} | xent loss = {model.training_history[key]['xent_loss'][-1]:.6f} |"
          f" snn loss = {model.training_history[key]['snn_loss'][-1]:.6f} | temp = {model.training_history[key]['temp_hist'][-1]:.6f} |"
          f" temp grad = {model.training_history[key]['temp_grad_hist'][-1]:.6e} | mean acc = {model.training_history[key]['accuracy'][-1]:.6f} |"
          f" mean f1 = {model.training_history[key]['f1'][-1]:.6f} |")


def display_epoch_metrics(model, epoch = -1, epochs = None, header = False):
    epochs = len(model.training_history[key1]['loss']) if epochs is None else epochs
    epoch =  len(model.training_history[key1]['loss'])  - 1 if epoch == -1 else epoch
    idx = epoch
    key1, key2 = 'train', 'val'
    header = True if epoch == 0 else header

    if model.unsupervised:
        if header:
            print(f"                     |   Trn_loss    PrimLoss      SNNL   |    temp*        grad     |                          |   Vld_loss    PrimLoss      SNNL   |                          |")
            print(f"---------------------+------------------------------------+--------------------------+--------------------------+------------------------------------+--------------------------|")
                 # "00:45:46 ep   1 / 10 |   9.909963    4.904229    5.005733 |  14.996347   -2.6287e-10 |                          |   9.833426    4.827625    5.005800 |                          |"
        print(f"{model.training_history[key2]['time'][idx]} ep {epoch + 1:3d} /{epochs:3d} |"
              f"  {model.training_history[key1]['loss'][idx]:9.6f}   {model.training_history[key1]['xent_loss'][idx]:9.6f}   {model.training_history[key1]['snn_loss'][idx]:9.6f} |"
              f"  {model.training_history[key1]['temp_hist'][idx]:9.6f}   {model.training_history[key1]['temp_grad_hist'][idx]:11.4e} |"
              f"                          |"
              f"  {model.training_history[key2]['loss'][idx]:9.6f}   {model.training_history[key2]['xent_loss'][idx]:9.6f}   {model.training_history[key2]['snn_loss'][idx]:9.6f} |"
              f"                          |")
        
    else:
        if header:
            print(f"                     |  Trn_loss     CEntropy      SNNL   |    temp        grad     |   ACC       F1     ROCAuc |   Vld_loss    CEntropy      SNNL   |    ACC      F1     ROCAuc |")
            print(f"---------------------+------------------------------------+-------------------------+---------------------------+------------------------------------+---------------------------|")
                 # "                     |  Trn_loss     CEntropy      SNNL   |    temp        grad     |   ACC       F1     ROCAuc |   Vld_loss    CEntropy      SNNL   |    ACC      F1     ROCAuc |"
                 # "---------------------+------------------------------------+-------------------------+---------------------------+------------------------------------+---------------------------|"
                 # "00:44:43 ep   1 / 10 |  10.054366    3.660260    6.394106 |  14.999862   1.5653e-04 |  0.7885   0.0796   0.5129 |   8.464406    2.070062    6.394344 |  0.8754   0.0223   0.5203 |"
        print(f"{model.training_history[key2]['time'][idx]} ep {epoch + 1:3d} /{epochs:3d} |"
              f"  {model.training_history[key1]['loss'][idx]:9.6f}   {model.training_history[key1]['xent_loss'][idx]:9.6f}   {model.training_history[key1]['snn_loss'][idx]:9.6f} |"
              f"  {model.training_history[key1]['temp_hist'][idx]:9.6f}   {model.training_history[key1]['temp_grad_hist'][idx]:11.4e} |"
              f"  {model.training_history[key1]['accuracy'][idx]:.4f}   {model.training_history[key1]['f1'][idx]:.4f}   {model.training_history[key1]['roc_auc'][idx]:.4f} |"
              f"  {model.training_history[key2]['loss'][idx]:9.6f}   {model.training_history[key2]['xent_loss'][idx]:9.6f}   {model.training_history[key2]['snn_loss'][idx]:9.6f} |"
              f"  {model.training_history[key2]['accuracy'][idx]:.4f}   {model.training_history[key2]['f1'][idx]:.4f}   {model.training_history[key2]['roc_auc'][idx]:.4f} |")



#-------------------------------------------------------------------------------------------------------------------
#  Metric routines
#-------------------------------------------------------------------------------------------------------------------     

def accuracy(y_true, y_pred) -> float:
    """
    Returns the classification accuracy of the model.

    Parameters
    ----------
    y_true: torch.Tensor
        The target labels.
    y_pred: torch.Tensor
        The predicted labels.

    Returns
    -------
    accuracy: float
        The classification accuracy of the model.
    """
    correct = (y_pred == y_true).sum().item()
    accuracy = correct / len(y_true)
    accuracy *= 100.0
    return accuracy

def binary_accuracy(y_true, y_prob):
    assert (y_true.ndim == 1 and y_true.size == y_prob.size) , f"binary accuracy:  y_true: {y_true.ndim}   {y_true.shape}  y_prob: {y_prob.ndim}   {y_prob.shape}   "
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum() / y_true.size

def binary_f1_score(y_true, y_prob):
    assert (y_true.ndim == 1 and y_true.size == y_prob.size) , f"binary f1 score:  y_true: {y_true.ndim}   {y_true.shape}  y_prob: {y_prob.ndim}   {y_prob.shape}   "
    y_prob = y_prob > 0.5
    return f1_score(y_true, y_prob)
