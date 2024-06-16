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
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
import logging
import random
import sys
from typing import List, Tuple
from types import SimpleNamespace
import yaml


from torchinfo import summary
import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt
import scipy.stats as sps 
import sklearn.metrics as skm 
from scipy.spatial.distance import pdist, squareform, euclidean
from snnl.models  import Autoencoder
logger = logging.getLogger(__name__) 


def get_device(verbose = False):
    gb = 2**30
    devices = torch.cuda.device_count()
    for i in range(devices):
        free, total = torch.cuda.mem_get_info(i)
        if verbose:
            print(f" device: {i}   {torch.cuda.get_device_name(i):30s} :  free: {free:,d} B   ({free/gb:,.2f} GB)    total: {total:,d} B   ({total/gb:,.2f} GB)")
    # device = 
    torch.cuda.empty_cache()
    del free, total
    device = f"{'cuda' if torch.cuda.is_available() else 'cpu'}:{torch.cuda.current_device()}"
    logger.info(f" Current CUDA Device is:  {device} - {torch.cuda.get_device_name()}" )
    return device

def set_device(device_id):
    # print(" Running on:",  torch.cuda.get_device_name(), torch.cuda.current_device())
    devices = torch.cuda.device_count()
    assert device_id < devices, f"Invalid device id, must be less than {devices}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = f"{device}:{device_id}"
    # print(f" Switch to {device} ")
    torch.cuda.set_device(device_id)
    logger.info(f" Switched to: {torch.cuda.get_device_name()} - {torch.cuda.current_device()}")
    return device
    
def parse_args(input = None):
    parser = argparse.ArgumentParser(description="DNN classifier with SNNL")
    group = parser.add_argument_group("Parameters")
    group.add_argument(
        "-s",
        "--seed",
        dest='random_seed',
        required=False,
        default=1234,
        type=int,
        help="the random seed value to use, default: [1234]",
    )
    group.add_argument(
        "-m",
        "--runmode",
        required=False,
        default="baseline",
        type=str,
        help="the model running mode - options: [baseline (default) | snnl]",
    )
    group.add_argument(
        "-c",
        "--configuration",
        required=True,
        # default="examples/hyperparameters/dnn.json",
        type=str,
        help="the path to the JSON file containing the hyperparameters to use",
    )
    group.add_argument('--wandb' , default=False, action=argparse.BooleanOptionalAction)
    group.add_argument('--run_id' , type=str, dest='exp_id', required=False, default=None, help="WandB run id (for run continuations)")
    group.add_argument('--ckpt'   , type=str, required=False, default=None, help="Checkpoint fle to resume training from")
    group.add_argument('--exp_title'  , type=str, required=False, default=None, help="Exp Title (overwrites yaml file value)")
    group.add_argument('--epochs', type=int, required=True, default=0, help="epochs to run")
    group.add_argument('--gpu_id', type=int, required=False, default=0, help="Cuda device id to use" )    
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

def load_configuration(input_params):

    with open(input_params.configuration) as f:
        _args = yaml.safe_load(f)
    
    input_params = vars(input_params)
    for k,v in input_params.items():
        logger.info(f" command line param [{k}] passed, value: [{v}]")
        if v is not None:
            _args[k] = v
    _args['ckpt'] = input_params['ckpt']
    _args['batch_size'] = _args['cellpainting_args']['batch_size']
    
    # args = types.SimpleNamespace(**args, **(vars(input_params)))
    return SimpleNamespace(**_args)    

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
#  Define Model
#-------------------------------------------------------------------------------------------------------------------     
def define_autoencoder_model(args, embedding_layer = 0, use_scheduler = True, use_temp_scheduler = False, device = None):
    assert  embedding_layer != 0, "embedding_layer cannot be zero"
    if device is None: 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    if args.runmode.lower() == "baseline":
        print(f"Defining model in baseline mode")
        model = Autoencoder(
            mode = "autoencoding",
            units=args.units,
 
            embedding_layer = embedding_layer,
            code_units  = args.code_units, 
            input_shape = args.input_shape, 
            sample_size = args.cellpainting_args['sample_size'],
            criterion   = torch.nn.MSELoss(reduction='mean'),
            loss_factor = args.loss_factor,
            learning_rate=args.learning_rate,
            adam_weight_decay = args.adam_weight_decay,
            use_scheduler = use_scheduler,
            
            use_snnl = False,
            snnl_factor= 0.0,
            use_temp_scheduler = use_temp_scheduler,
            device = device
            )
    elif args.runmode.lower() == "snnl":
        print(f"Defining model in SNNL mode ")
        model = Autoencoder(
            mode="latent_code",
            units=args.units,
 
            embedding_layer = embedding_layer,
            code_units    = args.code_units,
            input_shape   = args.input_shape,
            sample_size   = args.cellpainting_args['sample_size'],
            criterion     = torch.nn.MSELoss(reduction='mean'),
            loss_factor   = args.loss_factor,
            learning_rate = args.learning_rate,
            adam_weight_decay = args.adam_weight_decay,
            use_scheduler = use_scheduler,
            
            use_snnl=True,
            snnl_factor = args.snnl_factor,
            temperature = args.temperature,
            temperatureLR = args.temperatureLR,
            use_annealing = False,        
            use_sum = False,
            SGD_weight_decay = args.SGD_weight_decay,
            use_temp_scheduler = use_temp_scheduler,
            device = device
            )
    else:
        raise ValueError("Choose runmode between [baseline] and [snnl] only.")
        
    return model
    
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
    output = defaultdict(dict)
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    filename = os.path.join(results_dir, f"{filename}.json")
   
    model_attributes = model.__dict__
    for key, value in model_attributes.items():
        # print(f"{key:40s}, {type(value)}")
        if key == 'training_history':
            output[key] = value
        elif key[0] == "_"  or key == "layer_activations":
            continue
        elif type(value) in [torch.device, torch.optim.Adam , torch.optim.SGD, torch.optim.lr_scheduler.ReduceLROnPlateau]:
            continue
        else:
            output['params'][key] = value
    with open(filename, "w") as file:
        json.dump(output, file)
    logger.info(f" Model Results exported to {filename}.")


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
 
    results_dir = "results"
    filename = os.path.join(results_dir, f"{filename}.json")
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
    logger.info(f" Model exported to {path}.")


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
    logger.info(f" Model imported from {path}.")
    return torch.load(path)


def save_checkpoint(epoch, model, filename, update_latest=False, update_best=False):
    model_checkpoints_folder = os.path.join("ckpts")
    if not os.path.exists(model_checkpoints_folder):
        print(f"path {model_checkpoints_folder} doesn't exist")
    checkpoint = {'epoch': epoch,
                  'use_snnl': model.use_snnl,
                  'state_dict': model.state_dict,
                  'optimizer_state_dict': model.optimizer.state_dict(),
                  'temp_optimizer_state_dict': model.temp_optimizer.state_dict(),
                 }
    
    # checkpoint['scheduler'] =  model.scheduler.state_dict() if model.use_scheduler else None
    # checkpoint['temp_scheduler'] =  model.temp_scheduler.state_dict() if model.use_temp_scheduler else None 
    
        
    if update_latest:
        filename = os.path.join(model_checkpoints_folder, f"{filename}_model_latest.pt")
    elif update_best:
        filename = os.path.join(model_checkpoints_folder, f"{filename}_model_best.pt")
    else:
        filename = os.path.join(model_checkpoints_folder, f"{filename}.pt")
    torch.save(checkpoint, filename) 
    logger.info(f" Model exported to {filename}.")


def save_checkpoint_v2(epoch, model, filename, update_latest=False, update_best=False, verbose = False):
    from types import NoneType
    model_checkpoints_folder = os.path.join("ckpts")
    if not os.path.exists(model_checkpoints_folder):
        print(f"path {model_checkpoints_folder} doesn't exist")
        
    checkpoint = {'epoch'                     : epoch,
                  'state_dict'                : model.state_dict(),
                  'optimizer'                 : model.optimizer,
                  'temp_optimizer'            : model.temp_optimizer,
                  'optimizer_state_dict'      : model.optimizer.state_dict() if model.optimizer is not None else None,
                  'temp_optimizer_state_dict' : model.temp_optimizer.state_dict() if model.temp_optimizer is not None else None,
                  'scheduler'                 : model.scheduler,
                  'temp_scheduler'            : model.temp_scheduler,
                  'scheduler_state_dict'      : model.scheduler.state_dict() if model.use_scheduler else None,
                  'temp_scheduler_state_dict' : model.temp_scheduler.state_dict() if model.use_temp_scheduler else None ,
                  'params': dict()
                 }
    
    
    model_attributes = model.__dict__
    for key, value in model_attributes.items():
        if key not in checkpoint:
            if key[0] == '_' :
                if verbose:
                    print(f"{key:40s}, {str(type(value)):60s} -- {key} in ignore_attributes - will not be added")
            else:
                if verbose:
                    print(f"{key:40s}, {str(type(value)):60s} -- add to checkpoint dict")
                checkpoint['params'][key] = value
        else:
            if verbose:
                print(f"{key:40s}, {str(type(value)):60s} -- {key} already in checkpoint dict")
    if verbose:
        print(checkpoint.keys())    
    if update_latest:
        s_filename = os.path.join(model_checkpoints_folder, f"{filename}_model_latest.pt")
    elif update_best:
        s_filename = os.path.join(model_checkpoints_folder, f"{filename}_model_best.pt")
    else:
        filename = os.path.join(model_checkpoints_folder, f"{filename}.pt")
 
    torch.save(checkpoint, filename) 
    logger.info(f" Model exported to {filename}.")


def load_checkpoint(model, filename, verbose = False ):
    epoch = 9999
    try:
        checkpoints_folder = os.path.join("ckpts")
        checkpoint = torch.load(os.path.join(checkpoints_folder, filename))
        if verbose:
            print(checkpoint.keys())
            print(" --> load model state_dict")
        model.load_state_dict(checkpoint['state_dict'])
        if verbose:
            print(" --> load optimizer state_dict")
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if "scheduler" in checkpoint and (hasattr(model, 'scheduler')):
            model.scheduler = checkpoint['scheduler']
        epoch = checkpoint.get('epoch',0)
        logger.info(f" ==> Loaded from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}\n")
         
    # except FileNotFoundError:
    #     Exception("Previous state checkpoint not found.")
    except :
        print(sys.exc_info())

    return model, epoch


def load_checkpoint_v2(model, filename, dryrun = False):
    epoch = -1
    if filename[-3:] != '.pt':
        filename+='.pt'
    logging.info(f" Load model checkpoint from  {filename}")    
    ckpt_file = os.path.join("ckpts", filename)
    
    try:
        checkpoint = torch.load(ckpt_file)
    except FileNotFoundError:
        Exception("Previous state checkpoint not found.")
        print("FileNotFound Exception")
    except :
        print("Other Exception")
        print(sys.exc_info())

    for key, value in checkpoint.items():
        logging.debug(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
    print()
    
    
    if not dryrun:
        model.load_state_dict(checkpoint['state_dict'])

        for key, value in checkpoint['params'].items():
            logging.debug(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
            model.__dict__[key] = value
            
        if model.optimizer is not None:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if model.temp_optimizer is not None:
            model.temp_optimizer.load_state_dict(checkpoint['temp_optimizer_state_dict'])
        
        if model.scheduler is not None:
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if model.temp_scheduler is not None:
            model.temp_scheduler.load_state_dict(checkpoint['temp_scheduler_state_dict'])
    else:
        for key, value in checkpoint['params'].items():
            logging.debug(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
        
    epoch = checkpoint.get('epoch',-1)
    logger.info(f" ==> Loaded from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")

    return model, epoch


def load_model_from_ckpt(model, runmode = None, date = None, title = None, epochs = None, 
                         filename =None, cpb = None, factor = None , dryrun = False, v1 = True, verbose = False):
    # filename = f"AE_{args.model.lower()}_{date}_{title}_{epochs:03d}_cpb_{args.compounds_per_batch}_factor_{factor}.pt"    
    if filename is None:
        if factor is None:
            filename = f"{model.name}_{runmode}_{date}_{title}_ep_{epochs:03d}.pt"
        else:
            filename = f"{model.name}_{runmode}_{date}_{title}_{epochs:03d}_cpb_{cpb}_factor_{factor:d}.pt"
        print(filename)
        
    if os.path.exists(os.path.join('ckpts', filename)):
        # print(f"\n {filename}   *** Checkpoint EXISTS *** \n")
        if v1:
            model, last_epoch = load_checkpoint(model, filename)
        else:
            model, last_epoch = load_checkpoint_v2(model, filename, dryrun)
        _ = model.eval()
        # model = model.to(current_device)
        # model.to('cpu')
        if verbose:
            print(f" model device: {model.device}")
            print(f" model temperature: {model.temperature}")
    else:
        logger.error(f" {filename} *** Checkpoint DOES NOT EXIST *** \n")
        raise ValueError(f"\n {filename} *** Checkpoint DOES NOT EXIST *** \n")
        
    return model
    

def fix_checkpoint_v2 (filename):

    checkpoints_folder = os.path.join("ckpts")
    orig_ckpt_file = os.path.join(checkpoints_folder, filename+'.pt')
    fixed_ckpt_file = os.path.join(checkpoints_folder, filename+'_fixed.pt')
    
    print(f" ==> Original checkpoint: {orig_ckpt_file}")
    print(f" ==> Fixed    checkpoint: {fixed_ckpt_file}")
    
    try:
        checkpoint = torch.load(orig_ckpt_file)
        print(f"\n ==> Loaded from checkpoint {filename} successfully. last epoch on checkpoint: {checkpoint['epoch']}\n")
    except FileNotFoundError:
        Exception("Original checkpoint not found.")
    except :
        print("Other Exception")
        print(sys.exc_info())

    fixed_checkpoint = dict()
    fixed_checkpoint['params'] = dict()
    for key, value in checkpoint.items():
        if key in ['epoch', 'state_dict', 'optimizer_state_dict', 'temp_optimizer_state_dict', 'scheduler_state_dict', 'temp_scheduler_state_dict' ]:
            fixed_checkpoint[key] = value
            print(f"{key:40s}, {str(type(value)):60s}  -- major key set  ")
        else:
            fixed_checkpoint['params'][key] = value
            print(f"{key:40s}, {str(type(value)):60s}  -- params key set  ")
            
    try:
        torch.save(fixed_checkpoint, fixed_ckpt_file) 
        print(f"[INFO] Model exported to { fixed_ckpt_file}.")
    except :
        print("Other Exception 2")
        print(sys.exc_info())

    return  fixed_checkpoint
#-------------------------------------------------------------------------------------------------------------------
#  Plotting routines
#-------------------------------------------------------------------------------------------------------------------     

def plot_train_history(model, epochs= None, start= 0, n_bins = 25):
    key1, key2 = model.training_history.keys()
    key1 = 'trn' if key1 == 'train' else key1

    if epochs is None:
        epochs = len(model.training_history[key1]['trn_ttl_loss'])
     
    fig, axs = plt.subplots(1, 5, sharey=False, tight_layout=True, figsize=(5*4,4) )
    x_data = np.arange(start,epochs)
    labelsize = 6
    # We can set the number of bins with the *bins* keyword argument.
    i = 0
    _ = axs[i].plot(x_data, model.training_history[key1]['trn_ttl_loss'][start:epochs],label='Training');
    _ = axs[i].plot(x_data, model.training_history['val']['val_ttl_loss'][start:epochs],label='Validation');
    _ = axs[i].set_title(f'Total loss - epochs {start}-{epochs}', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    axs[i].legend()
    i +=1    
    _ = axs[i].plot(x_data, model.training_history[key1]['trn_prim_loss'][start:epochs],label='Training');
    _ = axs[i].plot(x_data, model.training_history['val']['val_prim_loss'][start:epochs],label='Validation');
    _ = axs[i].set_title(f'Primary loss - epochs {start}-{epochs}', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    i +=1
    _ = axs[i].plot(x_data, model.training_history[key1]['trn_snn_loss'][start:epochs]);
    _ = axs[i].plot(x_data, model.training_history['val']['val_snn_loss'][start:epochs]);
    _ = axs[i].set_title(f'Soft Nearest Neighbor Loss - epochs {start}-{epochs}', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    i +=1
    _ = axs[i].plot(x_data, model.training_history[key1]['trn_lr'][start:epochs]);
    # if model.use_snnl:
    #     _ = axs[i].plot(x_data, model.training_history[key1]['temp_lr'][start:epochs]);
    _ = axs[i].set_title(f'Learning Rate - epochs {start}-{epochs}', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    _ = axs[i].set_title(f'train_temp_hist - epochs {start}-{epochs}', fontsize= 10);
    if model.use_snnl:
        i +=1
        _ = axs[i].plot(x_data, model.training_history[key1]['temp_hist'][start:epochs]);
        _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)    # i +=1
        
    # batches = (len(model.training_history[key1]['temp_grads']) // len(model.training_history[key1]['trn_ttl_loss'])) *epochs
    # _ = axs[i].plot(model.training_history[key1]['temp_grads'][:batches])
    # _ = axs[i].set_title(f'Temperature Gradients - {epochs} epochs', fontsize= 10)
    # _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    # i +=1
    # _ = axs[i].plot(model.training_history[key1]['temp_grad_hist'][:epochs]);
    # _ = axs[i].set_title(f"Temperature Grad at end of epochs - {epochs} epochs", fontsize= 10);
    # _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    # i +=1
    # _ = axs[i].plot(model.training_history[key1]['layer_grads'][:epochs]);
    # _ = axs[i].set_title(f"Monitored layer gradient - {epochs} epochs", fontsize= 10);
    # _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)    
    # i +=1
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
 
    
    print(f" +------+-------------------------------------------------------+----------------------------------------------+---------------------------------------+")
    print(f" |      | Weights:                                              |  Biases:                                     |   Gradients:                          |")
    print(f" | layr |                      min           max         stdev  |             min          max          stdev  |      min          max          stdev  |")
    print(f" +------+-------------------------------------------------------+----------------------------------------------+---------------------------------------+")
          # f" |    0 | (1024, 1471)     -11.536192     5.169790     0.151953 |   1024   -8.655299     2.601529     2.123748 |   -0.010149     0.010479     0.000481 |""
    for k in layer_id.keys():
        print(f" | {k:4d} | {str(weights[k].shape):15s}  {weights[k].min():-10.6f}   {weights[k].max():-10.6f}   {weights[k].std():-10.6f}"
              f" |  {biases[k].shape[0]:5d}  {biases[k].min():-10.6f}   {biases[k].max():-10.6f}   {biases[k].std():-10.6f}"
              f" |  {grads[k].min():-10.6f}   {grads[k].max():-10.6f}   {grads[k].std():-10.6f} |")
    print(f" +------+-------------------------------------------------------+----------------------------------------------+---------------------------------------+")
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

def plot_TSNE_2(prj, lbl, cmp, key = None, layers = None, items = None, epoch = 0, limits = (None,None)):
    if layers is None:
        layers = range(len(prj))
        
    if items is None:
        lbl_len = len(lbl)
    elif not isinstance(items, list):
        items = list(items)
    fig, axs = plt.subplots(1, len(layers), sharey=False, tight_layout=True, figsize=(len(layers)*4,4) )

    for idx, layer in enumerate(layers):
        if items is None: 
            df = pd.DataFrame(dict(
                    x=prj[layer][:,0],
                    y=prj[layer][:,1],
                    tpsa=lbl[:],
                    compound = cmp[:]
                ))
        else:
            df = pd.DataFrame(dict(
                    x=prj[layer][items,0],
                    y=prj[layer][items,1],
                    tpsa=lbl[items],
                    compound = cmp[items]
                ))
        # print(key, np.bincount(lbl), df[key].unique() , palette_count)
        palette_count = len(df[key].unique())
        legend = True if layer in [0,4] else False
        lp=sb.scatterplot( data=df, x ="x", y = "y", hue=key, palette=sb.color_palette(n_colors=palette_count), ax=axs[idx]) #, size=6)
        _=lp.set_title(f'Epoch: {epoch} layer {layer}', fontsize = 10)
        lp.legend(loc = 'best', fontsize = 8)
        lp.set_xlim([limits[0], limits[1]])
        lp.set_ylim([limits[0], limits[1]])
    
    plt.show()
    return fig


def plot_classification_metrics(model, epochs= None, n_bins = 25):
    key1, key2 = model.training_history.keys()
    key1 = 'trn' if key1 == 'train' else key1
 
    if epochs is None:
        epochs = len(model.training_history[key1]['trn_ttl_loss'])    
    fig, axs = plt.subplots(1, 5, sharey=False, tight_layout=True, figsize=(5*4,4) )
    i = 0
    _ = axs[i].plot(model.training_history[key1]['trn_accuracy'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['val_accuracy'][:epochs]);
    _ = axs[i].set_title(f'Accuracy - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    i +=1
    _ = axs[i].plot(model.training_history[key1]['trn_f1'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['val_f1'][:epochs]);
    _ = axs[i].set_title(f'F1 Score - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    i += 1
    _ = axs[i].plot(model.training_history[key1]['trn_precision'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['val_precision'][:epochs]);
    _ = axs[i].set_title(f' Precision - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    i +=1
    _ = axs[i].plot(model.training_history[key1]['trn_roc_auc'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['val_roc_auc'][:epochs]);
    _ = axs[i].set_title(f' ROC AUC Score - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    i +=1
    _ = axs[i].plot(model.training_history[key1]['trn_recall'][:epochs]);
    _ = axs[i].plot(model.training_history['val']['val_recall'][:epochs]);
    _ = axs[i].set_title(f'Recall - {epochs} epochs', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    i +=1
    plt.show()        


def plot_classification_metrics_2(cm):
    fig, ax = plt.subplots(1, 3, figsize=(14, 8))
    
    pr_display = skm.PrecisionRecallDisplay.from_predictions(cm.labels, cm.logits, name="LinearSVC", plot_chance_level=True, ax=ax[0]);
    _ = pr_display.ax_.set_title(f" 2-class PR curve - epoch:{cm.epochs} ");
    
    roc_display = RocCurveDisplay.from_predictions(cm.labels, cm.logits, pos_label= 1, ax = ax[1])
    _ = roc_display.ax_.set_title(f" ROC Curve - epoch:{cm.epochs} ")
    
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true = cm.labels, y_pred =cm.y_pred, ax = ax[2], colorbar = False)
    _ = cm_display.ax_.set_title(f" Confusion matrix - epoch:{cm.epochs} ");
    
    plt.show()


def plot_regression_metrics(model, epochs= None, n_bins = 25):
    key1, key2 = model.training_history.keys()
    key1 = 'trn' if key1 == 'train' else key1
 
    if epochs is None:
        epochs = len(model.training_history[key1]['trn_ttl_loss'])    
    fig, axs = plt.subplots(1, 1, sharey=False, tight_layout=True, figsize=(1*4,4) )
    i = 0
    _ = axs.plot(model.training_history[key1]['trn_R2_score'][:epochs]);
    _ = axs.plot(model.training_history['val']['val_R2_score'][:epochs]);
    _ = axs.set_title(f'R2 Score - {epochs} epochs', fontsize= 10);
    _ = axs.tick_params(axis='both', which='major', labelsize=7, labelrotation=45)


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


def display_cellpainting_batch(batch_id, batch):
    data, labels, plates, compounds, cmphash,  = batch
    print("-"*135)
    print(f"  Batch Id: {batch_id}   {type(batch)}  Rows returned {len(batch[0])} features: {len(data[0])}  ")
    print(f"+-----+--------------+----------------+----------------------+--------+-------+--------------------------------------------------------+")
    print(f"| idx |   batch[0]   |    batch[1]    |      batch[2]        |batch[3]| [4]   |     batch[5]                                           | ") 
    print(f"|     |    SOURCE    |    COMPOUND    |       HASH           | BIN    | LABEL |     FEATURES                                           | ")
    print(f"+-----+--------------+----------------+----------------------+--------+-------+--------------------------------------------------------+")
         # "  0 | source_1     | JCP2022_006020 | -9223347314827979542 |   10 |  0 | tensor([-0.4223, -0.4150,  0.2968])"
         # "  1 | source_10    | JCP2022_006020 | -9223347314827979542 |   10 |  0 | tensor([-0.6346, -0.6232, -1.6046])"
    
    for i in range(len(labels)):
        print(f"  {i:3d} | {plates[i,0]:12s} | {compounds[i]:12s} | {cmphash[i,0]:20d} | {cmphash[i,1]:4d}   |  {int(labels[i]):2d}   | {data[i,:3]}")


def display_epoch_metrics(model, epoch = None, epochs = None, header = False):
    # key1, key2 = model.training_history.keys()
    key1 = 'trn' ##if key1 == 'trn' else ''
    key2 = 'val' ##if key1_p == 'trn' else ''
    
    history_len = len(model.training_history[key1][f'{key1}_ttl_loss'])
    epochs = history_len if epochs is None else epochs
    epoch  = 0 if epoch is None else epoch
    header = True if epoch == 0 else header
    
    idx = epoch
    if idx>=epochs:
        return
    if model.use_snnl:
        temp_hist = model.training_history[key1]['temp_hist'][idx]
        temp_grad_hist = model.training_history[key1]['temp_grad_hist'][idx]
        temp_LR = model.training_history[key1]["temp_lr"][idx] 
    else:
        temp_hist = 0
        temp_grad_hist = 0
        temp_LR = 0
        
    trn_LR = model.training_history[key1]["trn_lr"][idx] if model.use_scheduler else 0.0
 
    if model.unsupervised:
        if header:
            print(f"                     |   Trn_loss    PrimLoss      SNNL   |    temp*        grad     |   R2                     |   Vld_loss    PrimLoss      SNNL   |   R2                     |    LR       temp LR   |")
            print(f"---------------------+------------------------------------+--------------------------+--------------------------+------------------------------------+--------------------------|-----------------------|")
                 # "00:45:46 ep   1 / 10 |   9.909963    4.904229    5.005733 |  14.996347   -2.6287e-10 |                          |   9.833426    4.827625    5.005800 |                          |"
        print(f"{model.training_history[key2][f'{key2}_time'][idx]} ep {epoch + 1:3d} /{epochs:3d} |"
              f"  {model.training_history[key1][f'{key1}_ttl_loss'][idx]:9.6f}   {model.training_history[key1][f'{key1}_prim_loss'][idx]:9.6f}   {model.training_history[key1][f'{key1}_snn_loss'][idx]:9.6f} |"
              f"  {temp_hist:9.6f}   {temp_grad_hist:11.4e} |"
              f"  {model.training_history[key1][f'{key1}_R2_score'][idx]:7.4f}   {model.training_history['gen'].get('trn_best_metric_ep',0)+1:4d}          |"
              f"  {model.training_history[key2][f'{key2}_ttl_loss'][idx]:9.6f}   {model.training_history[key2][f'{key2}_prim_loss'][idx]:9.6f}   {model.training_history[key2][f'{key2}_snn_loss'][idx]:9.6f} |"
              f"  {model.training_history[key2][f'{key2}_R2_score'][idx]:7.4f}   {model.training_history['gen'].get('val_best_metric_ep',0)+1:4d}          |"
              f"  {trn_LR :9f}  {temp_LR :9f} |")
        
    else:
        if header:
            print(f"                     |  Trn_loss     PrimLoss      SNNL   |    temp         grad     |   ACC       F1     ROCAuc |   Vld_loss    PrimLoss      SNNL   |    ACC      F1     ROCAuc |")
            print(f"---------------------+------------------------------------+--------------+-----------+---------------------------+------------------------------------+---------------------------|")
                 # "                     |  Trn_loss     CEntropy      SNNL   |    temp        grad     |   ACC       F1     ROCAuc |   Vld_loss    CEntropy      SNNL   |    ACC      F1     ROCAuc |"
                 # "---------------------+------------------------------------+-------------------------+---------------------------+------------------------------------+---------------------------|"
                 # "00:44:43 ep   1 / 10 |  10.054366    3.660260    6.394106 |  14.999862   1.5653e-04 |  0.7885   0.0796   0.5129 |   8.464406    2.070062    6.394344 |  0.8754   0.0223   0.5203 |"
        print(f"{model.training_history[key2][f'{key2}_time'][idx]} ep {epoch + 1:3d} /{epochs:3d} |"
              f"  {model.training_history[key1][f'{key1}_ttl_loss'][idx]:9.6f}   {model.training_history[key1][f'{key1}_prim_loss'][idx]:9.6f}   {model.training_history[key1][f'{key1}_snn_loss'][idx]:9.6f} |"
              f"  {temp_hist:9.6f}   {temp_grad_hist:11.4e} |"
              f"  {model.training_history[key1][f'{key1}_accuracy'][idx]:.4f}   {model.training_history[key1][f'{key1}_f1'][idx]:.4f}   {model.training_history[key1][f'{key1}_roc_auc'][idx]:.4f} |"
              f"  {model.training_history[key2][f'{key2}_ttl_loss'][idx]:9.6f}   {model.training_history[key2][f'{key2}_prim_loss'][idx]:9.6f}   {model.training_history[key2][f'{key2}_snn_loss'][idx]:9.6f} |"
              f"  {model.training_history[key2][f'{key2}_accuracy'][idx]:.4f}   {model.training_history[key2][f'{key2}_f1'][idx]:.4f}   {model.training_history[key2][f'{key2}_roc_auc'][idx]:.4f} |" 
              f"  {trn_LR :9f}    {temp_LR :9f}")


def display_dist_metrics(dist_metrics, epochs, metric = 'euclidian'):
    titles = [f'INPUT features - {metric} distances - {epochs} epochs',
              f'EMBEDDED features - {metric} distances - {epochs} epochs',
              f'RECONSTURCTED features -  {metric} distances - {epochs} epochs']
    sub_titles = ["ALL group distances", "INTRA-group (same compound) distances", "INTER-group (diff compound) distances"]
    k0_list = ['inp', 'emb', 'out']
    k1_list = ['all', 'same', 'diff']
    k2_list = ['_min', '_max', '_avg', '_std', ]
    
    for idx0,(title, k0) in enumerate(zip(titles, k0_list)):
        print()        
        print(title)
        print('-'*len(title))    
        
        for idx1,(subtitle, k1) in enumerate(zip(sub_titles, k1_list)):
            key = k0+'_'+k1
            row = idx1   ## 0 (all groups) , 1: same grou, 2: diff_groups
            print(f"  {subtitle:37s} :    min,max:  [{dist_metrics[key+'_min']:8.4f}, {dist_metrics[key+'_max']:8.4f}]"
                  f"  | Individual point distances   mean: {dist_metrics['CL_'+key+ '_avg']:8.4f}   std: {dist_metrics['CL_'+key+ '_std']:8.4f}  |"
                  f"  Group Level Avg. Distances - mean: {dist_metrics[key+'_avg']:8.4f}   std: {dist_metrics[key+'_std']:8.4f} ")
    print()

    
def display_classification_metrics(cm):
    print(f" metrics at epoch {cm.epochs:^4d}")
    print('-'*22)
    print(f" F1 Score:  {cm.f1:.7f}")
    print(f" Accuracy:  {cm.accuracy*100:.2f}%")
    print(f" Precision: {cm.precision*100:.2f}%")
    print(f" Recall:    {cm.recall:.7f}")
    print(f" ROC_AUC:   {cm.roc_auc:.7f}")
    print()
    print(cm.cls_report)


def display_regr_metrics(rm):
    print(f" metrics at epoch {rm.epochs:^4d}")
    print('-'*22)
    print(f"RMSE Score : {rm.rmse_score:9.6f}")
    print(f" MSE Score : {rm.mse_score:9.6f}")
    print(f" MAE Score : {rm.mae_score:9.6f}")
    print(f"  R2 Score : {rm.R2_score:9.6f} ")    
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
    return skm.f1_score(y_true, y_prob)

#-------------------------------------------------------------------------------------------------------------------
#  Run Model on Test Dataloader
#-------------------------------------------------------------------------------------------------------------------     
def run_model_on_test_data(model, data_loader, embedding_layer, verbose = False):
    """
    embedding layer: layer that contains embedding (for encoding models)
    """
    out = SimpleNamespace()
    out.labels = np.empty((0))
    out.logits = np.empty((0))
    out.compounds = np.empty((0))
    out.embeddings = {} 
    
    for k, layer in enumerate(model.layers):
        if hasattr(layer,"out_features"):
            out.embeddings[k] = np.empty((0,layer.out_features))
        else:
            out.embeddings[k] = np.empty((0,model.layers[k-1].out_features))
        if verbose:
            print(f" layer {k:2d}: - {layer} {out.embeddings[k].shape}")

    out.y_true = np.empty((0,model.layers[0].in_features))
    if hasattr(model.layers[-1],"out_features"):
        out.y_pred = np.empty((0,model.layers[-1].out_features))
        if verbose:
            print(f" y_pred picked up from layer {k} of {len(model.layers)} layers")
    elif hasattr(model.layers[-2],"out_features"):
        out.y_pred = np.empty((0,model.layers[-2].out_features))
        if verbose:
            print(f" y_pred picked up from layer {k-1} of {len(model.layers)} layers")
    else:
        raise ValueError("last two layers are not linear")
        
    ###- Pass Test Dataset through network
    for idx, (batch_features, batch_labels, batch_wellinfo , batch_compounds, batch_hashbin) in enumerate(data_loader):
        
        #    (batch_features, batch_labels, plate_well, compound, hash) 
        batch_features = batch_features.to(model.device)
        batch_labels = batch_labels.to(model.device)
        output_activations, reconstruction =  model.forward(batch_features)
                
        for k,v in output_activations.items():
            out.embeddings[k] = np.concatenate((out.embeddings[k], v.detach().cpu().numpy()))
        out.compounds = np.concatenate((out.compounds, batch_compounds))    
        out.y_true = np.concatenate((out.y_true, batch_features.detach().cpu().numpy()))
        out.labels = np.concatenate((out.labels, batch_labels.detach().cpu().numpy()))
        out.y_pred = np.concatenate((out.y_pred, reconstruction.detach().cpu().numpy()))
        if verbose:
            print(f"   output :  {idx:2d} - Labels:{out.labels.shape[0]:5d}   y_true:{out.y_true.shape}   y_pred:{out.y_pred.shape} ")
    ###- end
    
    out.labels = np.array(out.labels.astype(np.int32))
    out.comp_labels = np.arange(out.labels.shape[0])//3 
    out.latent_embedding = out.embeddings[embedding_layer]
    # logits = np.concatenate((logits, out_logits.detach().numpy()[:,0]))
    if verbose:
        print(f" out.latent_embedding shape: {out.latent_embedding.shape}")
        print(f" out.y_true    :      shape: {out.y_true.shape}")
        print(f" out.y_pred    :      shape: {out.y_pred.shape}")
        print(f" out.compounds :      shape: {out.compounds.shape}")
        print(f" out.labels    :      shape: {out.labels.shape} - Pos Labels {out.labels.sum()}")
        print(f" out.comp_labels:     shape: {out.comp_labels.shape} - {out.comp_labels[:25]}")
        for k,v in out.embeddings.items():
            print(f" out.embeddings[{k:2d}]  :   shape: {v.shape}  {' <---- embedding layer' if k == embedding_layer else ''}")
   
    return out


#-------------------------------------------------------------------------------------------------------------------
#  Run Model on Test Dataloader
#-------------------------------------------------------------------------------------------------------------------     
def get_latent_representation(model, data_loader, embedding_layer, verbose = False):
    """
    embedding layer: layer that contains embedding (for encoding models)
    """
    print(f" embedding layer: {embedding_layer} - model.layer: in: {model.layers[embedding_layer].in_features} out: {model.layers[embedding_layer].out_features}")
    
    ###- Pass Test Dataset through network
    for idx, (batch_features, batch_labels, batch_wellinfo , batch_compounds, batch_hashbin) in enumerate(data_loader):      
                        
        latent_code =  model.compute_latent_code(batch_features.to(model.device))
        output_batch = np.hstack((batch_wellinfo, np.expand_dims(batch_compounds,-1), batch_hashbin, np.expand_dims(batch_labels,-1), latent_code.detach().cpu().numpy()))
        if idx == 0:
            output = output_batch
        else :
            output = np.vstack((output,output_batch))
        print(output_batch.shape, output.shape)
    
    # if verbose:
    #     print(f" out.latent_embedding shape: {out.embeddings.shape}")
    #     print(f" out.labels    :      shape: {out.labels.shape} - Pos Labels {out.labels.sum()}")
    #     print(f" out.wellinfo  :      shape: {out.wellinfo.shape}")
    #     print(f" out.compounds :      shape: {out.compounds.shape}")
    #     print(f" out.hashbin   :      shape: {out.hashbin.shape}")
    return output

def pairwise_cosine_distance(features: torch.Tensor) -> torch.Tensor:
    """
    Returns the pairwise cosine distance between two copies
    of the features matrix.

    Parameter
    ---------
    features: torch.Tensor                The input features.

    Returns
    -------
    distance_matrix: torch.Tensor         The pairwise cosine distance matrix.

    Example
    -------
    >>> import torch
    >>> from snnl import SNNLoss
    >>> _ = torch.manual_seed(42)
    >>> a = torch.rand((4, 2))
    >>> snnl = SNNLoss(temperature=1.0)
    >>> snnl.pairwise_cosine_distance(a)
    tensor([[1.1921e-07, 7.4125e-02, 1.8179e-02, 1.0152e-01],
            [7.4125e-02, 1.1921e-07, 1.9241e-02, 2.2473e-03],
            [1.8179e-02, 1.9241e-02, 1.1921e-07, 3.4526e-02],
            [1.0152e-01, 2.2473e-03, 3.4526e-02, 0.0000e+00]])
    """
    normalized_a = torch.nn.functional.normalize(features, dim=1, p=2)
    return 1.0 - torch.matmul(normalized_a, normalized_a.T)
    
    # a, b = features.clone(), features.clone()
    # normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
    # normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
    # normalized_b = torch.conj(normalized_b).T
    # product = torch.matmul(normalized_a, normalized_b.T)
    # distance_matrix = torch.sub(torch.tensor(1.0), product)
    # return torch.sub(torch.tensor(1.0), product)
    # return distance_matrix    
    
def pairwise_euclidean_distance(features: torch.Tensor ) -> torch.Tensor:
    """
    Returns the pairwise Euclidean distance between two copies
    of the features matrix.

    Parameter
    ---------
    features: torch.Tensor               The input features.

    Returns
    -------
    distance_matrix: torch.Tensor        The pairwise Euclidean distance matrix.

    Example
    -------
    >>> import torch
    >>> from snnl import SNNLoss
    >>> _ = torch.manual_seed(42)
    >>> a = torch.rand((4, 2))
    >>> snnl = SNNLoss(temperature=1.0)
    >>> snnl.pairwise_euclidean_distance(a)
    tensor([[1.1921e-07, 7.4125e-02, 1.8179e-02, 1.0152e-01],
            [7.4125e-02, 1.1921e-07, 1.9241e-02, 2.2473e-03],
            [1.8179e-02, 1.9241e-02, 1.1921e-07, 3.4526e-02],
            [1.0152e-01, 2.2473e-03, 3.4526e-02, 0.0000e+00]])
    """
    
    # distance_matrix = torch.Tensor(squareform(pdist(features)))
    # a, b = features.clone(), features.clone()
    # normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
    # normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
    # # normalized_b = torch.conj(normalized_b).T
    # product = torch.matmul(normalized_a, normalized_b.T)
    # distance_matrix = torch.sub(torch.tensor(1.0), product)
    return torch.sqrt(((features[:, :, None] - features[:, :, None].T) ** 2).sum(1))

def distance_metrics_sample_set(model_data, num_samples = 10, cps = 3, seed = 1236, display = True):
    """
    selects a random set of samples that have been passed through the model
    prepares the necessary datasets for distance metric computation 

    cps: Compounds per Sample
    
    """
    CPS = cps
    num_inputs  = model_data.y_true.shape[0] // CPS
    np.random.seed(seed)
    
    sample = np.random.randint(0,num_inputs-1, num_samples)
    sample = np.array(sorted(sample))
    sample *= 3
    
    sorted_sample = np.concatenate((sample, sample+1, sample+2))
    sorted_sample.sort()
    samp_cmpnd  = model_data.compounds[sorted_sample]
    samp_input  = model_data.y_true[[sorted_sample],:].squeeze()
    samp_embed  = model_data.latent_embedding[[sorted_sample],:].squeeze()
    samp_output = model_data.y_pred[[sorted_sample],:].squeeze()
    if display:
        print(f" Number of unique compounds    :   {num_inputs}")
        print(f" Sampled compounds      - shape:   {sample.shape} - {sample}")
        print(f" Sampled compounds      - shape:   {sorted_sample.shape} - {sorted_sample}")
        print(f" Sample Input features  - shape:   {samp_input.shape}")
        print(f" Sample Latent features - shape:   {samp_embed.shape}")
        print(f" Sample Output features - shape:   {samp_output.shape}")
    
    # emb_cos_distance = squareform(pdist(samp_embed , "cosine"))
    # out_cos_distance = squareform(pdist(samp_output , "cosine"))
    # inp_euc_distance = squareform(pdist(samp_input))
    # emb_euc_distance = squareform(pdist(samp_embed))
    # out_euc_distance = squareform(pdist(samp_output))
    
    samp_act = [samp_input, samp_embed, samp_output]
    return samp_act

def distance_metric_1(measurement):
    titles = [f'INPUT features - Euclidean distances - {epochs} epochs',
              f'EMBEDDED features - Euclidean distances - {epochs} epochs',
              f'RECONSTURCTED features -  Euclidean distances - {epochs} epochs']
    sub_titles = ["ALL group distances", "INTRA-group (same compound) distances", "INTER-group (diff compound) distances"]
    k1_list = ['all_grp', 'same_grp', 'diff_grp']
    k2_list = ['_mean',  '_median', '_stddev', '_min', '_max' ]
    num_samples = measurement.shape[0]//CPS
    grp_level = {0:np.zeros((num_samples,num_samples))}
    dm = dict()
    non_diag_elmnts = CPS *(CPS-1)
    for i in range(0, num_samples*CPS, CPS):
        for j in range(0, num_samples*CPS, CPS):
            if i == j:
                # print(f" compound sample {1+i//3:4d}   --  sum of dist errs: {measurement[i:i+3, j:j+3].sum():.4f}")
                grp_level[0][i//CPS,j//CPS] = measurement[i:i+CPS, j:j+CPS].sum() / non_diag_elmnts  ### 6.0 needs to be parameterized 
            else:
                grp_level[0][i//3,j//3] = measurement[i:i+CPS, j:j+CPS].mean()
    
    grp_level[1] = grp_level[0].diagonal()   ## same group
    grp_level[2] = grp_level[0][np.triu_indices(num_samples,k=1)] ## other_group
    
    # same_group  = grp_level[0].diagonal()   ## same group
    # other_group = grp_level[0][np.triu_indices(num_samples,k=1)]
 

    for idx,(subtitle, k1) in enumerate(zip(sub_titles, k1_list)):
        dm[k1+ k2_list[0]] = grp_level[idx].sum() / grp_level[idx].size
        dm[k1+ k2_list[1]] = np.median(grp_level[idx])
        dm[k1+ k2_list[2]] = np.std(grp_level[idx]) 
        dm[k1+ k2_list[3]] = grp_level[idx].min()
        dm[k1+ k2_list[4]] = grp_level[idx].max()
        # print(grp_level[idx])            
        # print()
        # print(dm[k1+'_mean'])
        # print()
        # print(dm[k1+'_median'])
        # print()
        # print(dm[k1+'_stddev'])
        # print()
        print(f"{idx} {subtitle:39s} :\t mean {dm[k1+'_mean']:8.4f}    median: {dm[k1+'_median']:8.4f}    stddev: {dm[k1+'_stddev'] :8.4f}"
              f"    min: {dm[k1+'_min'] :8.4f}    max: {dm[k1+'_max'] :8.4f}")
    print()
    return dm



def  compute_regression_metrics(mo):
    rm = SimpleNamespace()
    print(f" Compute Regression metrics for epoch {mo.epochs}")
    rm.epochs = mo.epochs
    rm.R2_score   = skm.r2_score(y_true = mo.y_true, y_pred = mo.y_pred)
    rm.mse_score  = skm.mean_squared_error(y_true = mo.y_true, y_pred = mo.y_pred)
    rm.rmse_score = skm.root_mean_squared_error(y_true = mo.y_true, y_pred = mo.y_pred)
    rm.mae_score  = skm.mean_absolute_error(y_true = mo.y_true, y_pred = mo.y_pred)
    # pearson_corr, pearson_p = sps.pearsonr(mo.y_true, mo.y_pred)
    return rm


def  compute_classification_metrics(mo):
    """
    mo : model outpust from 'run_model_on_test_data'
    """
    cm = SimpleNamespace()
    print(f" Compute Regression metrics for epoch {mo.epochs}") 
    cm.metrics = mo.metrics
    cm.accuracy = skm.accuracy_score(mo.labels, mo.y_pred)
    cm.roc_auc  = skm.roc_auc_score(mo.labels, mo.logits)
    cm.precision, cm.recall, cm.f1, _ = skm.precision_recall_fscore_support(mo.labels, mo.y_pred, average='binary', zero_division=0)
    cm.cls_report = skm.classification_report(mo.labels, mo.y_pred)
    (mo.labels == mo.y_pred).sum()
    
    # cm.test_accuracy = binary_accuracy(y_true=mo.labels, y_prob=mo.logits)
    # cm.test_f1 = binary_f1_score(y_true=mo.labels, y_prob=mo.logits)
    return cm


def compute_distance_metrics(sample_inputs, epochs = 0, metric ='euclidian',display = False ):
    assert metric in ['euclidian', 'cosine', 'correlation'], "metric must be one of ['euclidian', 'cosine', 'correlation']"
    print(f" Compute Distance {metric} distance metrics for epoch {epochs}") 
    titles = [f'INPUT features - {metric} distances - {epochs} epochs',
              f'EMBEDDED features - {metric} distances - {epochs} epochs',
              f'RECONSTURCTED features -  {metric} distances - {epochs} epochs']
    sub_titles = ["ALL group distances", "INTRA-group (same compound) dist ", "INTER-group (diff compound) dist "]
    k0_list = ['inp', 'emb', 'out']
    k1_list = ['all', 'same', 'diff']
    k2_list = ['_min', '_max', '_avg', '_std',   ]
    dm = dict()
    grp_level = dict()
    cmp_level = dict()
    
    for idx0,(k0, sample_input) in enumerate(zip(k0_list, sample_inputs)):
        pairwise_distance = squareform(pdist(sample_input, metric))
        
        # print(sample_input.shape, pairwise_distance.shape)
        num_samples = pairwise_distance.shape[0]//3

        grp_level[k0] = {0:np.zeros((num_samples,num_samples)),
                         1:np.zeros((num_samples,num_samples)),
                         2:np.zeros((num_samples,num_samples))}
        cmp_level[k0] = {0:np.empty(0),     ## all distances
                         1:np.empty(0),     ## all same group distances
                         2: np.empty(0)}    ## all non-same group distances
        
        for i in range(0,num_samples*3,3):
            for j in range(0,num_samples*3,3):
                tile = pairwise_distance[i:i+3, j:j+3]
                if i == j: ## same group
                    upper_triang = tile[np.triu_indices(3,k=1)]
                    cmp_level[k0][1] = np.hstack((cmp_level[k0][1], upper_triang))
                    grp_level[k0][0][i//3,j//3] = upper_triang.mean()
                    # print(tile,'\n', upper_triang, '\n', cmp_level[1], '\n')
                    # print(f" compound sample {1+i//3:4d}   --  sum of dist errs: {measurement[i:i+3, j:j+3].sum():.4f}")
                    # grp_level[1][i//3,j//3] = upper_triang.std() 
                    # grp_level[2][i//3,j//3] = np.median(upper_triang) 
                # elif i < j:
                else:
                    cmp_level[k0][2] = np.hstack((cmp_level[k0][2], tile.reshape(-1)))
                    grp_level[k0][0][i//3,j//3] = tile.mean()
                    # print(tile, '\n',tile.reshape(-1) , '\n', cmp_level[2], '\n')
                    # grp_level[1][i//3,j//3] = tile.std() 

        triupper_indices = np.triu_indices(num_samples,k=1)
        trilower_indices = np.tril_indices(num_samples,k=-1)
        
        cmp_level[k0][0] = np.hstack((cmp_level[k0][1],cmp_level[k0][2]))
        ## Diagonal distances (same group)
        grp_level[k0][1] = grp_level[k0][0].diagonal()     ## averarge distances on same groups
        grp_level[k0][2] = np.hstack((grp_level[k0][0][triupper_indices],grp_level[k0][0][trilower_indices]))   ##  distances non on diagonals (different groups)
  
        
        # grp_level[4] = grp_level[1].diagonal()   ## stds on same group
        # print(grp_level[0], '\n\n\n', grp_level[0].diagonal(), '\n\n\n',grp_level[0][triupper_indices], '\n\n\n' , grp_level[0][trilower_indices], '\n\n\n') 
        # grp_level[6] = grp_level[0][triupper_indices] ## avg. distances on diff_group
        # print( grp_level[6], '\n\n\n' )
        # grp_level[7] = np.hstack((grp_level[1][triupper_indices],grp_level[1][trilower_indices])) ## std devs on diff_group  
        # print(f" upper: {grp_level[0][triupper_indices].std()}  lower: {grp_level[0][trilower_indices].std()}    upper+lower: {grp_level[6].std()} ")
        

        for idx1,(subtitle, k1) in enumerate(zip(sub_titles, k1_list)):
            key = k0+'_'+k1
            row = idx1   ## 0 (all groups) , 1: same grou, 2: diff_groups
            dm[key+ '_min'] = grp_level[k0][row].min()                           ## min(distances)
            dm[key+ '_max'] = grp_level[k0][row].max()                           ## max(avg_distances)
            dm[key+ '_avg'] = grp_level[k0][row].sum() / grp_level[k0][row].size     ## averages(distances)
            dm[key+ '_std'] = grp_level[k0][row].std()                           ## std_dev(distances)
            dm['CL_'+key+ '_avg'] = cmp_level[k0][idx1].mean()                   ## averages(ALL distances)
            dm['CL_'+key+ '_std'] = cmp_level[k0][idx1].std()                    ## std_dev(ALL distances)
    
    if display:
        display_dist_metrics(dm,epochs, metric)
    return dm, grp_level



