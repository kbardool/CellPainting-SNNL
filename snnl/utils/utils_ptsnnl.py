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
import wandb
from matplotlib import pyplot as plt
import scipy.stats as sps 
import sklearn.metrics as skm 
from scipy.spatial.distance import pdist, squareform, euclidean
from .dataloader import CellpaintingDataset, InfiniteDataLoader, custom_collate_fn
from KevinsRoutines.utils.utils_wandb import init_wandb, wandb_log_metrics,wandb_watch
logger = logging.getLogger(__name__) 


def parse_args(input = None):
    parser = argparse.ArgumentParser(description="DNN classifier with SNNL")
    grp = parser.add_argument_group("Parameters")

    grp.add_argument("--config","--configuration", type=str, dest="configuration", required=True, help=" yaml file containing hyperparameters to use")
    grp.add_argument('--ckpt', type=str, required=False, default=None, help="Checkpoint fle to resume training from")
    grp.add_argument('--cpb', type=int, required=True, default=0, help="Compounds per batch" )    
    grp.add_argument('--exp_title',type=str, required=False, default=None, help="Exp Title (overwrites yaml file value)")
    grp.add_argument('--epochs', type=int, required=True, default=0, help="epochs to run")
    grp.add_argument('--gpu_id', type=int, required=False, default=0, help="Cuda device id to use" )    
    grp.add_argument('--lr', type=float, required=False, default=None, dest='learning_rate', help="Learning Rate" )    
    grp.add_argument('--run_id', type=str, required=False, default=None, dest='exp_id',  help="WandB run id (for run continuations)")
    grp.add_argument("--runmode", type=str, required=False, choices=['baseline', 'snnl'], default="base",
                     help="the model running mode: [baseline (default) | snnl]")
    grp.add_argument("--seed", type=int, required=False, default=1234, dest='random_seed', help="the random seed value to use, default: [1234]")
    grp.add_argument('--prim_opt', default=True, dest="use_prim_optimizer", action=argparse.BooleanOptionalAction)
    grp.add_argument('--temp_opt', default=False, dest="use_temp_optimizer", action=argparse.BooleanOptionalAction)
    grp.add_argument('--temp_annealing', default=False, dest="use_annealing", action=argparse.BooleanOptionalAction)
    grp.add_argument('--anneal_patience', type=int, required=False, default=15, dest="anneal_patience", help="annealing patience")
    grp.add_argument('--single_loss', default=False, dest="use_single_loss", action=argparse.BooleanOptionalAction,
                     help="Optimize Primary and SNNL loss together (or seperately when = False)")
    grp.add_argument('--temp', type=float, required=False, default=None, dest='temperature', help="Temperature initial value" )
    grp.add_argument('--adam_wd', type=float, required=False, default=None, dest='adam_weight_decay', help="Temperature initial value" )
    grp.add_argument('--loss_factor', type=float, required=False, default=None, dest='loss_factor', help="Primary Loss Factor" )
    grp.add_argument('--snnl_factor', type=float, required=False, default=None, dest='snnl_factor', help="SNN Loss Factor " )
    grp.add_argument('--temp_lr', type=float, required=False, default=None, dest='temperatureLR', help="Temperature learning rate")
    grp.add_argument('--wandb', default=False, required=False, dest='WANDB_ACTIVE', action=argparse.BooleanOptionalAction)
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
        logger.info(f" command line param {k:25s} : [{v}]")
        if v is not None:
            _args[k] = v

    _args.setdefault('use_prim_optimizer', True)
    _args.setdefault('use_prim_scheduler', True)
    _args.setdefault('use_temp_optimizer', False)
    _args.setdefault('use_annealing', False)
    _args.setdefault('use_sum', False)
    _args.setdefault('SGD_momentum', 0)

    if _args['runmode'] == 'baseline':
        _args['use_single_loss'] = True
        ttl_pfx = 's'
    else:
        ttl_pfx = 'd'

    opt_ttl = 'DualOpt' if (_args['use_temp_optimizer'] ) else 'SnglOpt'

    _args['exp_title'] = _args['exp_title'].format(ttl_pfx = ttl_pfx,
                                                   cpb = _args['cpb'],
                                                   ltnt = _args['code_units'],
                                                   hidden_1 = _args['hidden_1'])

    _args['exp_description'] = _args['exp_description'].format(runmode = _args['runmode'],
                                                               optimizers = opt_ttl,
                                                               ltnt = _args['code_units'],
                                                               hidden_1 = _args['hidden_1'],
                                                               cpb = _args['cpb'],)

    _args['cellpainting_args']['compounds_per_batch'] = _args['cpb']
    _args['ckpt'] = input_params['ckpt']
    _args['batch_size'] = _args['cellpainting_args']['batch_size']


    if _args['ckpt'] is None:
        _args['exp_date'] = datetime.now().strftime('%Y%m%d_%H%M')
        _args['exp_name'] = f"AE_{_args['exp_date']}"
    else:
        ckpt_parse = _args['ckpt'].split('_')
        print(ckpt_parse)
        _args['exp_date'] = ckpt_parse[5]+'_' + ckpt_parse[6]
        _args['exp_name'] = ckpt_parse[0]+'_' + _args['exp_date']

    _args['use_temp_scheduler'] = _args.setdefault('use_temp_scheduler', False) if _args['use_temp_optimizer'] else False


    assert not (_args['use_annealing'] and _args['use_temp_optimizer']), " Temperature annealing and Temp optimization are mutually exclusive"

    logger.info(f" command line param {'exp_title':25s} : [{_args['exp_title']}]")
    logger.info(f" command line param {'exp_description':25s} : [{_args['exp_description']}]")

    # ars = types.SimpleNamespace(**args, **(vars(input_params)))
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


def take_checkpoint(model, args,epoch, update_best = False, update_latest = False):
    assert not (update_best and update_latest), "best and last cant both be True"
    if args.WANDB_ACTIVE:
        wandb.unwatch(model)
        args.save_checkpoint(epoch, model, args, update_best = update_best, update_latest=update_latest)
        wandb_watch(item = model, criterion=None, log = 'all', log_freq = 1000, log_graph = False)
    else:
        args.save_checkpoint(epoch, model, args, update_best = update_best, update_latest=update_latest)

# -------------------------------------------------------------------------------------------------------------------
#  Display routines
# -------------------------------------------------------------------------------------------------------------------
def display_model_summary(model, dataset = 'cellpainting', batch_size = 300 ):
    col_names = ["input_size", "output_size", "num_params", "params_percent", "mult_adds", "trainable"]  # "kernel_size"
    if dataset == "cellpainting":
        summary_input_size = (batch_size, 1471)
    else:
        summary_input_size = (batch_size, 28, 28)
    print(summary(model, input_size=summary_input_size, col_names = col_names))


def display_model_hyperparameters(model, args, title='Model Hyperparameters'):
    if title is not None:
        print(f"{title}")
        print('-'*(len(title)+1))
    print(f" Current device         : {args.current_device}")        
    print(f" Model device           : {model.device}")
    print(f" Model epoch/batch      : {model.epoch} / {model.batch_count}")

    print(f" Model embedding_layer  : {model.embedding_layer}")
    print(f" loss_factor            : {model.loss_factor}")
    print(f" monitor_grads_layer    : {model.monitor_grads_layer}")
    print(f" Use Single Loss        : {model.use_single_loss}")
    print(f" Use Prim Optimizer     : {model.use_prim_optimizer}")
    print(f" Use Prim Scheduler     : {model.use_prim_scheduler}")
    print()
    if model.use_prim_optimizer:
        print(f" Main Optimizer Params  : {model.optimizers['prim']}") 
        sch_stat = ''
        for k,v in model.schedulers['prim'].state_dict().items():
            sch_stat += f"\n    {k}: {v}  "
        print(f" Scheduler              : {model.schedulers['prim']} {sch_stat}") 

    print()
    print(f" SNN Loss ")
    print(f"----------")
    print(f" Use SNN Loss           : {model.use_snnl}") 
    print(f" SNNL factor            : {model.snnl_factor}")
    print(f" Use Annealing          : {model.use_annealing}")
    print(f" Use Single Loss        : {model.use_single_loss}")

    if isinstance(model.temperature, float) or isinstance(model.temperature, int):
        print(f" SNNL Temperature       : {model.temperature} ")
    else:
        print(f" SNNL Temperature       : value: {model.temperature.item()}, device:{model.temperature.device},  requires_grad:{model.temperature.requires_grad} ")
    print()
    print(f" Use Temp Optimizer     : {model.use_temp_optimizer}") 
    print(f" Temperature LR         : {model.temperatureLR}")
    if model.use_temp_optimizer:
        if model.use_single_loss:
            print(f" Main Optimizer LR Grp 1 Temp): {model.optimizers['prim'].param_groups[1]['lr']}")
        else:
            print(f" Temperature Optimizer Params : {model.optimizers['temp']}")

    print(f" Use Temp Scheduler     : {model.use_temp_scheduler}")
    if model.use_temp_scheduler:
        sch_stat = ''
        for k,v in model.schedulers['temp'].state_dict().items():
            sch_stat += f"\n    {k}: {v}  "
        print(f" Temp Scheduler         : {model.schedulers['temp']} {sch_stat}")

    if model.resume_training:
        print()
        print(f" Training History")
        print(f"-----------------")
        # if model.resume_training:
        for th_key in ['trn', 'val']:
            print(f" Last epoch : {len(model.training_history[th_key][th_key+'_time'])}")
            for k,v in model.training_history[th_key].items():
                if isinstance(v[-1],str):
                    print(f" {k:22s} : {v[-1]:s}  ")
                else:
                    print(f" {k:22s} : {v[-1]:6f} ")

        print(f" ")
        print(f" Best Loss / Metrics")
        print(f"--------------------")
        print(f" Best training loss     : {model.training_history['gen']['trn_best_loss']:<8.6f} - epoch: {model.training_history['gen']['trn_best_loss_ep']}") 
        print(f" Best training metric   : {model.training_history['gen']['trn_best_metric']:<8.6f} - epoch: {model.training_history['gen']['trn_best_metric_ep']}") 
        print(f" ") 
        print(f" Best validation loss   : {model.training_history['gen']['val_best_loss']:<8.6f} - epoch: {model.training_history['gen']['val_best_loss_ep']}") 
        print(f" Best validation metric : {model.training_history['gen']['val_best_metric']:<8.6f} - epoch: {model.training_history['gen']['val_best_metric_ep']}") 
        print(f" ")
        print(f" Model best trn metric  : {model.trn_best_metric:<8.6f} - epoch: {model.trn_best_epoch}") 
        print(f" Model best val metric  : {model.val_best_metric:<8.6f} - epoch: {model.val_best_epoch}") 

def display_model_gradients(model, msg = 'Model Gradients'):
    print(f"\n {model.epoch}/ {model.batch_count} - {msg}")
    for k,v in model.named_parameters():
        if v.grad is not None:
            g_str = f" {k[:15]:15s} | {v.grad.min():13.8f} | {v.grad.max():13.8f} | {v.grad.sum():15.8f}"
            # p_str = f" Parm:  Min: {v.min():15.12f} | Max: {v.max():15.12f} | Sum: {v.sum():18.12f}"
            if v.grad.ndim > 1:
                print(f"{g_str} |   {v[:3,:3].reshape((-1)).detach().cpu().numpy()}")
            else:
                print(f"{g_str} |   {v[:9].detach().cpu().numpy()}")
        else:
            print(f" {k[:15]:15s} | {'None':13s} | {'None':13s} | {'None':15s} |")


def display_model_parameters(model, msg = ' Model Parameters '):
    print(f"\n {model.epoch}/ {model.batch_count} - {msg}")
    for k,v in model.named_parameters():
        if v.ndim > 1:
            print(f" {k[:20]:20s} | {str(v.shape):<25s} | {v.requires_grad} | {v.sum():15.6f} | {v[:3,:3].reshape((-1)).data}")
        else:
            print(f" {k[:20]:20s} | {str(v.shape):<25s} | {v.requires_grad} | {v.sum():15.6f} | {v[:9].data}")


def display_model_state_dict(model, msg =' Model State Dict '):
    print(f"\n {model.epoch}/ {model.batch_count} - {msg}")
    for k,v in model.state_dict().items():
        if v.ndim > 1:
            print(f" {k[:20]:20s} | {str(v.shape):<25s} | {v.requires_grad} | {v.sum():15.6f} | {v[:3,:3].reshape((-1)).data}")
        else:
            print(f" {k[:20]:20s} | {str(v.shape):<25s} | {v.requires_grad} | {v.sum():15.6f} | {v[:9].data}")


def display_cellpainting_batch(batch_id, batch):
    data, labels, plates, compound_ids, cmphash, other = batch
    print("-"*135)
    print(f"  Batch Id: {batch_id}   {type(batch)}  Rows returned {len(batch[0])} features: {len(data[0])}  ")
    print(f"+-----+------------------------------------+----------------+----------------------------+---------------------------------+-----+--------------------------------------------------------+")
    print(f"| idx |   batch[2]                         |    batch[3]    |      batch[2]              |           batch[5]              | [1] |     batch[0]                                           | ") 
    print(f"|     | SRCE      BATCH     PLATE     WELL |   COMPOUND_ID  |       CMPHASH / BIN        |   TPSA  / Ln(TPSA) / Log(TPSA)  | LBL |     FEATURES                                           | ")
    print(f"+-----+------------------------------------+----------------+----------------------------+---------------------------------+-----+--------------------------------------------------------+")
         ###    0 | source_11 Batch2    EC000046  K04      | JCP2022_009278 |  7406361908543180200 -  8  |   0   |   62.78000    4.13964   1.79782 | [-0.4377299 -0.4474466  1.1898487  0.2051901]
         # "  1 | source_10    | JCP2022_006020 | -9223347314827979542 |   10 |  0 | tensor([-0.6346, -0.6232, -1.6046])"
    
    for i in range(len(labels)):
      # print(f"  {i:3d} | {plates[i,0]:12s} | {compounds_ids[i]:12s}  | {cmphash[i,0]:20d} | {cmphash[i,1]:4d}   |  {int(labels[i]):2d}   | {data[i,:3]}")
        print(f"| {i:3d} | {batch[2][i,0]:9s} {batch[2][i,1][:9]:9s} {batch[2][i,2][:9]:9s} {batch[2][i,3]:>4s} | {batch[3][i]:12s} | {batch[4][i,0]:20d} - {batch[4][i,1]:2d}  |"\
              f"{batch[5][i,0]:11.5f}   {batch[5][i,1]:8.5f}  {batch[5][i,2]:8.5f} |  {int(batch[1][i]):1d}  | {batch[0][i,:4].detach().cpu().numpy()}")


def display_epoch_metrics(model, epoch = None, epochs = None, header = False):
    # key1, key2 = model.training_history.keys()
    key1 = 'trn' ##if key1 == 'trn' else ''
    key2 = 'val' ##if key1_p == 'trn' else ''

    history_len = len(model.training_history[key1][f'{key1}_ttl_loss'])
    epochs = history_len if epochs is None else epochs
    epoch  = 0 if epoch is None else epoch
    header = True if epoch == 0 else header

    idx = epoch
    if idx >= epochs:
        return
    if model.use_snnl:
        temp_hist = model.training_history[key1]['temp_hist'][idx]
        temp_grad_hist = model.training_history[key1]['temp_grad_hist'][idx]
        temp_LR = model.training_history[key1]["temp_lr"][idx] 
    else:
        temp_hist = 0
        temp_grad_hist = 0
        temp_LR = 0
 
    trn_LR = model.training_history[key1]["trn_lr"][idx] if model.use_prim_scheduler else 0.0
 
    if model.unsupervised:
        if header:
            print(f"  time   ep / eps |  Trn_loss   Primary      SNNL  |   temp*         grad    |   R2      BestEp         |  Vld_loss   Primary      SNNL  |   R2       BestEp          |   LR        temp LR    |")
            print(f"------------------+--------------------------------+-------------------------+--------------------------+--------------------------------+----------------------------|------------------------|")
                 # "00:45:46 ep   1 / 10 |   9.909963    4.904229    5.005733 |  14.996347   -2.6287e-10 |                          |   9.833426    4.827625    5.005800 |                          |"
        print(f"{model.training_history[key2][f'{key2}_time'][idx]} {epoch + 1:^3d}/{epochs:^4d} |"
              f" {model.training_history[key1][f'{key1}_ttl_loss'][idx]:8.4f}   {   model.training_history[key1][f'{key1}_prim_loss'][idx]:8.4f}   {   model.training_history[key1][f'{key1}_snn_loss'][idx]:8.4f} |"
              f" {temp_hist:10.6f}  {temp_grad_hist:11.4e} |"
              f" {model.training_history[key1][f'{key1}_R2_score'][idx]:8.4f}   {model.training_history['gen'].get('trn_best_metric_ep',0)+1:3d}           |"
              f" {model.training_history[key2][f'{key2}_ttl_loss'][idx]:8.4f}   {   model.training_history[key2][f'{key2}_prim_loss'][idx]:8.4f}   {   model.training_history[key2][f'{key2}_snn_loss'][idx]:8.4f} |"
              f" {model.training_history[key2][f'{key2}_R2_score'][idx]:8.4f}   {model.training_history['gen'].get('val_best_metric_ep',0)+1:3d}             |"
              f" {trn_LR :10.3e}  {temp_LR :10.3e} |")

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


def save_checkpoint_v1(epoch, model, filename, update_latest=False, update_best=False):
    model_checkpoints_folder = os.path.join("ckpts")
    if not os.path.exists(model_checkpoints_folder):
        print(f"path {model_checkpoints_folder} doesn't exist")
    checkpoint = {'epoch': epoch,
                  'use_snnl': model.use_snnl,
                  'state_dict': model.state_dict,
                  'optimizer_state_dict': model.optimizer.state_dict(),
                  'temp_optimizer_state_dict': model.temp_optimizer.state_dict(),
                 }

    # checkpoint['scheduler'] =  model.scheduler.state_dict() if model.use_prim_scheduler else None
    # checkpoint['temp_scheduler'] =  model.temp_scheduler.state_dict() if model.use_temp_scheduler else None 

    if update_latest:
        filename = os.path.join(model_checkpoints_folder, f"{filename}_model_latest.pt")
    elif update_best:
        filename = os.path.join(model_checkpoints_folder, f"{filename}_model_best.pt")
    else:
        filename = os.path.join(model_checkpoints_folder, f"{filename}.pt")
    torch.save(checkpoint, filename) 
    logger.info(f" Model exported to {filename}.")


def load_checkpoint_v1(model, filename, verbose = False ):
    epoch = 9999
    try:
        checkpoints_folder = os.path.join("ckpts")
        checkpoint = torch.load(os.path.join(checkpoints_folder, filename))
        epoch = checkpoint.get('epoch',0)
        logger.info(f" ==> Loading from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}\n")
        if verbose:
            print(checkpoint.keys())
            print(" --> load model state_dict")
        model.load_state_dict(checkpoint['state_dict'])
        if verbose:
            print(" --> load optimizer state_dict")
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if "scheduler" in checkpoint and (hasattr(model, 'scheduler')):
            model.scheduler = checkpoint['scheduler']
        logger.info(f" ==> Loaded from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}\n")
        logger.info(f"     Loaded model device    : {model.device}") 

    # except FileNotFoundError:
    #     Exception("Previous state checkpoint not found.")
    except :
        print(sys.exc_info())

    return model, epoch


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
                  'scheduler_state_dict'      : model.scheduler.state_dict() if model.use_prim_scheduler else None,
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
 
    filename = os.path.join(model_checkpoints_folder, f"{filename}.pt")
    torch.save(checkpoint, filename) 
    logger.info(f" Model exported to {filename}.")


def load_checkpoint_v2(model, filename, dryrun = False, verbose=False):
 
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

    epoch = checkpoint.get('epoch',-1)
    logger.info(f" ==> Loading from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")
    for key, value in checkpoint.items():
        logging.info(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
    
    if dryrun:
        for key, value in checkpoint['params'].items():
            logging.debug(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
    else:
        model.load_state_dict(checkpoint['state_dict'])
        if verbose:
            print(f"model state dict:\n {checkpoint['state_dict'].keys()}")
            print(f"   temperature  :   {checkpoint['state_dict']['temperature'].item()}")
            print(f"   snnl_criterion.temperature: {checkpoint['state_dict']['snnl_criterion.temperature'].item()}")

        for key, value in checkpoint['params'].items():
            logging.debug(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
            model.__dict__[key] = value
            
        if model.optimizer is not None:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if verbose:
                print(f"optimizer state dict:\n {checkpoint['optimizer_state_dict']['param_groups']}")

        if model.scheduler is not None:
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if verbose:
                print(f"scheduler state dict:\n  {checkpoint['scheduler_state_dict']}")
            
        
        if model.temp_optimizer is not None:
            model.temp_optimizer.load_state_dict(checkpoint['temp_optimizer_state_dict'])
            if verbose:
                print(f"temp optimizer state dict:\n  {checkpoint['temp_optimizer_state_dict']['param_groups']}")
            
        if model.temp_scheduler is not None:
            model.temp_scheduler.load_state_dict(checkpoint['temp_scheduler_state_dict'])
            if verbose:
                print(f"temp scheduler state dict:\n  {checkpoint['temp_scheduler_state_dict']}")

    logger.info(f" ==> Loaded from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")
    logger.info(f"     Model best metric      : {model.best_metric:6f} - epoch: {model.best_epoch}") 
    logger.info(f"     Loaded model device    : {model.device}") 

    if 'gen' not in model.training_history:
        print(f" Define self.training_history['gen'] ")
        model.training_history['gen'] = {'trn_best_metric' : 0, 'trn_best_metric_ep' : 0, 'trn_best_loss': np.inf, 'trn_best_loss_ep' : 0 ,
                                        'val_best_metric' : 0, 'val_best_metric_ep' : 0, 'val_best_loss': np.inf, 'val_best_loss_ep' : 0 }        
    
        for key in ['trn', 'val']:
            tmp = np.argmin(model.training_history[key][f'{key}_ttl_loss'])
            model.training_history['gen'][f'{key}_best_loss_ep'] = tmp
            model.training_history['gen'][f'{key}_best_loss']    = model.training_history[key][f'{key}_ttl_loss'][tmp]
            
            tmp1 = np.argmax(model.training_history[key][f'{key}_R2_score'])
            model.training_history['gen'][f'{key}_best_metric_ep'] = tmp1
            model.training_history['gen'][f'{key}_best_metric'] = model.training_history[key][f'{key}_R2_score'][tmp1]
            
        model.best_metric = model.training_history['gen'][f'val_best_metric']  
        model.best_epoch  = model.training_history['gen'][f'val_best_metric_ep']              
        
    return model, epoch


def save_checkpoint_v3(epoch, model, args, filename = None, update_latest=False, update_best=False, verbose = False):
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
                  'scheduler_state_dict'      : model.scheduler.state_dict() if model.use_prim_scheduler else None,
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
    
    if filename is None: 
        filename = f"{model.name}_{args.runmode[:4]}_{args.exp_title}_{args.exp_date}"      
        
        if update_latest:
            s_filename = f"LAST_ep_{epoch:03d}"
        elif update_best:
            s_filename = f"BEST"
        else:
            s_filename = f"ep_{epoch:03d}"
        filename = f"{filename}_{s_filename}"
        
    filename = os.path.join(model_checkpoints_folder, f"{filename}.pt")
    torch.save(checkpoint, filename) 
    logger.info(f" Model exported to {filename} - epoch: {epoch}")


load_checkpoint_v3 = load_checkpoint_v2


def save_checkpoint_v4(epoch, model, args, filename = None, update_latest=False, update_best=False, verbose = False):
    from types import NoneType
    model_checkpoints_folder = os.path.join("ckpts")
    if not os.path.exists(model_checkpoints_folder):
        print(f"path {model_checkpoints_folder} doesn't exist")
        
    checkpoint = {'epoch'                     : epoch,
                  'state_dict'                : model,
                  'optimizers'                : {k:v for k,v in model.optimizers.items()},
                #   'optimizers_state_dict'     : {k:v.state_dict() for k,v in model.optimizers.items()},
                  'schedulers'                : {k:v for k,v in model.schedulers.items()},
                #   'schedulers_state_dict'     : {k:v.state_dict() for k,v in model.schedulers.items()},
                  
                #   'optimizer_state_dict'      : model.optimizer.state_dict() if model.optimizer is not None else None,
                #   'temp_optimizer_state_dict' : model.temp_optimizer.state_dict() if model.temp_optimizer is not None else None,
                #   'scheduler_state_dict'      : model.scheduler.state_dict() if model.use_prim_scheduler else None,
                #   'temp_scheduler_state_dict' : model.temp_scheduler.state_dict() if model.use_temp_scheduler else None ,
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
    
    if filename is None: 
        filename = f"{model.name}_{args.runmode[:4]}_{args.exp_title}_{args.exp_date}"      
        
        if update_latest:
            s_filename = f"LAST_ep_{epoch:03d}"
        elif update_best:
            s_filename = f"BEST"
        else:
            s_filename = f"ep_{epoch:03d}"
        filename = f"{filename}_{s_filename}"
        
    filename = os.path.join(model_checkpoints_folder, f"{filename}.pt")
    torch.save(checkpoint, filename) 
    logger.info(f" Model exported to {filename} - epoch: {epoch}")


def load_checkpoint_v4(model, filename, dryrun = False, verbose=False):
 
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

    epoch = checkpoint.get('epoch',-1)
    logger.info(f" ==> Loading from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")
    for key, value in checkpoint.items():
        logging.info(f"{key:40s}, {str(type(value)):60s}")
    
    if dryrun:
        for key, value in checkpoint['params'].items():
            logging.debug(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
    else:
        # model = checkpoint['state_dict']
        if "model" in checkpoint:
            print(f" model entry in checkpoint ")
            model = checkpoint['model']
        else:
            print(f" model state_dict entry in checkpoint ")
            model.load_state_dict(checkpoint['state_dict'])
            
        model.optimizers = {k:v for k,v in checkpoint['optimizers'].items()}
        model.schedulers = {k:v for k,v in checkpoint['schedulers'].items()}

        for key, value in checkpoint['params'].items():
            logging.debug(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
            model.__dict__[key] = value     
        
        # if verbose:
            # print(f"model state dict:\n {checkpoint['state_dict']}\n")
            # for k,v in model.named_parameters():
            #     if v.requires_grad == False:
            #         v.requires_grad_()
            #         print(f" set {k} to requires_grad = True {v.requires_grad}")
            # display_model_parameters(model, 'loaded named parameters')
            # print(f"   temperature  :   {checkpoint['state_dict']['temperature'].item()}")
            # print(f"   snnl_criterion.temperature: {checkpoint['state_dict']['snnl_criterion.temperature'].item()}")

        # for k,v in checkpoint['optimizers_state_dict'].items():
        #     model.optimizers[k].load_state_dict(v)
        #     if verbose:
        #         print(f"optimizer state dict:\n {v['param_groups']}")

        # for k,v in checkpoint['schedulers_state_dict'].items():
        #     model.schedulers[k].load_state_dict(v)
        #     if verbose:
        #         print(f"scheduler state dict:\n  {v}")           
        
        # if model.optimizer is not None:
        #     model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # if model.scheduler is not None:
        #     model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #     if verbose:
        #         print(f"scheduler state dict:\n  {checkpoint['scheduler_state_dict']}")           
        
        # if model.temp_optimizer is not None:
        #     model.temp_optimizer.load_state_dict(checkpoint['temp_optimizer_state_dict'])
        #     if verbose:
        #         print(f"temp optimizer state dict:\n  {checkpoint['temp_optimizer_state_dict']['param_groups']}")
            
        # if model.temp_scheduler is not None:
        #     model.temp_scheduler.load_state_dict(checkpoint['temp_scheduler_state_dict'])
        #     if verbose:
        #         print(f"temp scheduler state dict:\n  {checkpoint['temp_scheduler_state_dict']}")


    if 'gen' not in model.training_history:
        print(f" Define self.training_history['gen'] ")
        model.training_history['gen'] = {'trn_best_metric' : -np.inf, 'trn_best_metric_ep' : -1, 'trn_best_loss': np.inf, 'trn_best_loss_ep' : -1 ,
                                        'val_best_metric' : -np.inf, 'val_best_metric_ep' : -1, 'val_best_loss': np.inf, 'val_best_loss_ep' : -1 }   
    
        for key in ['trn', 'val']:
            tmp = np.argmin(model.training_history[key][f'{key}_ttl_loss'])
            model.training_history['gen'][f'{key}_best_loss_ep'] = tmp
            model.training_history['gen'][f'{key}_best_loss']    = model.training_history[key][f'{key}_ttl_loss'][tmp]
            
            tmp1 = np.argmax(model.training_history[key][f'{key}_R2_score'])
            model.training_history['gen'][f'{key}_best_metric_ep'] = tmp1
            model.training_history['gen'][f'{key}_best_metric'] = model.training_history[key][f'{key}_R2_score'][tmp1]

        model.trn_best_metric = model.training_history['gen'][f'trn_best_metric']  
        model.trn_best_epoch  = model.training_history['gen'][f'trn_best_metric_ep']                          
        model.val_best_metric = model.training_history['gen'][f'val_best_metric']  
        model.val_best_epoch  = model.training_history['gen'][f'val_best_metric_ep']              
    
    logger.info(f" ==> Loaded from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")
    logger.info(f"     Model best training metric   : {model.trn_best_metric:6f} - epoch: {model.trn_best_epoch}") 
    logger.info(f"     Model best validation metric : {model.val_best_metric:6f} - epoch: {model.val_best_epoch}") 
        
    return model, epoch

# def load_checkpoint_v41(model, filename, dryrun = False, verbose=False):
#     """
#     load model state dict, params, optimizer state dict and scheduler state dict. 
#     Temp routine for 20240709_2235 run - should be remomved once v5 is verfied to be working correctly
#     """
#     if filename[-3:] != '.pt':
#         filename+='.pt'
#     logging.info(f" Load model checkpoint from  {filename}")    
#     ckpt_file = os.path.join("ckpts", filename)
    
#     try:
#         checkpoint = torch.load(ckpt_file)
#     except FileNotFoundError:
#         Exception("Previous state checkpoint not found.")
#         print("FileNotFound Exception")
#     except :
#         print("Other Exception")
#         print(sys.exc_info())

#     epoch = checkpoint.get('epoch',-1)
#     logger.info(f" ==> Loading from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")
#     for key, value in checkpoint.items():
#         print(f" {key:30}  tpye: {type(value)}")

#     model.load_state_dict(checkpoint['state_dict'])
#     # model.optimizers = {k:v for k,v in checkpoint['optimizers'].items()}
#     # model.schedulers = {k:v for k,v in checkpoint['schedulers'].items()}

#     for key, value in checkpoint['params'].items():
#         print(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
#         model.__dict__[key] = value     

#     for k,v in checkpoint['optimizers_state_dict'].items():
#         model.optimizers[k].load_state_dict(v)
#         if verbose:
#             print(f"optimizer state dict:\n {v['param_groups']}")

#     for k,v in checkpoint['schedulers_state_dict'].items():
#         model.schedulers[k].load_state_dict(v)
#         if verbose:
#             print(f"scheduler state dict:\n  {v}")           
    
    
#     # if verbose:
#     print(f"model state dict:\n {checkpoint['state_dict']}\n")
#     for k,v in model.named_parameters():
#         if v.requires_grad == False:
#             print(f" set {k} to requires_grad = True {v.requires_grad}")
#             # v.requires_grad_()
#             # print(f" set {k} to requires_grad = True {v.requires_grad}")
            
#     display_model_parameters(model, 'loaded named parameters')
#     print(f"   temperature  :   {checkpoint['state_dict']['temperature'].item()}")
#     print(f"   snnl_criterion.temperature: {checkpoint['state_dict']['snnl_criterion.temperature'].item()}")


#     if 'gen' not in model.training_history:
#         print(f" GEN not in model.training_history ! ")
#         return None,-1
    
#     logger.info(f" ==> Loaded from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")
#     logger.info(f"     Model best training metric   : {model.trn_best_metric:6f} - epoch: {model.trn_best_epoch}") 
#     logger.info(f"     Model best validation metric : {model.val_best_metric:6f} - epoch: {model.val_best_epoch}") 
        
#     return model, epoch


def save_checkpoint_v5(epoch, model, args, filename = None, update_latest=False, update_best=False, verbose = False):
    from types import NoneType
    model_checkpoints_folder = os.path.join("ckpts")
    if not os.path.exists(model_checkpoints_folder):
        print(f"path {model_checkpoints_folder} doesn't exist")
        
    checkpoint = {'epoch'      : epoch,
                  'model'      : model,
                  'optimizers' : {k:v for k,v in model.optimizers.items()},
                  'schedulers' : {k:v for k,v in model.schedulers.items()},
                  'params'     : dict()
                #   'optimizers_state_dict'     : {k:v.state_dict() for k,v in model.optimizers.items()},
                #   'schedulers_state_dict'     : {k:v.state_dict() for k,v in model.schedulers.items()},
                #   'optimizer_state_dict'      : model.optimizer.state_dict() if model.optimizer is not None else None,
                #   'temp_optimizer_state_dict' : model.temp_optimizer.state_dict() if model.temp_optimizer is not None else None,
                #   'scheduler_state_dict'      : model.scheduler.state_dict() if model.use_prim_scheduler else None,
                #   'temp_scheduler_state_dict' : model.temp_scheduler.state_dict() if model.use_temp_scheduler else None ,
                 }
    ## save model attributes 
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
    if filename is None: 
        filename = f"{model.name}_{args.runmode[:4]}_{args.exp_title}_{args.exp_date}"

        if update_latest:
            s_filename = f"LAST_ep_{epoch:03d}"
        elif update_best:
            s_filename = f"BEST"
        else:
            s_filename = f"ep_{epoch:03d}"
        filename = f"{filename}_{s_filename}"

    filename = os.path.join(model_checkpoints_folder, f"{filename}.pt")
    torch.save(checkpoint, filename) 
    logger.info(f" Model exported to {filename} - epoch: {epoch}")


def load_checkpoint_v5(model, filename, dryrun = False, verbose=False, quiet = False):

    if filename[-3:] != '.pt':
        filename += '.pt'

    logging.info(f"Load model checkpoint from  {filename}")
    ckpt_file = os.path.join("ckpts", filename)

    try:
        checkpoint = torch.load(ckpt_file)
    except FileNotFoundError:
        Exception("Previous state checkpoint not found.")
        print("FileNotFound Exception")
    except Exception as e:
        print("Other Exception")
        print(sys.exc_info())
    if verbose:
        print(f" checkpoint keys: {checkpoint.keys()}")

    epoch = checkpoint.get('epoch',-1)

    if dryrun:
        for key, value in checkpoint['params'].items():
            logging.info(f"{key:40s}, {str(type(value)):60s}  -- model attr set")
        # model.load_state_dict(checkpoint['state_dict'])
    else:
        model = checkpoint['model']

        model.optimizers = {k : v for k,v in checkpoint['optimizers'].items()}
        model.schedulers = {k : v for k,v in checkpoint['schedulers'].items()}

    if verbose:
        for key, value in checkpoint.items():
            logging.info(f"{key:40s}, {str(type(value)):60s} ")
        for k in ['trn', 'val']:
            print(k)
            for kk,vv in model.training_history[k].items():
                print(f" {k}-{kk}   checkpoint len: {len(vv)} ")
        print(f"model :\n {checkpoint['model']}\n")
        # for k,v in model.named_parameters():
        #     if v.requires_grad == False:
        #         v.requires_grad_()
        #         print(f" set {k} to requires_grad = True {v.requires_grad}")
        display_model_hyperparameters(model, ' Loaded hyperparameters ')
        display_model_parameters(model, 'loaded named parameters')

        # for k,v in checkpoint['optimizers_state_dict'].items():
        #     model.optimizers[k].load_state_dict(v)
        #     if verbose:
        #         print(f"optimizer state dict:\n {v['param_groups']}")

        # for k,v in checkpoint['schedulers_state_dict'].items():
        #     model.schedulers[k].load_state_dict(v)
        #     if verbose:
        #         print(f"scheduler state dict:\n  {v}")

    # logger.info(f" ==> Loading from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")
    logger.info(f" Loaded from checkpoint {filename} successfully. last epoch on checkpoint: {epoch}")
    logger.info(f" Model best training metric   : {model.trn_best_metric:6f} - epoch: {model.trn_best_epoch+1}") 
    logger.info(f" Model best validation metric : {model.val_best_metric:6f} - epoch: {model.val_best_epoch+1}") 

    return model, epoch


def load_model_from_ckpt(model, args = None, runmode = None, date = None, title = None, epochs = None, 
                         filename =None, cpb = None, factor = None,
                         dryrun = False, 
                         cuda_device = None,
                         verbose = False):
    # filename = f"AE_{args.model.lower()}_{date}_{title}_{epochs:03d}_cpb_{args.compounds_per_batch}_factor_{factor}.pt"    
    if filename is None:
        if factor is None:
            filename = f"{model.name}_{runmode}_{date}_{title}_ep_{epochs:03d}.pt"
        else:
            filename = f"{model.name}_{runmode}_{date}_{title}_{epochs:03d}_cpb_{cpb}_factor_{factor:d}.pt"
        print(filename)

    if os.path.exists(os.path.join('ckpts', filename)):
        model, last_epoch = args.load_checkpoint(model, filename, dryrun,verbose = verbose)

        _ = model.eval()

        if cuda_device is not None:
            model.device = cuda_device
            # model = model.cuda(device=cuda_device)
            model = model.to(cuda_device)

        # model.to('cpu')
        logger.info(f" Loaded model device     : {model.device}")
        if args.runmode == 'snnl':
            logger.info(f" Loaded model temperature: {model.temperature.item():.6f}")
    else:
        logger.error(f" {filename} *** Checkpoint DOES NOT EXIST *** \n")
        raise ValueError(f"\n {filename} *** Checkpoint DOES NOT EXIST *** \n")

    return model, last_epoch



def check_checkpoints(filenames, verbose=False):
    if not isinstance(filenames, list):
        filenames = [filenames]

    for filename in filenames:
        if filename[-3:] != '.pt':
            filename += '.pt'
        ckpt_file = os.path.join("ckpts", filename)
        try:
            checkpoint = torch.load(ckpt_file)
        except FileNotFoundError:
            Exception("Previous state checkpoint not found.")
            print(f" FileNotFound Exception: {ckpt_file}")
        except :
            print("Other Exception")
            print(sys.exc_info())
        else:
            print(f" File Exists: {filename:60s}   Checkpoint epoch: {checkpoint['epoch']}   Checkpoint keys: {checkpoint.keys()}")

# -------------------------------------------------------------------------------------------------------------------
#  Plotting routines
# -------------------------------------------------------------------------------------------------------------------     
def plot_train_history(model, epochs= None, start= 0, n_bins = 25):
    key1 = 'trn' 

    if epochs is None:
        epochs = len(model.training_history[key1]['trn_ttl_loss'])
    fig, axs = plt.subplots(1, 5, sharey=False, tight_layout=True, figsize=(5*4,4) )
    x_data = np.arange(start,epochs)
    # print(f" epochs : {epochs}  x_data:{x_data}")
    labelsize = 6
    # We can set the number of bins with the *bins* keyword argument.
    i = 0
    _ = axs[i].plot(model.training_history['trn']['trn_ttl_loss'][start:epochs], label='Training');
    _ = axs[i].plot(model.training_history['val']['val_ttl_loss'][start:epochs], label='Validation');
    _ = axs[i].set_title(f'Total loss - epochs {start}-{epochs}', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    axs[i].legend()
    i +=1
    _ = axs[i].plot(model.training_history['trn']['trn_prim_loss'][start:epochs], label='Training');
    _ = axs[i].plot(model.training_history['val']['val_prim_loss'][start:epochs], label='Validation');
    _ = axs[i].set_title(f'Primary loss - epochs {start}-{epochs}', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    i +=1
    _ = axs[i].plot(model.training_history['trn']['trn_snn_loss'][start:epochs], label='Training');
    _ = axs[i].plot(model.training_history['val']['val_snn_loss'][start:epochs], label='Validation');
    _ = axs[i].set_title(f'Soft Nearest Neighbor Loss - epochs {start}-{epochs}', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    i +=1
    _ = axs[i].plot(model.training_history[key1]['trn_lr'][start:epochs], label='Prim_LR');
    _ = axs[i].set_title(f'Primary Loss Learning Rate - epochs {start}-{epochs}', fontsize= 10);
    _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)
    if model.use_snnl:
        _ = axs[i].plot(model.training_history[key1]['temp_lr'][start:epochs], label='Temp_LR');
        _ = axs[i].set_title(f'Prim/Temp Loss LR - epochs {start}-{epochs}', fontsize= 10);
    if model.use_snnl:
        i +=1
        _ = axs[i].plot(model.training_history[key1]['temp_hist'][start:epochs], label='Temperature');
        _ = axs[i].tick_params(axis='both', which='major', labelsize=6, labelrotation=45)    # i +=1
        _ = axs[i].set_title(f'train_temp_hist - epochs {start}-{epochs}', fontsize= 10);

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


def plot_classification_metrics(model, epochs= None, n_bins = 25):
    key1 = 'trn' 
 
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
    
    roc_display = skm.RocCurveDisplay.from_predictions(cm.labels, cm.logits, pos_label= 1, ax = ax[1])
    _ = roc_display.ax_.set_title(f" ROC Curve - epoch:{cm.epochs} ")
    
    cm_display = skm.ConfusionMatrixDisplay.from_predictions(y_true = cm.labels, y_pred =cm.y_pred, ax = ax[2], colorbar = False)
    _ = cm_display.ax_.set_title(f" Confusion matrix - epoch:{cm.epochs} ");
    
    plt.show()


def plot_regression_metrics(model, epochs= None, start = 0, n_bins = 25):
    plots = 1
    if epochs is None:
        epochs = len(model.training_history['trn']['trn_ttl_loss'])
    fig, axs = plt.subplots(1, plots, sharey=False, tight_layout=True, figsize=(plots*4,4) )
    i = 0
    _ = axs.plot(model.training_history['trn']['trn_R2_score'][start:epochs]);
    _ = axs.plot(model.training_history['val']['val_R2_score'][start:epochs]);
    _ = axs.set_title(f'R2 Score - {epochs} epochs', fontsize= 10);
    _ = axs.tick_params(axis='both', which='major', labelsize=7, labelrotation=45)
    

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


def plot_TSNE(prj, lbl, cmp, key = None, layers= [0,1,2,3,4], start = 0, end = None, epoch = 0, limits=(None,None) ):
    if end is None:
        end = len(lbl)
    fig, axs = plt.subplots(1, 5, sharey=False, tight_layout=True, figsize=(20,4))

    for idx, layer in enumerate(layers):
        df = pd.DataFrame(dict(
            x=prj[layer][start:end,0],
            y=prj[layer][start:end,1],
            tpsa=lbl[start:end],
            compound = cmp[start:end]
        ))
        # print(key, np.bincount(lbl), df[key].unique() , palette_count)
        palette_count = len(df[key].unique())
        legend = True  # # if layer in [0,4] else False
        lp = sb.scatterplot(data=df, x ="x", y = "y", hue=key, palette=sb.color_palette(n_colors=palette_count), ax=axs[idx], legend = legend)  # , size=6)
        _ = lp.set_title(f'Epoch: {epoch} layer {layer}', fontsize = 10)
        lp.set_xlim([limits[0], limits[1]])
        lp.set_ylim([limits[0], limits[1]])    
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
        lp = sb.scatterplot(data=df, x ="x", y = "y", hue=key, palette=sb.color_palette(n_colors=palette_count), ax=axs[idx])   # size=6)
        _ = lp.set_title(f'Epoch: {epoch} layer {layer}', fontsize = 10)
        lp.legend(loc = 'best', fontsize = 8)
        lp.set_xlim([limits[0], limits[1]])
        lp.set_ylim([limits[0], limits[1]])

    plt.show()
    return fig


