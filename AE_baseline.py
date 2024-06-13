# Autoencoder with Soft Nearest Neighbor Loss
# Copyright (C) 2024  Kevin Bardool
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
__author__ = "Kevin Bardool"
__version__ = "1.0.0"
"""Sample module for using Autoencoder with SNNL"""
 

# %%
import os
import sys
import csv
import json
import time
import types, copy, pprint
import logging 
from datetime import datetime
for p in ['./src','../..']:
    if p not in sys.path:
        # print(f"insert {p}")
        sys.path.insert(0, p)

# import shutil
# import getpass
import yaml

from matplotlib import pyplot as plt
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torchinfo import summary

# logger = logging.getLogger('AutoEncoder')

torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=150, profile=None, sci_mode=None)
pp = pprint.PrettyPrinter(indent=4)
pd.options.display.width = 132
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=150, profile=None, sci_mode=None)
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan')


# os.environ["WANDB_NOTEBOOK_NAME"] = "Autoencoder_dev.ipynb"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import wandb
from KevinsRoutines.utils.utils_wandb    import  init_wandb, wandb_log_metrics,wandb_watch
from KevinsRoutines.utils.utils_general  import  list_namespace 

from snnl.utils import parse_args, get_hyperparameters, set_global_seed, get_device, set_device
from snnl.utils import display_epoch_metrics
from snnl.utils import CellpaintingDataset, InfiniteDataLoader, custom_collate_fn
from snnl.utils import display_model_summary, define_autoencoder_model
from snnl.utils import save_checkpoint_v2, load_checkpoint_v2
# from snnl.utils import load_model, save_model, import_results, export_results, save_checkpoint, load_checkpoint 
# from snnl.utils import plot_model_parms, plot_train_history, plot_classification_metrics, plot_regression_metrics

timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
logger = logging.getLogger(__name__) 
logLevel = os.environ.get('LOG_LEVEL', 'INFO').upper()
FORMAT = '%(asctime)s - %(name)s - %(levelname)s: - %(message)s'
logging.basicConfig(level="INFO", format= FORMAT)

logger.info(f" Excution started : {timestamp} ")
logger.info(f" Pytorch version  : {torch.__version__}")
logger.info(f" Search path      : {sys.path}")

# %%
#
# Parse input arguments 
#
# if __name__ == "__main__":
# input_args = f" --runmode            baseline" \
#              f" --configuration      hyperparameters/autoencoder_cellpainting-25.yaml"
args = parse_args()
current_device = get_device()
if args.gpu_id is not None:
    current_device = set_device(args.gpu_id)
    _ = get_device()

with open(args.configuration) as f:
    args = types.SimpleNamespace(**yaml.safe_load(f), **(vars(args)))
args.batch_size = args.cellpainting_args['batch_size']
args.compounds_per_batch = args.cellpainting_args['compounds_per_batch']
set_global_seed(args.seed)

if args.ckpt is not None:
    if os.path.exists(os.path.join('ckpts', args.ckpt)):
        # logging.info(f" Checkpoint {args.ckpt} found")
        logger.info(f" Resuming training using checkpoint: {args.ckpt}")
    else:
        logger.error(f" *** Checkpoint {args.ckpt} not found *** \n")
        raise ValueError(f"\n *** Checkpoint DOESNT EXIST *** \n")
        
# %% [markdown]
#
#  Define dataset and dataloaders
#
trn_file_sz = args.cellpainting_args['train_end'] - args.cellpainting_args['train_start']
val_file_sz = args.cellpainting_args['val_end'] - args.cellpainting_args['val_start']
smp_sz = args.cellpainting_args['sample_size']
buf_sz = args.cellpainting_args['compounds_per_batch']
bth_sz = args.cellpainting_args['batch_size']
recs_per_batch = smp_sz * bth_sz * buf_sz

for file_sz in [trn_file_sz, val_file_sz]:
    bth_per_epoch = file_sz // recs_per_batch
    print(f" - Each mini-batch contains {recs_per_batch/smp_sz} compounds with {smp_sz} samples per each compound : total {recs_per_batch} rows")
    print(f" - Number of {recs_per_batch} row full size batches per epoch: {bth_per_epoch}")
    print(f" - Rows covered by {bth_per_epoch} full size batches ({recs_per_batch} rows) per epoch:  {(file_sz // recs_per_batch) * recs_per_batch}")
    print(f" - Last partial batch contains : {file_sz % recs_per_batch} rows")
    print() 

## Load CellPainting Dataset
if args.dataset == 'cellpainting':
    print(f" load {args.dataset}")
    train_dataset = CellpaintingDataset(type='train',    **args.cellpainting_args)
    train_loader = InfiniteDataLoader(dataset=train_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 0, collate_fn = custom_collate_fn)
    val_dataset = CellpaintingDataset(type='val',    **args.cellpainting_args)
    val_loader = InfiniteDataLoader(dataset=val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 0, collate_fn = custom_collate_fn)

# %%
#
# WandB initialization 
#
WANDB_ACTIVE = args.wandb
# args.project_name = 'CellPainting_Profiles'
# args.exp_title = f"snglOpt-{args.code_units}Ltnt"
# args.exp_description = f"Autoencoder Training in Baseline mode - Single Optimizer, {args.code_units} dim latent layer"

if args.run_id is not None:
    resume_wandb = True
    args.exp_id   = args.run_id
else:
    resume_wandb = False
    args.exp_id   = None
    args.exp_name = 'AE_'+datetime.now().strftime('%m%d_%H%M')


if WANDB_ACTIVE:
    wandb_run = init_wandb(args)
    args.exp_date = '2024'+args.exp_name[3:7]
    logger.info(f" WandB tracking started - run id is : {args.exp_id}")
    logger.info(f" Experiment name {args.exp_name} - description: {args.exp_description}")
else: 
    args.exp_date = datetime.now().strftime('%Y%m%d')
    logger.info(f" *** W&&B Logging is INACTIVE *** ")


# %%
# ## Define Model

# ## Override arguments
# args.temperature   = 0.0
# args.loss_factor   = 1.0        ## 2.0e+00
# args.snnl_factor   = 0.0       ## 1.0e+00
# args.learning_rate = 1.0e-03    ## 0.001
# args.temperatureLR = 0.0e-00    ## 1e-4
# print(f"   Latent dim        {args.code_units}")
# print(f"   loss_factor       {args.loss_factor}")
# print(f"   snnl_factor       {args.snnl_factor}")
# print(f"   temperature       {args.temperature}")
# print(f"   learning_rate     {args.learning_rate}")
# print(f"   temperatureLR:    {args.temperatureLR}")

model = define_autoencoder_model(args, embedding_layer = 4, device = current_device)
list_namespace(args)

if WANDB_ACTIVE:
    wandb_watch(item = model, criterion=None, log = 'all', log_freq = 1000, log_graph = True)

# %%
#
# Load model checkpoint if provided 
#
if args.ckpt is not None:
    resume_training = True
    mdl , last_epoch = load_checkpoint_v2(model, args.ckpt)  
    model.train()
    model.device = current_device
    model = model.cuda(device=current_device)
    logging.info(f" Loaded Model device {model.device} -  Last completed epoch : {last_epoch}")
    starting_epoch = last_epoch
    epochs = last_epoch + args.epochs
    
else:
    resume_training = False
    starting_epoch = 0
    epochs = args.epochs
logging.info(f" Run epochs {starting_epoch+1} to {epochs} ")

print()
print(f" Current device      : {current_device}")
print(f" Model device        : {model.device}")
print(f" SNNL temperature    : {model.temperature}")
print(f" Learning rate       : {model.optimizer.param_groups[0]['lr']}") 
print(f" snnl_factor         : {model.snnl_factor}")
print(f" loss_factor         : {model.loss_factor}")
print(f" monitor_grads_layer : {model.monitor_grads_layer}")
print(f" Use Scheduler       : {model.use_scheduler}") 
print(f" Use snnl            : {model.use_snnl}") 
if model.use_snnl:
    print(f" Temperature         : {model.temperature.item()}")
    print(f" Temperature LR      : {model.optimizer.param_groups[1]['lr']}") 
    print(f" Use Temp Scheduler  : {model.use_temp_scheduler}") 
    print(f" Temp Scheduler      : {model.temp_scheduler}") 
    if model.temp_optimizer is not None:
        print(f" Temperature LR       : {model.temp_optimizer.param_groups[0]['lr']}") 
print()
if resume_training:    
    for th_key in ['trn', 'val']:
        for k,v in model.training_history[th_key].items():
            if isinstance(v[-1],str):
                print(f" {k:20s} : {v[-1]:s}  ")
            else:
                print(f" {k:20s} : {v[-1]:6f} ")
        print()    
print(f" {datetime.now().strftime('%Y%m%d_%H%M%S')}  epoch {starting_epoch+1:4d} of {epochs:4d}")
    
if WANDB_ACTIVE:
    wandb.config.update(args,allow_val_change=True )

#
#  Running Training Loop
#

header = True

for epoch in range(starting_epoch,epochs):
    train_loss = model.epoch_train(train_loader, epoch)
    val_loss = model.epoch_validate(val_loader, epoch)
    
    display_epoch_metrics(model, epoch, epochs, header)
    header = False
    model.scheduling_step(val_loss)
    
    if WANDB_ACTIVE:
        epoch_metrics = {x:y[-1] for x,y in model.training_history['val'].items()} | \
                        {x:y[-1] for x,y in model.training_history['trn'].items()} 
        wandb_log_metrics( data = epoch_metrics, step = epoch)
    
    if (epoch + 1) % args.save_every == 0:
        filename = f"{model.name}_{args.runmode}_{args.exp_date}_{args.exp_title}_ep_{epoch+1:03d}"
        save_checkpoint_v2(epochs, model, filename, update_latest=False, update_best=False)    

if WANDB_ACTIVE:
    wandb_run.finish()
    WANDB_ACTIVE = False


logger.info(f" Execution complete ")
# %%
# starting_epoch = 350
# epochs = 35
# starting_epoch = epochs
# epochs += 100
# print(f" run epochs {starting_epoch+1} to {epochs} ")

# filename = f"{model.name}_{ex_runmode}_{ex_date}_{ex_title}_{epochs:03d}.pt"
# print(filename)

# filename = f"{model.name}_{args.runmode}_{ex_date}_{ex_title}_ep_{ex_epoch:03d}"   
# if filename[-3:] != '.pt':
#     filename+='.pt'
# print(filename)

# starting_epoch = last_epoch
# epochs = last_epoch + 100
# starting_epoch = epoch + 1
# print(f" run epochs {starting_epoch+1} to {epochs} ")
