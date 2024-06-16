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

import wandb
from KevinsRoutines.utils.utils_wandb    import  init_wandb, wandb_log_metrics,wandb_watch
from KevinsRoutines.utils.utils_general  import  list_namespace 

from snnl.utils import parse_args, load_configuration, set_global_seed, get_device, set_device
from snnl.utils import display_epoch_metrics
from snnl.utils import CellpaintingDataset, InfiniteDataLoader, custom_collate_fn
from snnl.utils import display_model_summary, define_autoencoder_model
from snnl.utils import save_checkpoint_v2, load_checkpoint_v2

timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
logger = logging.getLogger(__name__) 
logLevel = os.environ.get('LOG_LEVEL', 'INFO').upper()
FORMAT = '%(asctime)s - %(name)s - %(levelname)s: - %(message)s'
logging.basicConfig(level="INFO", format= FORMAT)

logger.info(f" Excution started : {timestamp} ")
logger.info(f" Pytorch version  : {torch.__version__}")
logger.debug(f" Search path      : {sys.path}")


#
# Parse input arguments 
#

cli_args = parse_args()
cli_args

args = load_configuration(cli_args)
set_global_seed(args.random_seed)
# args.exp_title
# args.ckpt

current_device = get_device()
if args.gpu_id is not None:
    current_device = set_device(args.gpu_id)
    _ = get_device()

if args.ckpt is not None:
    if os.path.exists(os.path.join('ckpts', args.ckpt)):
        # logging.info(f" Checkpoint {args.ckpt} found")
        logger.info(f" Resuming training using checkpoint: {args.ckpt}")
    else:
        logger.error(f" *** Checkpoint {args.ckpt} not found *** \n")
        raise ValueError(f"\n *** Checkpoint DOESNT EXIST *** \n")
        
#
#  Define dataset and dataloaders
#    Load CellPainting Dataset
#

logging.info(f" load {args.dataset}")
train_dataset = CellpaintingDataset(type='train', **args.cellpainting_args)
train_loader = InfiniteDataLoader(dataset=train_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 0, collate_fn = custom_collate_fn)
val_dataset = CellpaintingDataset(type='val', **args.cellpainting_args)
val_loader = InfiniteDataLoader(dataset=val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 0, collate_fn = custom_collate_fn)


#
# WandB initialization 
#
WANDB_ACTIVE = args.wandb

if WANDB_ACTIVE:
    if args.exp_id is not None:
        logger.info(" Resume WandB Run")
        resume_wandb = True
    else:
        logger.info(" Initialize WandB Run")
        resume_wandb = False
        args.exp_name = 'AE_'+datetime.now().strftime('%m%d_%H%M')
    
    wandb_run = init_wandb(args)
    
    args.exp_id = wandb_run.id
    args.exp_date = '2024'+args.exp_name[3:7]
    
    logger.info(f" WandB tracking started ")
    logger.info(f" Experiment Run id: {args.exp_id}   / {wandb_run.id}")
    logger.info(f" Experiment Name  : {args.exp_name} / {wandb_run.name} ")
    logger.info(f" Experiment Date  : {args.exp_date}  ")
    logger.info(f" Experiment Notes : {args.exp_description} / {wandb_run.notes} ")
else: 
    logger.info(f" *** W&&B Logging is INACTIVE *** ")
    args.exp_name = 'AE_'+datetime.now().strftime('%m%d_%H%M')
    args.exp_date = datetime.now().strftime('%Y%m%d')    


# %%
#  Define Model
# 
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

model = define_autoencoder_model(args, embedding_layer = args.embedding_layer, device = current_device)
list_namespace(args)

if WANDB_ACTIVE:
    wandb_watch(item = model, criterion=None, log = 'all', log_freq = 1000, log_graph = True)

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
    logger.info(f" RESUME TRAINING - Run epochs {starting_epoch+1} to {epochs} ")
    
else:
    resume_training = False
    starting_epoch = 0
    epochs = args.epochs
    logger.info(f" INITIALIZE TRAINING - Run epochs {starting_epoch+1} to {epochs} ")

print(f" Current device       : {current_device}")
print(f" Model device         : {model.device}")
print(f" Model embedding_layer: {model.embedding_layer}")
print(f" SNNL temperature     : {model.temperature}")
print(f" Learning rate        : {model.optimizer.param_groups[0]['lr']}") 
print(f" snnl_factor          : {model.snnl_factor}")
print(f" loss_factor          : {model.loss_factor}")
print(f" monitor_grads_layer  : {model.monitor_grads_layer}")
print(f" Use Scheduler        : {model.use_scheduler}") 
print(f" Use snnl             : {model.use_snnl}") 
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

if 'gen' not in model.training_history:
    print(f" Define self.training_history['gen'] ")
    model.training_history['gen'] = {'trn_best_metric' : 0, 'trn_best_metric_ep' : 0, 'trn_best_loss': 0, 'trn_best_loss_ep' : 0 ,
                                    'val_best_metric' : 0, 'val_best_metric_ep' : 0, 'val_best_loss': 0, 'val_best_loss_ep' : 0 }        

    for key in ['trn', 'val']:
        tmp = np.argmin(model.training_history[key][f'{key}_ttl_loss'])
        model.training_history['gen'][f'{key}_best_loss_ep'] = tmp
        model.training_history['gen'][f'{key}_best_loss']    = model.training_history[key][f'{key}_ttl_loss'][tmp]
        
        tmp1 = np.argmax(model.training_history[key][f'{key}_R2_score'])
        model.training_history['gen'][f'{key}_best_metric_ep'] = tmp1
        model.training_history['gen'][f'{key}_best_metric'] = model.training_history[key][f'{key}_R2_score'][tmp1]
 
logger.info(f" Best training loss     : {model.training_history['gen']['trn_best_loss']:6f} - epoch: {model.training_history['gen']['trn_best_loss_ep']}") 
logger.info(f" Best training metric   : {model.training_history['gen']['trn_best_metric']:6f} - epoch: {model.training_history['gen']['trn_best_metric_ep']}") 
logger.info(f" ") 
logger.info(f" Best validation loss   : {model.training_history['gen']['val_best_loss']:6f} - epoch: {model.training_history['gen']['val_best_loss_ep']}") 
logger.info(f" Best validation metric : {model.training_history['gen']['val_best_metric']:6f} - epoch: {model.training_history['gen']['val_best_metric_ep']}") 
logger.info(f" ")

if WANDB_ACTIVE:
    wandb.config.update(args,allow_val_change=True )

#
#  Running Training Loop
#

logger.info(f" Experiment run id:  {args.exp_id}")
logger.info(f" Experiment Name  :  {args.exp_name} ")
logger.info(f" Experiment Date  :  {args.exp_date} ")
logger.info(f" Experiment Title :  {args.exp_title} ")
logger.info(f" Experiment Notes :  {args.exp_description}")
logger.info(f" Run epochs {starting_epoch+1:4d} to {epochs:4d}")

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

#
# Finish WandB logging
#
if WANDB_ACTIVE:
    wandb_run.finish()
    WANDB_ACTIVE = False


logger.info(f" Execution complete ")
 
