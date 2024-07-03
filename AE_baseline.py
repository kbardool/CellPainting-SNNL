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
import wandb

torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=150, profile=None, sci_mode=None)
pp = pprint.PrettyPrinter(indent=4)
pd.options.display.width = 132
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=150, profile=None, sci_mode=None)
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan')

from KevinsRoutines.utils.utils_wandb    import  init_wandb, wandb_log_metrics,wandb_watch
from KevinsRoutines.utils.utils_general  import  list_namespace 
from snnl.utils import parse_args, load_configuration, set_global_seed, get_device, set_device
from snnl.utils import display_epoch_metrics, display_model_parameters
from snnl.utils import CellpaintingDataset, InfiniteDataLoader, custom_collate_fn
from snnl.utils import display_model_summary, define_autoencoder_model
from snnl.utils import save_checkpoint_v3, load_checkpoint_v2

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
EXP_DATE = datetime.now().strftime('%Y%m%d_%H%M')

if WANDB_ACTIVE:
    if args.exp_id is not None:
        logger.info(" Resume WandB Run")
        resume_wandb = True
    else:
        logger.info(" Initialize WandB Run")
        resume_wandb = False
        args.exp_name = f"AE_{EXP_DATE}"
    
    wandb_run = init_wandb(args)
    args.exp_date = args.exp_name[3:]
    
    logger.info(f" WandB tracking started ")
    logger.info(f" Project  :  {args.project_name}")    
    logger.info(f" Run id   :  {args.exp_id}   / {wandb_run.id}")
    logger.info(f" Name     :  {args.exp_name} / {wandb_run.name} ")
    logger.info(f" Title    :  {args.exp_title}")    
    logger.info(f" Date     :  {args.exp_date}  ")
    logger.info(f" Notes    :  {args.exp_description} / {wandb_run.notes} ")
else: 
    logger.info(f" *** W&&B Logging is INACTIVE *** ")
    args.exp_name = f"AE_{EXP_DATE}"
    args.exp_date = EXP_DATE

#
#  Define Model
# 
model = define_autoencoder_model(args, device = current_device)
list_namespace(args)

if WANDB_ACTIVE:
    wandb_watch(item = model, criterion=None, log = 'all', log_freq = 1000, log_graph = True)

#
# Load model checkpoint if provided 
#
if args.ckpt is not None:
    model.resume_training = True
    model , last_epoch = load_checkpoint_v2(model, args.ckpt, verbose = True )  
    model.train()
    model.device = current_device
    model = model.cuda(device=current_device)
    logging.info(f" Loaded Model device {model.device} -  Last completed epoch : {last_epoch}")
    model.starting_epoch = last_epoch
    model.ending_epoch = last_epoch + args.epochs
    logging.info(f" RESUME TRAINING - Run {args.epochs} epochs: epoch {model.starting_epoch+1} to {model.ending_epoch} ")

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
    
else:
    model.resume_training = False
    model.starting_epoch = 0
    model.ending_epoch = args.epochs
    logging.info(f" INITIALIZE TRAINING - Run {args.epochs} epochs: epoch {model.starting_epoch+1} to {model.ending_epoch} ")

display_model_parameters(model)

logger.info(f" Experiment run id      : {args.exp_id}")
logger.info(f" Experiment Name        : {args.exp_name} ")
logger.info(f" Experiment Date        : {args.exp_date} ")
logger.info(f" Experiment Title       : {args.exp_title} ")
logger.info(f" Experiment Notes       : {args.exp_description}")
logger.info(f" Run epochs             : {model.starting_epoch+1:4d} to {model.ending_epoch:4d}")

if WANDB_ACTIVE:
    wandb.config.update(args,allow_val_change=True )
    
#
#  Running Training Loop
#
header = True

for epoch in range(model.starting_epoch, model.ending_epoch):
    train_loss = model.epoch_train(train_loader, epoch)
    val_loss = model.epoch_validate(val_loader, epoch)
    
    display_epoch_metrics(model, epoch, model.ending_epoch, header)
    header = False
    model.scheduling_step(val_loss)
    
    if WANDB_ACTIVE:
        epoch_metrics = {x:y     for x,y in model.training_history['gen'].items()} | \
                        {x:y[-1] for x,y in model.training_history['val'].items()} | \
                        {x:y[-1] for x,y in model.training_history['trn'].items()} 
        wandb_log_metrics( data = epoch_metrics, step = epoch)
    
    if model.new_best:
        save_checkpoint_v3(epoch+1, model, args, update_best=True)        
    if (epoch + 1) % args.save_every == 0:
        save_checkpoint_v3(epoch+1, model, args)    
        
#        
# Write final checkpoint 
#
logger.info(f" Final checkpoint epoch {epoch+1}")
save_checkpoint_v3(epoch+1, model, args, update_latest=True)            

#
# Finish WandB logging
#
if WANDB_ACTIVE:
    wandb_run.finish()
    WANDB_ACTIVE = False


logger.info(f" Execution complete ")
 
