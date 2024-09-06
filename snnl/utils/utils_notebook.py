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


# from torchinfo import summary
import numpy as np
import pandas as pd
import seaborn as sb
import torch
import wandb
from matplotlib import pyplot as plt
import scipy.stats as sps 
import sklearn.metrics as skm 
from scipy.spatial.distance import pdist, squareform, euclidean
from snnl.models import Autoencoder
from .dataloader import CellpaintingDataset, InfiniteDataLoader, custom_collate_fn
from KevinsRoutines.utils.utils_wandb  import  init_wandb, wandb_log_metrics,wandb_watch
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------------------------
#  Define Model
# -------------------------------------------------------------------------------------------------------------------
def define_autoencoder_model(args,  device = None, verbose = False):
    
    assert args.embedding_layer != 0, "embedding_layer cannot be zero"
    print(f" EMBEDDING LAYER: {args.embedding_layer}")

    if device is None: 
        if args.current_device is not None:
            device = args.current_device
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f" Device {device} will be used")

    if args.runmode.lower() == "baseline":
        logger.info(f"Defining model in baseline mode")
        model = Autoencoder(
            mode = "autoencoding",
            device = device,

            units           = args.units,
            code_units      = args.code_units, 
            embedding_layer = args.embedding_layer,
            input_shape     = args.input_shape, 
            sample_size     = args.cellpainting_args['sample_size'],
            use_single_loss = args.use_single_loss,

            criterion       = torch.nn.MSELoss(reduction='mean'),
            loss_factor     = args.loss_factor,

            learning_rate      = args.learning_rate,
            use_prim_optimzier = args.use_prim_optimizer,
            use_prim_scheduler = args.use_prim_scheduler,
            adam_weight_decay  = args.adam_weight_decay,
            use_snnl = False,
            verbose = verbose )

    elif args.runmode.lower() == "snnl":
        logger.info(f"Defining model in SNNL mode ")
        model = Autoencoder(
            mode="latent_code",
            device = device,

            units           = args.units,
            code_units      = args.code_units,
            embedding_layer = args.embedding_layer,
            input_shape     = args.input_shape,
            sample_size     = args.cellpainting_args['sample_size'],
            use_single_loss = args.use_single_loss,

            criterion       = torch.nn.MSELoss(reduction='mean'),
            loss_factor     = args.loss_factor,

            learning_rate      = args.learning_rate,
            use_prim_optimzier = args.use_prim_optimizer,
            use_prim_scheduler = args.use_prim_scheduler,
            adam_weight_decay  = args.adam_weight_decay,

            use_snnl=True,
            snnl_factor = args.snnl_factor,
            temperature = args.temperature,
            use_temp_optimzier = args.use_temp_optimizer,
            use_temp_scheduler = args.use_temp_scheduler,
            temperatureLR = args.temperatureLR,
            SGD_weight_decay = args.SGD_weight_decay,
            SGD_momentum  = args.SGD_momentum,
            use_annealing = args.use_annealing,
            anneal_patience = args.anneal_patience,
            use_sum = args.use_sum,

            verbose = verbose )
    else:
        raise ValueError("Choose runmode between [baseline] and [snnl] only.")

    return model


def build_dataloaders(args, data = None):
    """
    build a dictionary of dataloaders

    data : keys to the dataset settings (and resulting keys in output dictionary)
    """
    dataset = dict()
    data_loader = dict()

    #### Load CellPainting Dataset
    logging.info(f" load {args.dataset}")
    print(f" load {args.dataset}")
    for datatype in data:
        dataset[datatype] = CellpaintingDataset(type = datatype, **args.cellpainting_args)
        data_loader[datatype] = InfiniteDataLoader(dataset = dataset[datatype], batch_size= args.batch_size,shuffle = False, num_workers = 0, collate_fn = custom_collate_fn)    
    return data_loader


def setup_wandb(args, verbose = False):
    if verbose:
        logger.info(f"WANDB_ACTIVE parameter                : {args.WANDB_ACTIVE}")
        logger.info(f"Project Name     (wandb_run.project)  : {args.project_name}")
        logger.info(f"Experiment Id    (wandb_run.id)       : {args.exp_id}")
        logger.info(f"Experiment Title (wandb_run.title)    : {args.exp_title}")
        logger.info(f"Experiment Notes (wandb_run.notes)    : {args.exp_description}")
        logger.info(f"Initial Exp Name (wandb_run.name)     : {args.exp_name}")
        logger.info(f"Initial Exp Date (extract from name)  : {args.exp_date}")

    if args.WANDB_ACTIVE:
        wandb_status = "***** Initialize NEW  W&B Run *****" if args.exp_id is None else "***** Resume EXISTING W&B Run *****" 
        logger.info(f"{wandb_status}")

        wandb_run = init_wandb(args)
        args.exp_id = wandb_run.id
        args.exp_date = args.exp_name[3:]
        logger.info(f" Experiment Name  : {args.exp_name}")
        logger.info(f" Experiment Date  : {args.exp_date}")
    else: 
        wandb_status = "***** W&B Logging INACTIVE *****"
        wandb_run = None

    logger.info(f"{wandb_status}")
    logger.info(f"WANDB_ACTIVE     : {args.WANDB_ACTIVE}")
    logger.info(f"Project Name     : {args.project_name}")
    logger.info(f"Experiment Id    : {args.exp_id}")
    logger.info(f"Experiment Name  : {args.exp_name}")
    logger.info(f"Experiment Date  : {args.exp_date}")
    logger.info(f"Experiment Title : {args.exp_title}")
    logger.info(f"Experiment Notes : {args.exp_description}")
    return wandb_run


def init_resume_training(model, args, verbose = False):

    if args.ckpt is not None:
        model.resume_training = True
        model, last_epoch = args.load_checkpoint(model, args.ckpt, verbose = verbose)
        model.train()
        model.device = args.current_device
        model = model.to(args.current_device)
        # model = model.cuda(device=current_device)
        logging.info(f" Loaded Model device {model.device} -  Last completed epoch : {last_epoch}")
        model.starting_epoch = last_epoch
        model.ending_epoch = last_epoch + args.epochs
        logging.info(f" RESUME TRAINING - Run {args.epochs} epochs: epoch {model.starting_epoch+1} to {model.ending_epoch} ")

        if 'gen' not in model.training_history:
            print('*' * 60)
            print(f" Define self.training_history['gen'] ")
            print('*' * 60)
            model.training_history['gen'] = {'trn_best_metric' : 0, 'trn_best_metric_ep' : 0, 'trn_best_loss': np.inf, 'trn_best_loss_ep' : 0,
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

    model.best_metric = model.training_history['gen'][f'val_best_metric']  
    model.best_epoch  = model.training_history['gen'][f'val_best_metric_ep']  
    return model
