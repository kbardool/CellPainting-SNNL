from typing import Dict, Tuple

# import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"

import numpy as np
import pandas as pd
import torch
import argparse
from types import SimpleNamespace
from datetime import datetime
from pt_datasets import create_dataloader
from typing import Dict, List, Tuple
from collections import defaultdict

 
from snnl.models import Model 
#-------------------------------------------------------------------------------------------------------------------
#  Model DNN class  
#-------------------------------------------------------------------------------------------------------------------

class DNN(Model):
    """
    Feed-forward Neural Network
    
    A feed-forward neural network that optimizes
    softmax cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the softmax cross entropy.
    """

    # _criterion = torch.nn.CrossEntropyLoss()

    def __init__(
        self,
        mode="classifier",
        criterion=torch.nn.CrossEntropyLoss(),
        units: List or Tuple = [(784, 500), (500, 500), (500, 10)],
        activations: List = ["relu", "relu", "softmax"],
        learning_rate: float = 1e-3,
        use_snnl: bool = False,
        loss_factor: float = 1.0,
        snnl_factor: float = 100.0,
        temperature: int = None,
        temperatureLR: float = None,
        adam_weight_decay: float = 0.0,
        SGD_weight_decay: float = 0.0,
        use_annealing: bool = True,
        monitor_grads_layer: int = None,        
        unsupervised: bool = False,
        use_sum: bool = False,
        stability_epsilon: float = 1e-5,
        dropout_p: float = 0.5,
        sample_size: int = 1,        
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose: bool = False,
        use_scheduler: bool = False, 
        scheduler_mode: str = 'min', 
        use_temp_scheduler: bool = False, 
        temp_scheduler_mode: str = 'min'
    ):
        """
        Constructs a feed-forward neural network classifier.

        Parameters
        ----------
        units: list or tuple
            An iterable that consists of the number of units in each hidden layer.
        learning_rate: float
            The learning rate to use for optimization.
        use_snnl: bool
            Whether to use soft nearest neighbor loss or not.
        factor: float
            The balance between SNNL and the primary loss.
            A positive factor implies SNNL minimization,
            while a negative factor implies SNNL maximization.
        temperature: int
            The SNNL temperature.
        use_annealing: bool
            Whether to use annealing temperature or not.
        use_sum: bool
            Use summation of SNNL across hidden layers if True,
            otherwise get the minimum SNNL.
        stability_epsilon: float
            A constant for helping SNNL computation stability
        device: torch.device
            The device to use for model computations.
        """
        self.name = "DNN"
        self.layer_types = []
        self.monitor_grads_layer = monitor_grads_layer
        self.layer_activations = activations
        self.use_scheduler = use_scheduler
        self.scheduler_mode = scheduler_mode
        self.use_temp_scheduler = use_temp_scheduler
        self.temp_scheduler_mode = temp_scheduler_mode
        
        super().__init__(
            mode=mode,
            criterion=criterion.to(device),
            device=device,
            use_snnl=use_snnl,
            loss_factor=loss_factor,
            snnl_factor=snnl_factor,
            temperature=temperature,
            temperatureLR=temperatureLR,
            unsupervised = unsupervised,
            use_annealing=use_annealing,
            use_sum=use_sum,
            # batch_size = batch_size, 
            sample_size = sample_size,
            stability_epsilon=stability_epsilon,
            verbose=verbose,
        )
        print('\n'+'-'*60)        
        print(f" Building DNN from NOTEBOOK")
        print('-'*60)
        # self.primary _ weight = torch.tensor([2])

        self.layers = torch.nn.ModuleList()
        for idx in range(len( units )):
            (layer, in_features, out_features) = units[idx]
            layer = layer.lower()
        
            if layer =='linear':
                self.layers.append( torch.nn.Linear(in_features=in_features, out_features=out_features))
                self.layer_types.append('linear')
            elif layer == 'dropout':
                self.layers.append( torch.nn.Dropout(p=dropout_p))
                self.layer_types.append('dropout')
            elif layer == 'relu':
                self.layers.append(torch.nn.ReLU())
                in_features = units[idx -1][2]
                out_features = units[idx -1][2]
                self.layer_types.append('relu')
            elif layer == 'sigmoid':
                in_features = units[idx -1][2]
                out_features = units[idx -1][2]
                self.layers.append(torch.nn.Sigmoid())
                self.layer_types.append('sigmoid')
            elif layer == 'none':
                in_features = units[idx -1][2]
                out_features = units[idx -1][2]
                pass
            else :
                raise Exception(f"Unrecognized Layer found in layer definitions {layer}")            
            print(f"    layer:  {idx:3d}  type:{layer:15s}  input: {in_features:6d}  output: {out_features:6d}  " )
        
        for index, layer in enumerate(self.layers):
            if index < (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif index == (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            else:
                pass

        parameter_dict = defaultdict(list)
        for name, parm in self.named_parameters():
            if name in ['temperature']:
                parameter_dict['SGD'].append(parm)
            else:
                parameter_dict['Adam'].append(parm)
        # self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.Adam(params=parameter_dict['Adam'], lr=learning_rate, weight_decay = adam_weight_decay)
        if self.use_scheduler:
            self.scheduler = self._ReduceLROnPlateau(self.optimizer, mode=self.scheduler_mode, factor=0.5, patience=25, 
                                                     threshold=0.00001, threshold_mode='rel', cooldown=10, min_lr=0, eps=1e-08, verbose =True)
        else:
            self.scheduler = None
            
        if self.use_snnl:
            self.temp_optimizer = torch.optim.SGD(params=parameter_dict['SGD'], lr=self.temperatureLR, momentum=0.9, weight_decay = SGD_weight_decay)
            if self.use_temp_scheduler:
                self.temp_scheduler = self._ReduceLROnPlateau(self.temp_optimizer, mode=self.temp_scheduler_mode, factor=0.5, patience=25, 
                                                         threshold=0.00001, threshold_mode='rel', cooldown=10, min_lr=0, eps=1e-08, verbose =True)
            else:
                self.temp_scheduler = None
            
        # if not use_snnl:
        #     # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.to(self.device)

        print()
        print(f"    DNN init() -- layer_types        : {self.layer_types}")
        print(f"    DNN init() -- layer_activations  : {self.layer_activations}")     
        print(f"    DNN init() -- mode               : {self.mode}")
        print(f"    DNN init() -- unsupervised       : {self.unsupervised}")
        print(f"    DNN init() -- Primary Crtierion  : {self.primary_criterion}")
        print(f"    DNN init() -- monitor_grads_layer: {self.monitor_grads_layer}")
        print(f"    DNN init() -- use_snnl           : {self.use_snnl}")
        # if self.use_snnl:
        print(f"    DNN init() -- SNNL Crtierion     : {self.snnl_criterion}")
        print(f"    DNN init() -- temperature        : {self.temperature}")
        print(f"    DNN init() -- temperature LR     : {self.temperatureLR}")
        if self.use_scheduler:
            print(f"    DNN init() -- Scheduler          : {self.scheduler}")
            print(f"    DNN init() -- Scheduler mode     : {self.scheduler_mode}")

    def forward_old(self, features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        logits: torch.Tensor
            The model output.
        """
        # features = features.view(features.shape[0], -1)
        activations = {}
        num_layers = len(self.layers)
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = torch.relu(layer(features))
            elif index == num_layers - 1 :     ## last layer
                activations[index] = layer(activations.get(index - 1))
            else:                              ## middle layers
                activations[index] = torch.relu(layer(activations.get(index - 1)))
        
        logits = torch.sigmoid(activations.get(len(activations) - 1))
        
        return activations, logits 



    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        logits: torch.Tensor
            The model output.
        """
        # features = features.view(features.shape[0], -1)
        activations = {}
        # num_layers = len(self.layers)
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:                              ## middle & last layer
                activations[index] = layer(activations[index - 1])              
 
        logits = activations [len(activations) - 1]
        
        # for index, non_linearity in enumerate(self.layer_activations):
        #     non_linearity = non_linearity.lower()
        #     if non_linearity == 'relu':
        #         activations[index] = torch.relu(activations[index])
        #     elif non_linearity == 'sigmoid':
        #         activations[index] = torch.sigmoid(activations[index])
        #     elif non_linearity == 'none':
        #         pass
        #     else :
        #         raise Exception(f"Unrecognized type found in activations {non_linearity}")
 
        return activations, logits 
    
    def fit(
        self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 1
    ) -> None:
        """
        Trains the DNN model.

        Parameters
        ----------
        data_loader: torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs: int
            The number of epochs to train the model.
        show_every: int
            The interval in terms of epoch on displaying training progress.
        """
        header = True
        for epoch in range(epochs):
            train_loss = self.epoch_train( train_loader, epoch, factor = snnl_factor, verbose = False)
            self.model_history('train', train_loss)

            val_loss  = self.epoch_validate( val_loader, epoch, factor = snnl_factor, verbose = False)
            self.model_history('val', val_loss)

            display_epoch_metrics(self, epoch, epochs, header)
            header = False            
                

    def predict(
        self, features: torch.Tensor, return_likelihoods: bool = False
    ) -> torch.Tensor:
        """
        Returns model classifications

        Parameters
        ----------
        features: torch.Tensor
            The input features to classify.
        return_likelihoods: bool
            Whether to return classes with likelihoods or not.

        Returns
        -------
        predictions: torch.Tensor
            The class likelihood output by the model.
        classes: torch.Tensor
            The class prediction by the model.
        """
        outputs, logits = self.forward(features)
        return outputs, logits
        # predictions, classes = torch.max(outputs.data, dim=1)
        # return (predictions, classes) if return_likelihoods else classes



