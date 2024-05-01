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
from snnl.utils import binary_accuracy, binary_f1_score

 
from dev_base import Model 
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
        unsupervised: bool = None,
        use_sum: bool = False,
        stability_epsilon: float = 1e-3,

        sample_size: int = 1,        
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose: bool = False,
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
        print(f" Building DNN from NOTEBOOK")
        self.primary_weight = torch.tensor([2])

        self.layers = torch.nn.ModuleList()
        self.layer_type = []
        for idx, (type,in_features, out_features) in enumerate(units):
            type = type.lower()
            if type =='linear':
                self.layers.append( torch.nn.Linear(in_features=in_features, out_features=out_features))
                self.layer_type.append('linear')
            elif type == 'dropout':
                self.layers.append( torch.nn.dropout(p=dropout_p))
                self.layer_type.append('dropout')
        self.layer_activations = activations
        self.monitor_grads_layer = -1
        
        print(f" layer_types      : {self.layer_type}")
        print(f" layer_activations: {self.layer_activations}")
        # self.layers = torch.nn.ModuleList(
        #     [
        #         torch.nn.Linear(in_features=in_features, out_features=out_features)
        #         for in_features, out_features in units
        #     ]
        # )
        
        for index, layer in enumerate(self.layers):
            if index < (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif index == (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            else:
                pass

        self.name = "DNN"
 
        parameter_dict = defaultdict(list)
        for name, parm in self.named_parameters():
            if name in ['temperature']:
                parameter_dict['SGD'].append(parm)
            else:
                parameter_dict['Adam'].append(parm)
        self.optimizer = torch.optim.Adam(params=parameter_dict['Adam'], lr=learning_rate, weight_decay = adam_weight_decay)
        self.temp_optimizer = torch.optim.SGD(params=parameter_dict['SGD'], lr=self.temperatureLR, momentum=0.9, weight_decay = SGD_weight_decay)
        
        # self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        
        if not use_snnl:
            # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
            self.criterion = criterion

        self.to(self.device)
        
        print(f"    DNN _init()_    -- mode:              {self.mode}")
        print(f"    DNN _init()_    -- unsupervised :     {self.unsupervised}")
        print(f"    DNN _init()_    -- use_snnl :         {self.use_snnl}")
        if not use_snnl:
            print(f"    DNN _init()_    -- Crtierion  :       {self.criterion}")
        else:
            print(f"    DNN _init()_    -- Crtierion  :       {self.snnl_criterion}")
        print(f"    DNN _init()_    -- temperature :      {temperature}")
        print(f"    DNN _init()_    -- temperature LR:    {temperatureLR}")
 

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
        num_layers = len(self.layers)
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:                              ## middle & last layer
                activations[index] = layer(activations[index - 1])
                
        assert (len(self.layers) == len(activations)) , f" lengths of self.layers {len(self.layers)} and activations {len(activations)} do not match"
        
        for index, non_linearity in enumerate(self.layer_activations):
            non_linearity = non_linearity.lower()
            if non_linearity == 'relu':
                activations[index] = torch.relu(activations[index])
            elif non_linearity == 'sigmoid':
                activations[index] = torch.sigmoid(activations[index])
            elif non_linearity == 'none':
                pass
            else :
                raise Exception(f"Unrecognized type found in activations {non_linearity}")
                
        logits = activations [len(activations) - 1]
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
            if self.use_snnl:
                train_loss = self.epoch_train( train_loader, epoch, factor = snnl_factor, verbose = False)
                self.model_history('train', train_loss)

                val_loss  = self.epoch_validate( val_loader, epoch, factor = snnl_factor, verbose = False)
                self.model_history('val', val_loss)

                display_epoch_metrics(self, epoch, epochs, header)
                header = False            
                
            else:
                epoch_loss, epoch_accuracy = self.epoch_train(data_loader)
                self.train_loss.append(epoch_loss)
                self.train_accuracy.append(epoch_accuracy)
                if (epoch + 1) % show_every == 0:
                    print(f"epoch {epoch + 1}/{epochs}")
                    print(
                        f"\tmean loss = {self.train_loss[-1]:.6f}\t|\tmean acc = {self.train_accuracy[-1]:.6f}"
                    )

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



