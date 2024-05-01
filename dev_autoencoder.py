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
from sklearn.metrics import accuracy_score, f1_score, roc_curve, precision_recall_curve, precision_recall_fscore_support, \
                            roc_auc_score
from dev_base import Model
#-------------------------------------------------------------------------------------------------------------------
#  Autoencoder class  
#-------------------------------------------------------------------------------------------------------------------

class Autoencoder(Model):
    """
    A feed-forward autoencoder neural network that optimizes
    binary cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the binary cross entropy.
    """

    _supported_modes = ["autoencoding", "latent_code"]
    _criterion = torch.nn.BCELoss()

    def __init__(
        self,
        criterion = torch.nn.BCELoss(),
        mode: str = "autoencoding",
        units: List = [] ,
        code_units: int = 50,
        activations: List = [],
        embedding_layer: int = 0,
        learning_rate: float = 1e-3,
        use_snnl: bool = False,
        loss_factor: float = 1.0,
        snnl_factor: float = 100.0,
        temperature: int = None,
        temperatureLR: int = None,
        adam_weight_decay: float = 0.0,
        SGD_weight_decay: float = 0.0,        
        use_annealing: bool = True,
        monitor_grads_layer: int = None,
        use_sum: bool = False,
        stability_epsilon: float = 1e-5,
        input_shape: int = 0 ,
        sample_size: int = None,  
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose: bool = False,
    ):
        """
        Constructs the autoencoder model with the following units,
        <input_shape>-500-500-2000-<code_dim>-2000-500-500-<input_shape>

        Parameters
        ----------
        input_shape: int
            The dimensionality of the input features.
        code_dim: int
            The dimensionality of the latent code.
        learning_rate: float
            The learning rate to use for optimization.
        use_snnl: bool
            Whether to use soft nearest neighbor loss or not.
        factor: float
            The balance factor between SNNL and the primary loss.
            A positive factor implies SNNL minimization, while a negative
            factor implies SNNL maximization.
        temperature: int
            The SNNL temperature.
        use_annealing: bool
            Whether to use annealing temperature or not.
        use_sum: bool
            Use summation of SNNL across hidden layers if True,
            otherwise get the minimum SNNL.
        mode: str
            The mode in which the soft nearest neighbor loss
            will be used.
        code_units: int
            The number of units in which the SNNL will be applied.
        stability_epsilon: float
            A constant for helping SNNL computation stability.
        device: torch.device
            The device to use for the model computations.
        """
        if mode not in Autoencoder._supported_modes:
            raise ValueError(f"Mode {mode} is not supported.")
        assert sample_size is not None, f"sample_size must be specified in Autoencoder initialization"
        
        super().__init__(
            mode=mode,
            criterion=criterion,
            device=device,
            use_snnl=use_snnl,
            loss_factor=loss_factor,
            snnl_factor=snnl_factor,
            temperature=temperature,
            temperatureLR=temperatureLR,
            code_units=code_units,
            embedding_layer=embedding_layer,
            use_annealing=use_annealing,
            use_sum=use_sum,
            sample_size = sample_size,
            stability_epsilon=stability_epsilon,
            verbose=verbose,
        )
        
        self.monitor_grads_layer = monitor_grads_layer
        self.layer_activations = activations
        
        print('\n'+'-'*60)
        print(f" Building Autoencoder from NOTEBOOK")
        print('-'*60)
           
        self.layers = torch.nn.ModuleList()
        self.layer_types = []
        for idx, ((layer, in_features, out_features), non_linearity) in enumerate(zip( units,activations)):
            type = layer.lower()
            non_linearity = non_linearity.lower()
            print(f"    layer/non_linearity pair:  {idx:3d}  type:{layer:15s}  input: {in_features:6d}  output: {out_features:6d}  non_linearity: {non_linearity:15s}" )
        
            if layer =='linear':
                self.layers.append( torch.nn.Linear(in_features=in_features, out_features=out_features))
                self.layer_types.append('linear')
            elif layer == 'dropout':
                self.layers.append( torch.nn.dropout(p=dropout_p))
                self.layer_types.append('dropout')
            else :
                raise Exception(f"Unrecognized Layer found in layer definitions {type}")
                
            if non_linearity == 'relu':
                self.layers.append(torch.nn.ReLU())
                self.layer_types.append('relu')
            elif non_linearity == 'sigmoid':
                self.layers.append(torch.nn.Sigmoid())
                self.layer_types.append('sigmoid')
            elif non_linearity == 'none':
                pass
            else :
                raise Exception(f"Unrecognized type found in activations {non_linearity}")
        print(f"    layer_types      : {self.layer_types}")
        print(f"    layer_activations: {self.layer_activations}")             
        
        # self.layers = torch.nn.ModuleList(
        #     [
        #         torch.nn.Linear(in_features=input_shape, out_features=500),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Linear(in_features=500, out_features=500),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Linear(in_features=500, out_features=2000),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Linear(in_features=2000, out_features=code_dim),
        #         torch.nn.Sigmoid(),
        #         torch.nn.Linear(in_features=code_dim, out_features=2000),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Linear(in_features=2000, out_features=500),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Linear(in_features=500, out_features=500),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Linear(in_features=500, out_features=input_shape),
        #         torch.nn.ReLU(inplace=True),
        #         # torch.nn.Sigmoid(),
        #     ]
        # )
 
        for index, layer in enumerate(self.layers):
            if (index == 6 or index == 14) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                pass

        self.name = "Autoencoder"

        parameter_dict = defaultdict(list)
        for name, parm in self.named_parameters():
            print(f"    parameter: {name}  {parm.shape}")
            if name in ['temperature']:
                parameter_dict['SGD'].append(parm)
            else:
                parameter_dict['Adam'].append(parm)
        self.optimizer = torch.optim.Adam(params=parameter_dict['Adam'], lr=learning_rate, weight_decay = adam_weight_decay)
        self.temp_optimizer = torch.optim.SGD(params=parameter_dict['SGD'], lr=self.temperatureLR, momentum=0.9, weight_decay = SGD_weight_decay)        
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # if not use_snnl:
        #     self.criterion = Autoencoder._criterion.to(device)
        
        self.to(self.device) 
        print(f"    AE init() -- mode               : {self.mode}")
        print(f"    AE init() -- unsupervised       : {self.unsupervised}")
        print(f"    AE init() -- use_snnl           : {self.use_snnl}")
        print(f"    AE init() -- Primary Crtierion  : {self.primary_criterion}")
        print(f"    AE init() -- SNNL Crtierion     : {self.snnl_criterion}")
        print(f"    AE init() -- temperature        : {self.temperature}")
        print(f"    AE init() -- temperature LR     : {self.temperatureLR}")
        print(f"    AE init() -- monitor_grads_layer: {self.monitor_grads_layer}")

    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        reconstruction: torch.Tensor
            The model output.
        """
        activations = {}
        if features.ndim > 2:
            features = features.view(features.shape[0], -1)
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations.get(index - 1))
        reconstruction = activations.get(len(activations) - 1)
        return activations, reconstruction

    def compute_latent_code(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes the latent code representation for the features
        using a trained autoencoder network.

        Parameters
        ----------
        features: torch.Tensor
            The features to represent in latent space.

        Returns
        -------
        latent_code: np.ndarray
            The latent code representation for the features.
        """
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features)
        activations = {}
        for index, layer in enumerate(self.layers[:8]):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations.get(index - 1))
        latent_code = activations.get(len(activations) - 1)
        latent_code = latent_code.detach().numpy()
        return latent_code

    
    def fit(
        self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 1
    ) -> None:
        """
        Trains the autoencoder model.

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
        for epoch in range(starting_epoch,epochs):
            train_loss = model.epoch_train(train_loader, epoch)
            model.model_history('train', train_loss)
        
            validation_loss = model.epoch_validate(val_loader, epoch)
            model.model_history('val', validation_loss)
            
            display_epoch_metrics(model, epoch, epochs, header)
            header = False    
         

    def fit_orig(
        self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 1
    ) -> None:
        """
        Trains the autoencoder model.

        Parameters
        ----------
        data_loader: torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs: int
            The number of epochs to train the model.
        show_every: int
            The interval in terms of epoch on displaying training progress.
        """
        if self.use_snnl:
            self.train_snn_loss = []
            self.train_recon_loss = []

        for epoch in range(epochs):
            epoch_loss = self.epoch_train(data_loader, epoch)
            if type(epoch_loss) is tuple:
                self.model_history('train', epoch_loss)
                # self.train_loss.append(epoch_loss[0])
                # self.train_snn_loss.append(epoch_loss[1])
                # self.train_recon_loss.append(epoch_loss[2])
                # if (epoch + 1) % show_every == 0:
                    # print(
                        # f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                    # )
                    # print(
                        # f"\trecon loss = {self.train_recon_loss[-1]:.6f}\t|\tsnn loss = {self.train_snn_loss[-1]:.6f}"
                    # )
            else:
                self.train_loss.append(epoch_loss)
                if (epoch + 1) % show_every == 0:
                    print(
                        f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                    )
 


