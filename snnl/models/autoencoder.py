from typing import Dict, Tuple

# import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from snnl.models import Model
# from snnl.utils  import display_epoch_metrics
logger = logging.getLogger(__name__) 

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
        mode: str = "autoencoding",
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        
        units: List = [] ,
        code_units: int = 50,
        embedding_layer: int = 0,
        input_shape: int = 0 ,
        sample_size: int = None,  
        dropout_p: float = 0.5,            

        criterion = None,
        use_single_loss: bool = False, 
        loss_factor: float = 1.0,
        use_prim_optimzier: bool = False,
        use_prim_scheduler: bool = False, 
        learning_rate: float = 1e-3,
        adam_weight_decay: float = 0.0,
        
        use_snnl: bool = False,
        snnl_factor: float = 0.0,
        temperature: int = 0.0,

        use_temp_optimzier: bool = False,
        use_temp_scheduler: bool = False, 
        temperatureLR: int = 0.0,
        SGD_weight_decay: float = 0.0,        
        SGD_momentum: float = 0.0,        
        use_annealing: bool = False,
        use_sum: bool = False,
        
        monitor_grads_layer: int = None,
        stability_epsilon: float = 1e-5,
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
        snnl_factor: float
            The balance factor between SNNL and the primary loss.
            A positive factor implies SNNL minimization, while a negative
            factor implies SNNL maximization.
        temperature: int
            The SNNL temperature.
        temp_leanring: bool
            Learn the SNNL temperature through backpropigation
            Uses static temperature when False 
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
        if use_temp_scheduler:
            assert use_snnl, f" temp_scheduler = True,  but use_snnl = False"
        # assert not(self.use_annealing and self.use_temp_optimizer)," Temperature annealing and Temp optimization are mutually exclusive"
        
        
        self.name = "AE"
        self.layer_types = []
        self.non_linearities = []
        # self.layer_activations = activations                
        self.monitor_grads_layer = monitor_grads_layer
        
        self.use_single_loss= use_single_loss
        self.optimizers = {}
        self.schedulers = {} 

        self.optimizer = None
        self.scheduler = None 
        self.learning_rate = learning_rate       
        self.use_prim_optimizer = use_prim_optimzier
        self.use_prim_scheduler = use_prim_scheduler
        self.use_annealing = use_annealing
     
        self.temp_optimizer =  None
        self.temp_scheduler = None 
        self.temperatureLR = temperatureLR
        self.use_temp_optimizer = use_temp_optimzier
        self.use_temp_scheduler = use_temp_scheduler
        self.adam_weight_decay = adam_weight_decay
        self.SGD_weight_decay = SGD_weight_decay
        self.SGD_momentum = SGD_momentum
        
        super().__init__(
            mode=mode,
            criterion=criterion,
            device=device,
            use_snnl=use_snnl,
            loss_factor=loss_factor,
            snnl_factor=snnl_factor,
            temperature = temperature,
            temperatureLR = temperatureLR,
            code_units=code_units,
            embedding_layer=embedding_layer,
            use_annealing=use_annealing,
            use_sum=use_sum,
            sample_size = sample_size,
            stability_epsilon=stability_epsilon,
            verbose=verbose,
        )
        if verbose:
            print('\n'+'-'*60)
            print(f" Building Autoencoder from NOTEBOOK")
            print('-'*60)
           
        self.layers = torch.nn.ModuleList() 
        for idx, ((layer, in_features, out_features) ) in enumerate( units ):
            type = layer.lower()
            logger.info(f"    layer pair:  {idx:3d}  type:{layer:15s}  input: {in_features:6d}  output: {out_features:6d}  " 
                  f"  weights: [{out_features}, {in_features}]   ")
        
            if layer =='linear':
                self.layers.append( torch.nn.Linear(in_features=in_features, out_features=out_features))
                self.layer_types.append('linear')
            elif layer == 'dropout':
                self.layers.append( torch.nn.Dropout(p=dropout_p))
                self.layer_types.append('dropout')
            elif layer == 'relu':
                self.layers.append(torch.nn.ReLU())
                self.layer_types.append('relu')
                self.non_linearities.append('relu')
            elif layer == 'sigmoid':
                self.layers.append(torch.nn.Sigmoid())
                self.non_linearities.append('sigmoid')
            elif layer == 'none':
                pass
            else :
                raise Exception(f"Unrecognized Layer found in layer definitions {type}")        
 
        for index, layer in enumerate(self.layers):
            if (index == 6 or index == 14) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                pass
 
        self.temp_params = [p for name, p in self.named_parameters() if 'temperature' in name]
        self.network_params = [p for name, p in self.named_parameters() if 'temperature' not in name]
    
        if self.use_prim_optimizer:
            self.setup_prim_optimizer()
        if self.use_temp_optimizer:
            self.setup_temp_optimizer()

        self.to(self.device) 
        if verbose:         
            print(f"    AE init() -- mode               : {self.mode}")
            print(f"    AE init() -- unsupervised       : {self.unsupervised}")
            print(f"    AE init() -- layer_types        : {self.layer_types}")
            print(f"    AE init() -- non linearities    : {self.non_linearities}")               
            print(f"    AE init() -- Primary Crtierion  : {self.primary_criterion}")
            print(f"    AE init() -- monitor_grads_layer: {self.monitor_grads_layer}")
            print(f"    AE init() -- Primary optimizer  : {self.use_prim_optimizer}")
            print(f"    AE init() -- Primary scheduler  : {self.use_prim_scheduler}")
            print(f"    AE init() -- use_snnl           : {self.use_snnl}")
            print(f"    AE init() -- SNNL Crtierion     : {self.snnl_criterion}")
            print(f"    AE init() -- temperature        : {self.temperature}")
            print(f"    AE init() -- temperature LR     : {self.temperatureLR}")
            print(f"    AE init() -- Temperature optmzr : {self.use_temp_scheduler}")
            print(f"    AE init() -- Temperature schdlr : {self.use_temp_scheduler}")

    def setup_prim_optimizer(self):
        
        params= self.network_params + self.temp_params if self.use_single_loss else self.network_params
 
        self.optimizers['prim'] = torch.optim.Adam(params = params, lr=self.learning_rate, weight_decay = self.adam_weight_decay)
        
        if self.use_prim_scheduler:
            self.schedulers['prim'] = self._ReduceLROnPlateau(self.optimizers['prim'], mode='min', factor=0.5, patience=50, 
                                                     threshold=0.000001, threshold_mode='rel', cooldown=10, min_lr=0, eps=1e-08)         
        
    def setup_temp_optimizer(self):

        self.optimizers['temp'] = torch.optim.SGD(params=self.temp_params, lr=self.temperatureLR, 
                                              momentum = self.SGD_momentum, 
                                              weight_decay = self.SGD_weight_decay)            

        if self.use_temp_scheduler:
            self.schedulers['temp'] = self._ReduceLROnPlateau(self.optimizers['temp'], mode='min', factor=0.5, patience=50, 
                                                        threshold=0.000001, threshold_mode='rel', cooldown=10, min_lr=0, eps=1e-08)
 

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
        activations[0] = self.layers[0](features)
        
        for index, layer in enumerate(self.layers[1:],1):
            activations[index] = layer(activations.get(index-1))
        
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
        
        for index, layer in enumerate(self.layers[:self.embedding_layer+1]):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations.get(index - 1))
        latent_code = activations.get(len(activations) - 1)
        # latent_code = latent_code.detach().numpy()
        return latent_code

    
    def fit(
        self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
        starting_epoch: int, epochs: int, show_every: int = 1
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
            train_loss = self.epoch_train(train_loader, epoch)
            self.model_history('train', train_loss)
        
            validation_loss = self.epoch_validate(val_loader, epoch)
            self.model_history('val', validation_loss)
            
            display_epoch_metrics(self, epoch, epochs, header)
            header = False    
         



