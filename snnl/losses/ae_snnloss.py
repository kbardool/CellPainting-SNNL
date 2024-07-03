from typing import Dict, Tuple

# import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from types import SimpleNamespace
# import argparse
# from datetime import datetime
# from collections import defaultdict
# from scipy.spatial.distance import pdist, squareform, euclidean


#-------------------------------------------------------------------------------------------------------------------
#  Soft Nearest Neighborhood Loss
#-------------------------------------------------------------------------------------------------------------------     
class SNNLoss(torch.nn.Module):
    """
    A composite loss of the Soft Nearest Neighbor Loss
    computed at each hidden layer, and a softmax
    cross entropy (for classification) loss or binary
    cross entropy (for reconstruction) loss.

    Presented in
    "Improving k-Means Clustering Performance with Disentangled Internal
    Representations" by Abien Fred Agarap and Arnulfo P. Azcarraga (2020),
    and in
    "Analyzing and Improving Representations with the Soft Nearest Neighbor
    Loss" by Nicholas Frosst, Nicolas Papernot, and Geoffrey Hinton (2019).

    https://arxiv.org/abs/2006.04535/
    https://arxiv.org/abs/1902.01889/
    """

    _supported_modes = {
        "classifier"  : True,
        # "classifier"  : False,
        "resnet"      : False,
        "autoencoding": True,
        "latent_code" : True,
        "sae"         : True,
        "custom"      : False,
        "moe"         : False,
    }

    def __init__(
        self,
        mode: str = "classifier",
        device: torch.device = None,      
        temperature: float = None,
        unsupervised: bool = None,
        use_annealing: bool = False,
        use_sum: bool = False,
        code_units: int = 30,
        stability_epsilon: float = 1e-5,
        sample_size: int = None, 
        verbose = False
    ):
        """
        Constructs the Soft Nearest Neighbor Loss.

        Parameters
        ----------
        mode: str                               The mode in which the soft nearest neighbor loss will be used. 
                                                Default: [classifier]
        temperature: float                      The SNNL temperature.
        use_annealing: bool                     Whether to use annealing temperature or not.
        use_sum: bool                           If true, the sum of SNNL across all hidden layers are used.
                                                Otherwise, the minimum SNNL will be obtained.
        code_units: int                         The number of units in which the SNNL will be applied.
        stability_epsilon: float                A constant for helping SNNL computation stability.
        """
        super().__init__()
        self.mode = mode.lower()
        assert sample_size is not None, f"sample_size must be specified in SNNLoss initialization"
        if (self.mode == "latent_code") and (code_units <= 0):
            raise ValueError("[code_units] must be greater than 0 when mode == 'latent_code'." )
        assert isinstance(code_units, int), f"Expected dtype for [code_units] is int, but {code_units} is {type(code_units)}"
        

        if self.mode not in SNNLoss._supported_modes:
            raise ValueError(f"Mode {mode} is not supported.")
        
        if unsupervised is None:
            self.unsupervised = SNNLoss._supported_modes.get(self.mode)
        else:
            self.unsupervised = unsupervised
            
        self.temperature = temperature
        self.device = device 
        self.use_annealing = use_annealing
        self.use_sum = use_sum
        self.code_units = code_units
        self.stability_epsilon = stability_epsilon

        self.sample_size = sample_size
        self.verbose = verbose
        if verbose:
            print('\n'+'-'*60)
            print(f" Build SNNLoss from AE_snnloss")
            print('-'*60)
            print(f"    SNNLoss _init()_    -- mode: {mode} was found in SNNLoss._supported_modes --   is unsupervised: {SNNLoss._supported_modes.get(self.mode)}")
            print(f"    SNNLoss _init()_    -- unsupervised :     {self.unsupervised}")
            print(f"    SNNLoss _init()_    -- use_annealing :    {self.use_annealing}")
            print(f"    SNNLoss _init()_    -- sample_size :      {self.sample_size}")
            print(f"    SNNLoss _init()_    -- temperature :      {self.temperature}")

    def forward(
        self,
        model: torch.nn.Module,
        outputs: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        epoch: int = 0 ,
        batch: int = 0
    ) -> Tuple:
        """
        Defines the forward pass for the Soft Nearest Neighbor Loss.

        Parameters
        ----------
        model: torch.nn.Module              The model whose parameters will be optimized.
        features: torch.Tensor              The input features.
        labels: torch.Tensor                The corresponding labels for the input features.
        outputs: torch.Tensor               The model outputs.
        epoch: int                          The current training epoch.

        Returns
        -------
        train_loss: float                   The composite training loss.
        primary_loss: float                 The primary loss function value.
        snn_loss: float                     The soft nearest neighbor loss value.
        """ 
        # if self.use_annealing:
        #     temp_before = self.temperature.item()
        #     new_temperature = 1.0 / ((1.0 + epoch) ** 0.55)
        #     with torch.no_grad():
        #         self.temperature.copy_(new_temperature)
        #     print(f" {epoch}/{batch} - anneal temp  -  before: {temp_before:10.6f}     new_temp: {self.temperature.item():10.6f}  param: {new_temperature:10.6f}")
                 
        self.layers_snnl = []
        self.batch = batch
        self.epoch = epoch
        activations = self.compute_activations(model=model, features=features)

        for key, value in activations.items():
            if len(value.shape) > 2:
                value = value.view(value.shape[0], -1)
                
            if (self.mode == "latent_code"): 
                if (key == model.embedding_layer):
                    value = value[:, : self.code_units]
                else:
                    continue
 
            self.dist_mat = self.pairwise_cosine_distance(features=value)
            self.dist_mat_exp= self.distance_matrix_exponential(features=value)

            if torch.any(torch.isnan(self.dist_mat_exp)):
                print(f"batch {model.batch_count} - layer {key}")
                self.display_debug_info_1()
                print(f"Nan::: {self.epoch}/{self.batch} pairwise similarity matrix - {self.dist_mat.shape} Min: {self.dist_mat.min()}  Max: {self.dist_mat.max()}")        
                print(f"Nan::: {self.epoch}/{self.batch} pairwise distance exponent - {self.dist_mat_exp.shape} Min: {self.dist_mat_exp.min()}  Max: {self.dist_mat_exp.max()}")        

            self.sampling_probability = self.compute_sampling_probability()
            self.summed_masked_pick_probability = self.mask_sampling_probability(labels)
            
            snnl = torch.mean( -torch.log(self.stability_epsilon + self.summed_masked_pick_probability))
            
            # print(f" SNNLoss.forward() - key: {key:3d}  val: {value.shape}  {value.ndim}  features sum: {value.sum():.4f}  Temp:{self.temperature.item():.4f}  SNNL: {snnl:.4f}")
            # self.display_debug_info_2()
            
            if torch.isnan(snnl):
                print(f"batch {model.batch_count} - layer {key}")
                self.display_debug_info_1()
                raise ValueError(" Nan encountered...." )
 
            if self.mode == "latent_code":
                if key == model.embedding_layer:
                    self.layers_snnl.append(snnl)
                    break
            else:
                self.layers_snnl.append(snnl)
                
        self.snn_loss =  torch.min(torch.stack(self.layers_snnl))
        return self.snn_loss

    def compute_activations(
        self, model: torch.nn.Module, features: torch.Tensor
    ) -> Dict:
        """
        Returns the hidden layer activations of a model.

        Parameters
        ----------
        model: torch.nn.Module
            The model whose hidden layer representations shall be computed.
        features: torch.Tensor
            The input features.

        Returns
        -------
        activations: Dict
            The hidden layer activations of the model.
        """
        activations = dict()       
        for index, layer in enumerate(model.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations[index - 1])
                    
        return activations

    def pairwise_cosine_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns the pairwise cosine distance between two copies
        of the features matrix.

        Parameter
        ---------
        features: torch.Tensor              The input features.

        Returns
        -------
        distance_matrix: torch.Tensor       The pairwise cosine distance matrix.

        """
        normalized_a = torch.nn.functional.normalize(features, dim=1, p=2)
        pairwise_dist =  torch.matmul(normalized_a, normalized_a.T)
        return torch.clamp(1.0 - pairwise_dist, 0.0, None)
        
   
    def pairwise_euclidean_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns the pairwise Euclidean distance between two copies
        of the features matrix.

        Parameter
        ---------
        features: torch.Tensor            The input features.

        Returns
        -------
        distance_matrix: torch.Tensor     The pairwise Euclidean distance matrix.
        """
        # a, b = features.clone(), features.clone()
        # normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
        # normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
        # normalized_b = torch.conj(normalized_b).T
        # product = torch.matmul(normalized_a, normalized_b.T)
        # distance_matrix = torch.sub(torch.tensor(1.0), product)
        return torch.sqrt(((features[:, :, None] - features[:, :, None].T) ** 2).sum(1))
   
    def distance_matrix_exponential(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the exp(distance matrix).

        Parameters
        ----------
        features: torch.Tensor            The input features.
        distance_matrix: torch.Tensor     The pairwise distance matrix to normalize.
        device: torch.device              The device to use for computation.

        Returns
        -------
        dme : torch.Tensor                The exponental of all elements of pairwise distance matrix.
        """
        ## 30-3-2024 KB Added stability_epsilson to self.temperature
        dme = torch.exp(-(self.dist_mat / self.temperature)) - torch.eye(features.shape[0]).to(self.device)
        # print(f"{self.epoch}/{self.batch} distance matrix exponent   - {dme.shape} Min: {dme.min()}  Max: {dme.max()}")
        return dme

    def compute_sampling_probability(
        self, 
        # pairwise_distance_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the probability of sampling `j` based on distance between points `i` and `j`.

        Parameter
        ---------
        pairwise_distance_matrix: torch.Tensor          The normalized pairwise distance matrix.

        Returns
        -------
        pick_probability: torch.Tensor                  The probability matrix for selecting neighbors.
        """
        sampling_probability = self.dist_mat_exp / (self.stability_epsilon + self.dist_mat_exp.sum(1).unsqueeze(1) )
        sampling_probability = torch.clamp(sampling_probability,0.0,None)
        return sampling_probability

    def mask_sampling_probability(
        self, 
        labels: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Masks the sampling probability, to zero out diagonal of sampling probability, and returns the sum per row.

        Parameters
        ----------
        labels: torch.Tensor                            The labels of the input features.
        sampling_probability: torch.Tensor              The probability matrix of picking neighboring points.

        Returns
        -------
        summed_masked_pick_probability: torch.Tensor    The probability matrix of selecting a class-similar data points.
        """
        self.sample_labels = torch.arange(labels.shape[0]).to(self.device) // self.sample_size
        self.masking_matrix = torch.squeeze(torch.eq(self.sample_labels, self.sample_labels.unsqueeze(1)).float())
        self.masked_pick_probability = self.sampling_probability * self.masking_matrix
        summed_masked_pick_probability = torch.sum(self.masked_pick_probability, dim=1)   
        return summed_masked_pick_probability



    def display_debug_info_1(self):
        print('--- NAN Encountred ','-'*100)

        # print('--- Distance matrix ','-'*80)
        print(f"{self.epoch}/{self.batch} : Temperature              : {self.temperature.item()}  ")
        print(f"{self.epoch}/{self.batch} : Pairwise distance matrix : {self.dist_mat.shape}  "
              f"Min: {self.dist_mat.min()}  argmin: {self.dist_mat.argmin().item()}   Max: {self.dist_mat.max()}  "
              f"argmax:  {self.dist_mat.argmax().item()}  Sum: {self.dist_mat.sum()}")        
        # print(f"{self.epoch}/{self.batch} normalized features tensor - {normalized_a.shape} Min: {normalized_a.min()}  Max: {normalized_a.max()}")        
        
        # print('--- -(distance matrix/temp) ','-'*80)
        _tmp1 = -(self.dist_mat / self.temperature) 
        print(f"{self.epoch}/{self.batch} : -(Distance matrix / Temp): {_tmp1.shape}  "
              f"Min: {_tmp1.min()}  argmin: {_tmp1.argmin().item()}   Max: {_tmp1.max()}  "
              f"argmax:  {_tmp1.argmax().item()}  Sum: {_tmp1.sum()}")
        
        # print('--- exp(-(Distance matrix/temp) ','-'*80)
        _tmp1 =  torch.exp(-(self.dist_mat / self.temperature)) 
        print(f"{self.epoch}/{self.batch} : exp((-Dist mat / Temp))  : {_tmp1.shape}  "
              f"Min: {_tmp1.min()}  argmin: {_tmp1.argmin().item()}   Max: {_tmp1.max()}  "
              f"argmax:  {_tmp1.argmax().item()}  Sum: {_tmp1.sum()}")
        print(f"{self.epoch}/{self.batch} : Distance matrix exponent : {self.dist_mat_exp.shape}  "
              f"Min: {self.dist_mat_exp.min()}  argmin: {self.dist_mat.argmin().item()}   Max: {self.dist_mat_exp.max()}  "
              f"argmax:  {self.dist_mat.argmax().item()}  Sum: {self.dist_mat_exp.sum()}")
        
        # print('--- Pick Probability','-'*80)
        # print(self.sampling_probability)
        print(f"{self.epoch}/{self.batch} : Pick Probability Matrix  : {self.sampling_probability.shape}  "
              f"Min: {self.sampling_probability.min()}  argmin: {self.sampling_probability.argmin()}   Max: {self.sampling_probability.max()}  "
              f"argmax:  {self.sampling_probability.argmax()}  Sum: {self.sampling_probability.sum()}")
        
        # print('--- Summed Picked Probability ','-'*80)
        summ = (self.stability_epsilon + self.summed_masked_pick_probability)
        print(f"{self.epoch}/{self.batch} : Summed pick_probability  :  {summ.shape}  "
              f"Min {summ.min().item()}  argmin: {summ.argmin()}  Max: {summ.max()}  " 
              f"argmax:  {summ.argmax().item()}  Sum: {summ.sum()}")
        print(f"{self.epoch}/{self.batch} : mean(-log(summed_picked_probabilty)) -  {torch.mean(-torch.log(summ))}")
        print('-'*120)
        
    def display_debug_info_2(self):
        print('--- display_debug_info_2() ','-'*100)
        print(f"{self.epoch}/{self.batch} pairwise similarity matrix : {self.dist_mat.shape}  "
              f"Min: {self.dist_mat.min()}  Max: {self.dist_mat.max()}  Sum: {self.dist_mat.sum():.4f}")   
        print(f"{self.epoch}/{self.batch} distance matrix exponent   : {self.dist_mat_exp.shape}  "
              f"Min: {self.dist_mat_exp.min()}  Max: {self.dist_mat_exp.max()}   Sum: {self.dist_mat_exp.sum()}")
        print(f"{self.epoch}/{self.batch} sampling prob              : {self.sampling_probability.shape}  "
              f"Min: {self.sampling_probability.min()}  Max: {self.sampling_probability.max()}   Sum: {self.sampling_probability.sum():.4f}")
        print(f"{self.epoch}/{self.batch} sum mask pick              : {self.summed_masked_pick_probability.shape}  "
              f"Min: {self.summed_masked_pick_probability.min()}  Max: {self.summed_masked_pick_probability.max():.4f}  Sum: {self.summed_masked_pick_probability.sum():.4f}")
        print(f"                   - snnl: {self.snnl:.5f}")
        print(self.sampling_probability[:12,:12])
        print(self.sample_labels )