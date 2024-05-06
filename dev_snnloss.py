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
from typing import Dict, List, Tuple
from collections import defaultdict


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
        mode: str
            The mode in which the soft nearest neighbor loss
            will be used. Default: [classifier]
        # criterion: object
        #     The primary loss to use.
        #     Default: [torch.nn.CrossEntropyLoss()]
 
        temperature: float
            The SNNL temperature.
        use_annealing: bool
            Whether to use annealing temperature or not.
        use_sum: bool
            If true, the sum of SNNL across all hidden layers are used.
            Otherwise, the minimum SNNL will be obtained.
        code_units: int
            The number of units in which the SNNL will be applied.
        stability_epsilon: float
            A constant for helping SNNL computation stability.
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
        self.use_annealing = use_annealing
        self.use_sum = use_sum
        self.code_units = code_units
        self.stability_epsilon = stability_epsilon

        self.sample_size = sample_size
        self.verbose = verbose
        print('\n'+'-'*60)
        print(f" Build SNNLoss from NOTEBOOK")
        print('-'*60)
        print(f"    SNNLoss _init()_    -- mode: {mode} was found in SNNLoss._supported_modes --   is unsupervised: {SNNLoss._supported_modes.get(self.mode)}")
        # print(f"    SNNLoss _init()_    -- primary_criterion: {self.primary_criterion}")
        print(f"    SNNLoss _init()_    -- unsupervised :     {self.unsupervised}")
        print(f"    SNNLoss _init()_    -- use_annealing :    {self.use_annealing}")
        print(f"    SNNLoss _init()_    -- sample_size :      {self.sample_size}")
        print(f"    SNNLoss _init()_    -- temperature :      {self.temperature.item()}")

    def forward(
        self,
        model: torch.nn.Module,
        outputs: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        # epoch: int,
    ) -> Tuple:
        """
        Defines the forward pass for the Soft Nearest Neighbor Loss.

        Parameters
        ----------
        model: torch.nn.Module
            The model whose parameters will be optimized.
        features: torch.Tensor
            The input features.
        labels: torch.Tensor
            The corresponding labels for the input features.
        outputs: torch.Tensor
            The model outputs.
        epoch: int
            The current training epoch.

        Returns
        -------
        train_loss: float
            The composite training loss.
        primary_loss: float
            The primary loss function value.
        snn_loss: float
            The soft nearest neighbor loss value.
        """ 
        # if self.use_annealing:
        #     self.temperature = 1.0 / ((1.0 + epoch) ** 0.55)
        #     print(f"anneal temperature : {self.temperature}")

        # if self.verbose:  
        #     print(f" SNNLoss.forward() - self.mode       {self.mode}")
        #     print(f" SNNLoss.forward() - features.shape  {features.shape}")
        #     print(f" SNNLoss.forward() - temperature     {self.temperature}")
        #     print(f" SNNLoss.forward() - labels.shape    {labels.shape}\n {labels}")
        #     print(f" outputs: {type(outputs)}   shape: {outputs.shape}   {outputs[:10]}")
        #     print(f" labels : {type(labels)}    shape: {labels.shape}    {labels[:10]}")
                
        self.layers_snnl = []
        activations = self.compute_activations(model=model, features=features)

        for key, value in activations.items():
            if len(value.shape) > 2:
                value = value.view(value.shape[0], -1)
                
            if (self.mode == "latent_code"): 
                if (key == model.embedding_layer):
                    value = value[:, : self.code_units]
                else:
                    continue
            elif (self.mode == "sae") and (key == 9) :
                value = value[:, : self.code_units]
              
            self.distance_matrix = self.pairwise_cosine_distance(features=value)
            self.pairwise_distance_matrix = self.normalize_distance_matrix(features=value,device=model.device)
            self.sampling_probability = self.compute_sampling_probability()
            self.summed_masked_pick_probability = self.mask_sampling_probability(labels, device=model.device)
            snnl = torch.mean( -torch.log(self.stability_epsilon + self.summed_masked_pick_probability))
            
            # print(f" SNNLoss.forward() - key: {key:3d}  val: {value.shape}  {value.ndim}  features sum: {value.sum():.4f}  Temp:{self.temperature.item():.4f}  SNNL: {snnl:.4f}")
            # self.display_debug_info_2()
            
            if torch.isnan(snnl):
                print(f"batch {model.batch_count} - layer {key}")
                self.display_debug_info_1()
 
            if self.mode == "latent_code":
                if key == model.embedding_layer:
                    self.layers_snnl.append(snnl)
                    break
            elif self.mode == "resnet":
                if key > 6:
                    self.layers_snnl.append(snnl)
            else:
                self.layers_snnl.append(snnl)
                
        if self.use_sum:
            self.snn_loss = torch.stack(self.layers_snnl).sum()
            # print(f" latent code snn_loss (use sum = True) - {torch.stack(self.layers_snnl)}")
            # print(f" latent code snn_loss (use sum = True) - {self.snn_loss}")
        else:
            self.snn_loss =  torch.min(torch.stack(self.layers_snnl))
            # print(f" latent code snn_loss (use sum = False) - {torch.stack(self.layers_snnl)}")
            # print(f" latent code snn_loss (use sum = False) - {self.snn_loss} ")
 
        # Get Primary loss
        # primary_loss = self.primary_criterion(outputs, features if self.unsupervised else labels)
        # if self.unsupervised:
        #     # print(f" call  primary_criterion(outputs, features)")
        #     self.primary_loss = torch.tensor(0,requires_grad = True, dtype=torch.float32, device = model.device)
        #     # print(f"    outputs:  {type(outputs)}  {outputs.device}  {outputs.shape}")
        #     # print(f"    features: {type(features)}   {features.device}  {features.shape}")
        #     # primary_loss = self.primary_criterion(outputs, features)
        #     # print(f"primary loss {primary_loss}   {type(primary_loss)}  {primary_loss.shape} {primary_loss.size()} {primary_loss.dtype} {primary_loss.requires_grad}") 
        #     pass
        # else:
        #     # print(f" call  primary_criterion(outputs, labels)")
        #     # self.primary_loss = self.primary_criterion(outputs, torch.unsqueeze(labels,-1))
        #     self.primary_loss = self.primary_criterion(outputs.squeeze(), labels)
        #     # print(f" outputs {outputs.squeeze()[:10]} ") 
        #     # print(f" labels: {labels[:10]}  ")
        #     # print(f" loss {self.primary_criterion(outputs.squeeze()[:2], labels[:2])} ")
        #     # print(f" {self.primary_loss.size()} {self.primary_loss.dtype} {self.primary_loss}")
        
        # if self.mode != "moe":
        #     self.train_loss = torch.add(self.primary_loss, torch.mul(self.factor, self.snn_loss))
        #     return self.train_loss, self.primary_loss, self.snn_loss
        # else:
        #     return self.primary_loss, self.snn_loss
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
        if self.verbose:
            print("_"*40)
            print(f"\n COMPUTE_ACTIVATIONS()")
            print("_"*40)
        activations = dict()
        
        if self.mode == "classifier":
            # layers = model.layers[:-1] if self.mode == "classifier" else model.layers
            layers = model.layers
            for index, layer in enumerate(layers):
                if index == 0:
                    activations[index] = layer(features)
                else:
                    activations[index] = layer(activations[index - 1])
                    
        elif self.mode in [ "autoencoding", "latent_code"]:
            layers = model.layers
            for index, layer in enumerate(layers):
                if index == 0:
                    activations[index] = layer(features)
                else:
                    activations[index] = layer(activations[index - 1])

        elif self.mode == "sae":
            for index, layer in enumerate(
                torch.nn.Sequential(*model.encoder, *model.decoder)
            ):
                activations[index] = layer(
                    features if index == 0 else activations[index - 1]
                )
        elif self.mode == "resnet":
            for index, (name, layer) in enumerate(list(model.resnet.named_children())):
                if index == 0:
                    activations[index] = layer(features)
                elif index == 9:
                    value = activations[index - 1].view(
                        activations[index - 1].shape[0], -1
                    )
                    activations[index] = layer(value)
                else:
                    activations[index] = layer(activations[index - 1])
        elif self.mode == "custom":
            for index, layer in enumerate(list(model.children())):
                activations[index] = (
                    layer(features) if index == 0 else layer(activations[index - 1])
                )
        elif self.mode == "moe":
            layers = dict(model.named_children())
            layers = layers.get("feature_extractor")
            if isinstance(layers[0], torch.nn.Linear) and len(features.shape) > 2:
                features = features.view(features.shape[0], -1)
            for index, layer in enumerate(layers):
                activations[index] = (
                    layer(features) if index == 0 else layer(activations.get(index - 1))
                )
        return activations

    def pairwise_cosine_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns the pairwise cosine distance between two copies
        of the features matrix.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        distance_matrix: torch.Tensor
            The pairwise cosine distance matrix.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> snnl.pairwise_cosine_distance(a)
        tensor([[1.1921e-07, 7.4125e-02, 1.8179e-02, 1.0152e-01],
                [7.4125e-02, 1.1921e-07, 1.9241e-02, 2.2473e-03],
                [1.8179e-02, 1.9241e-02, 1.1921e-07, 3.4526e-02],
                [1.0152e-01, 2.2473e-03, 3.4526e-02, 0.0000e+00]])
        """
        a, b = features.clone(), features.clone()
        normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
        normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
        # normalized_b = torch.conj(normalized_b).T
        product = torch.matmul(normalized_a, normalized_b.T)
        distance_matrix = torch.sub(torch.tensor(1.0), product)
        return distance_matrix
   
    def pairwise_euclidean_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns the pairwise Euclidean distance between two copies
        of the features matrix.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        distance_matrix: torch.Tensor
            The pairwise Euclidean distance matrix.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> snnl.pairwise_euclidean_distance(a)
        tensor([[1.1921e-07, 7.4125e-02, 1.8179e-02, 1.0152e-01],
                [7.4125e-02, 1.1921e-07, 1.9241e-02, 2.2473e-03],
                [1.8179e-02, 1.9241e-02, 1.1921e-07, 3.4526e-02],
                [1.0152e-01, 2.2473e-03, 3.4526e-02, 0.0000e+00]])
        """
        a, b = features.clone(), features.clone()
        normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
        normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
        # normalized_b = torch.conj(normalized_b).T
        product = torch.matmul(normalized_a, normalized_b.T)
        distance_matrix = torch.sub(torch.tensor(1.0), product)
        return distance_matrix
   
    def normalize_distance_matrix(
        self,
        features: torch.Tensor,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> torch.Tensor:
        """
        Normalizes the pairwise distance matrix.

        Parameters
        ----------
        features: torch.Tensor
            The input features.
        distance_matrix: torch.Tensor
            The pairwise distance matrix to normalize.
        device: torch.device
            The device to use for computation.

        Returns
        -------
        pairwise_distance_matrix: torch.Tensor
            The normalized pairwise distance matrix.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> distance_matrix = snnl.pairwise_cosine_distance(a)
        >>> snnl.normalize_distance_matrix(a, distance_matrix, device=torch.device("cpu"))
        tensor([[-1.1921e-07,  9.2856e-01,  9.8199e-01,  9.0346e-01],
                [ 9.2856e-01, -1.1921e-07,  9.8094e-01,  9.9776e-01 ],
                [ 9.8199e-01,  9.8094e-01, -1.1921e-07,  9.6606e-01 ],
                [ 9.0346e-01,  9.9776e-01,  9.6606e-01,  0.0000e+00 ]])
        """
        ## 30-3-2024 KB Added stability_epsilson to self.temperature
        # pairwise_distance_matrix = torch.exp(
        #     -(distance_matrix / (self.stability_epsilon + self.temperature))
        # ) - torch.eye(features.shape[0]).to(device)

        pairwise_distance_matrix = torch.exp(-(self.distance_matrix / self.temperature)) - torch.eye(features.shape[0]).to(device)
        
        return pairwise_distance_matrix

    def compute_sampling_probability(
        self, 
        # pairwise_distance_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the probability of sampling `j` based
        on distance between points `i` and `j`.

        Parameter
        ---------
        pairwise_distance_matrix: torch.Tensor
            The normalized pairwise distance matrix.

        Returns
        -------
        pick_probability: torch.Tensor
            The probability matrix for selecting neighbors.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> distance_matrix = snnl.pairwise_cosine_distance(a)
        >>> distance_matrix = snnl.normalize_distance_matrix(a, distance_matrix)
        >>> snnl.compute_sampling_probability(distance_matrix)
        tensor([[-4.2363e-08,  3.2998e-01,  3.4896e-01,  3.2106e-01],
                [ 3.1939e-01, -4.1004e-08,  3.3741e-01,  3.4319e-01 ],
                [ 3.3526e-01,  3.3491e-01, -4.0700e-08,  3.2983e-01 ],
                [ 3.1509e-01,  3.4798e-01,  3.3693e-01,  0.0000e+00 ]])
        """
        sampling_probability = self.pairwise_distance_matrix / (self.stability_epsilon + self.pairwise_distance_matrix.sum(1).unsqueeze(1) )
        sampling_probability = torch.clamp(sampling_probability,0.0,None)
        
        return sampling_probability

    def mask_sampling_probability(
        self, 
        labels: torch.Tensor, 
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> torch.Tensor:
        """
        Masks the sampling probability, to zero out diagonal
        of sampling probability, and returns the sum per row.

        Parameters
        ----------
        labels: torch.Tensor
            The labels of the input features.
        sampling_probability: torch.Tensor
            The probability matrix of picking neighboring points.

        Returns
        -------
        summed_masked_pick_probability: torch.Tensor
            The probability matrix of selecting a
            class-similar data points.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> distance_matrix = snnl.pairwise_cosine_distance(a)
        >>> distance_matrix = snnl.normalize_distance_matrix(a, distance_matrix)
        >>> pick_probability = snnl.compute_sampling_probability(distance_matrix)
        >>> snnl.mask_sampling_probability(labels, pick_probability)
        tensor([0.3490, 0.3432, 0.3353, 0.3480])
        """
        self.sample_labels = torch.arange(labels.shape[0]).to(device) // self.sample_size
        self.masking_matrix = torch.squeeze(torch.eq(self.sample_labels, self.sample_labels.unsqueeze(1)).float())
        self.masked_pick_probability = self.sampling_probability * self.masking_matrix
        summed_masked_pick_probability = torch.sum(self.masked_pick_probability, dim=1)
    
        return summed_masked_pick_probability

    def display_debug_info_1(self):
        print('-'*80)
        print('--- Distance matrix ','-'*80)
        print(self.distance_matrix)
        print(self.distance_matrix.min().item(), self.distance_matrix.argmin().item())
        print('--- Pairwise distance matrix ','-'*80)
        print(self.pairwise_distance_matrix)
        print(self.pairwise_distance_matrix.min().item(), self.pairwise_distance_matrix.argmin().item())
        print('--- Pick Probability','-'*80)
        print(self.pick_probability)
        print('min  ', self.pick_probability.min().item(), self.pick_probability.argmin().item())
        print('--- Summed Picked Probability ','-'*80)
        summ = (self.stability_epsilon + self.summed_masked_pick_probability)
        print(summ) 
        print('min summed_masked_pick_probability ', summ.min().item(), summ.argmin().item())
        print( -torch.log(summ))
        
    def display_debug_info_2(self):
        print(f"                   - distance mat : {self.distance_matrix.shape}  {self.distance_matrix.min()}   {self.distance_matrix.max()}   sum: {self.distance_matrix.sum():.4f}")
        print(f"                   - pairwise dist: {self.pairwise_distance_matrix.shape}  {self.pairwise_distance_matrix.min()}   {self.pairwise_distance_matrix.max()}   sum: {self.pairwise_distance_matrix.sum():.4f}")
        print(f"                   - sampling prob: {self.sampling_probability.shape}  {self.sampling_probability.min()}  {self.sampling_probability.max()}   sum: {self.sampling_probability.sum():.4f}")
        print(f"                   - sum mask pick: {self.summed_masked_pick_probability.shape}  {self.summed_masked_pick_probability.min()} {self.summed_masked_pick_probability.max():.4f}"
              f" sum: {self.summed_masked_pick_probability.sum():.4f}")
        print(f"                   - snnl: {snnl:.5f}")
        print(self.sampling_probability[:12,:12])
        print(self.sample_labels )