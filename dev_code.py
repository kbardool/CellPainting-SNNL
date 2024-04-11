from typing import Dict, Tuple

# import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"

import numpy as np
import pandas as pd
import torch
import argparse
from pt_datasets import create_dataloader
from typing import Dict, List, Tuple


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
        "classifier"  : False,
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
        criterion: object = torch.nn.CrossEntropyLoss(),
        factor: float = 100.0,
        temperature: float = None,
        unsupervised: bool = None,
        use_annealing: bool = False,
        use_sum: bool = False,
        code_units: int = 30,
        stability_epsilon: float = 1e-5,
        sample_size: int = 1, 
        verbose = False
    ):
        """
        Constructs the Soft Nearest Neighbor Loss.

        Parameters
        ----------
        mode: str
            The mode in which the soft nearest neighbor loss
            will be used. Default: [classifier]
        criterion: object
            The primary loss to use.
            Default: [torch.nn.CrossEntropyLoss()]
        factor: float
            The balance factor between SNNL and the primary loss.
            A positive factor implies SNNL minimization, while a negative
            factor implies SNNL maximization.
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
        print(f" Build SNNLoss dfrom NOTEBOOK")
        mode = mode.lower()
        if mode not in SNNLoss._supported_modes:
            raise ValueError(f"Mode {mode} is not supported.")
        if (mode == "latent_code") and (code_units <= 0):
            raise ValueError("[code_units] must be greater than 0 when mode == 'latent_code'." )
        assert isinstance(
            code_units, int
        ), f"Expected dtype for [code_units]: int, but {code_units} is {type(code_units)}"
        
        self.mode = mode
        self.primary_criterion = criterion
        if unsupervised is None:
            self.unsupervised = SNNLoss._supported_modes.get(self.mode)
        else:
            self.unsupervised = unsupervised
        self.factor = factor
        self.temperature = temperature
        self.use_annealing = use_annealing
        self.use_sum = use_sum
        self.code_units = code_units
        self.stability_epsilon = stability_epsilon
        
        self.sample_size = sample_size
        self.verbose = verbose
        print(f" Building SNNLoss from NOTEBOOK")
        print(f"    SNNLoss _init()_    -- mode: {mode} was found in SNNLoss._supported_modes --   is unsupervised: {SNNLoss._supported_modes.get(self.mode)}")
        print(f"    SNNLoss _init()_    -- primary_criterion: {self.primary_criterion}")
        print(f"    SNNLoss _init()_    -- unsupervised :     {self.unsupervised}")
        print(f"    SNNLoss _init()_    -- use_annealing :    {self.use_annealing}")
        print(f"    SNNLoss _init()_    -- sample_size :      {self.sample_size}")

    def forward(
        self,
        model: torch.nn.Module,
        outputs: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
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

        if self.verbose:
            
            print(f" SNNLoss.forward() - self.mode       {self.mode}")
            print(f" SNNLoss.forward() - features.shape  {features.shape}")
            print(f" SNNLoss.temperature                 {self.temperature}")
            print(f" SNNLoss.forward() - labels.shape    {labels.shape}\n {labels}")
            
        ## If Unsupervised we 
        # primary_loss = self.primary_criterion(outputs, features if self.unsupervised else labels)
        
        if self.unsupervised:
            self.primary_loss = torch.tensor(0,requires_grad = True, dtype=torch.float32, device = model.device)
            # print(f" call  primary_criterion(outputs, features)")
            # print(f"    outputs:  {type(outputs)}  {outputs.device}  {outputs.shape}")
            # print(f"    features: {type(features)}   {features.device}  {features.shape}")
            # primary_loss = self.primary_criterion(outputs, features)
            # print(f"primary loss {primary_loss}   {type(primary_loss)}  {primary_loss.shape} {primary_loss.size()} {primary_loss.dtype} {primary_loss.requires_grad}") 
            pass
        else:
            self.primary_loss = self.primary_criterion(outputs, labels)
            # print(f" call  primary_criterion(outputs, labels)")
            # print(f"primary losss {primary_loss}   {type(primary_loss)}  {primary_loss.shape} {primary_loss.size()} {primary_loss.dtype} {primary_loss.requires_grad}") 
        activations = self.compute_activations(model=model, features=features)

        self.layers_snnl = []

        for key, value in activations.items():
            if self.verbose:
                print(f"Number of layers {len(activations)}")
                
            if len(value.shape) > 2:
                value = value.view(value.shape[0], -1)
            if key == 7 and self.mode == "latent_code":
                value = value[:, : self.code_units]
            elif key == 9 and self.mode == "sae":
                value = value[:, : self.code_units]
                
            self.distance_matrix = self.pairwise_cosine_distance(features=value)

            self.pairwise_distance_matrix = self.normalize_distance_matrix(features=value, 
                                                                           distance_matrix=self.distance_matrix, 
                                                                           device=model.device)

            self.pick_probability = self.compute_sampling_probability(self.pairwise_distance_matrix)
            self.summed_masked_pick_probability = self.mask_sampling_probability(labels = labels, 
                                                                                 sampling_probability = self.pick_probability, 
                                                                                 device=model.device)
            snnl = torch.mean( -torch.log(self.stability_epsilon + self.summed_masked_pick_probability))
            
            if torch.isnan(snnl):
                # print('-'*80)
                print(f"batch {model.batch_count} - layer {key}")
                # print('--- Distance matrix ','-'*80)
                # print(self.distance_matrix)
                # print(self.distance_matrix.min().item(), self.distance_matrix.argmin().item())
                # print('--- Pairwise distance matrix ','-'*80)
                # print(self.pairwise_distance_matrix)
                # print(self.pairwise_distance_matrix.min().item(), self.pairwise_distance_matrix.argmin().item())
                # print('--- Pick Probability','-'*80)
                # print(self.pick_probability)
                # print('min  ', self.pick_probability.min().item(), self.pick_probability.argmin().item())
                # print('--- Summed Picked Probability ','-'*80)
                # summ = (self.stability_epsilon + self.summed_masked_pick_probability)
                # print(summ) 
                # print('min summed_masked_pick_probability ', summ.min().item(), summ.argmin().item())
                # print( -torch.log(summ))
 
            if self.mode == "latent_code":
                if key == 7:
                    self.layers_snnl.append(snnl)
                    break
            elif self.mode == "resnet":
                if key > 6:
                    self.layers_snnl.append(snnl)
            else:
                self.layers_snnl.append(snnl)
                
        if self.use_sum:
            self.snn_loss = torch.stack(self.layers_snnl).sum()
        else:
            self.snn_loss = torch.stack(self.layers_snnl)
            self.snn_loss = torch.min(self.snn_loss)
        
        # if self.verbose:
        #     print('layers_snnl')
        #     print(layers_snnl)
        #     print(f'snn_loss: {snn_loss}')

        if self.mode != "moe":
            self.train_loss = torch.add(self.primary_loss, torch.mul(self.factor, self.snn_loss))
            return self.train_loss, self.primary_loss, self.snn_loss
        else:
            return self.primary_loss, self.snn_loss

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
        
        if self.mode in ["classifier", "autoencoding", "latent_code"]:
            # layers = model.layers[:-1] if self.mode == "classifier" else model.layers
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
        distance_matrix: torch.Tensor,
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

        pairwise_distance_matrix = torch.exp(-(distance_matrix / self.temperature)) - torch.eye(features.shape[0]).to(device)
        
        return pairwise_distance_matrix

    def compute_sampling_probability(
        self, pairwise_distance_matrix: torch.Tensor
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
        pick_probability = pairwise_distance_matrix / (
            self.stability_epsilon + torch.sum(pairwise_distance_matrix, 1).view(-1, 1)
        )
        # if self.verbose:        
        #     torch.set_printoptions(linewidth=150)
        #     print(f"\n pick_probability {pick_probability.shape}")
        #     print(pick_probability)      

        return pick_probability

    def mask_sampling_probability(
        self, 
        labels: torch.Tensor, 
        sampling_probability: torch.Tensor,
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
        sample_labels = torch.arange(labels.shape[0]).to(device) // self.sample_size
        masking_matrix = torch.squeeze(torch.eq(sample_labels, sample_labels.unsqueeze(1)).float())
        # masking_matrix = torch.squeeze(torch.eq(labels, labels.unsqueeze(1)).float())

        masked_pick_probability = sampling_probability * masking_matrix
        
        summed_masked_pick_probability = torch.sum(masked_pick_probability, dim=1)
        self.sample_labels = sample_labels
        self.masking_matrix= masking_matrix
        self.masked_pick_probability = masked_pick_probability
 
        
        # if self.verbose:
        #     torch.set_printoptions(linewidth=150)
        #     print(f"\n mask_sampling_probability()")
        #     print("_"*40)    
        #     print(f"\n labels  {labels.shape}")
        #     print(labels)
        #     print(f"\n masking_matrix {masking_matrix.shape}")
        #     print(masking_matrix)
        #     print(f"\n masked_pick_probability {masked_pick_probability.shape}")
        #     print(masked_pick_probability)
        #     print(f"\n summed_masked_pick_probability: {summed_masked_pick_probability.shape}")
        #     print(summed_masked_pick_probability)

        return summed_masked_pick_probability

#-------------------------------------------------------------------------------------------------------------------
#  Model Base class  
#-------------------------------------------------------------------------------------------------------------------

class Model(torch.nn.Module):

    _unsupervised_supported_modes = {
        "classifier"  : False,
        "resnet"      : False,
        "autoencoding": True,
        "latent_code" : True,
        "sae"         : True,
        "custom"      : False,
        "moe"         : False,
    }    
    def __init__(
        self,
        mode: str,
        criterion: object,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        use_snnl: bool = False,
        factor: float = 100.0,
        temperature: float = 100.0,
        use_annealing: bool = False,
        use_sum: bool = False,
        unsupervised: bool = None,
        code_units: int = 0,
        stability_epsilon: float = 1e-5,
        temperatureLR: float = 1e-3,
        # batch_size: int = 0,
        sample_size: int = 1,
        verbose: bool = False,
    ):
        super().__init__()
        mode = mode.lower()

        self.mode = mode
        self.device = device
        self.train_loss = []
        self.use_snnl = use_snnl
        self.factor = factor
        self.code_units = code_units
        self.stability_epsilon = stability_epsilon
        self.verbose = verbose
        self.primary_criterion = criterion
        self.sample_size = sample_size
        self.temperature_gradients = [] 
        self.temperatureLR = temperatureLR
        
        print(f" Building Base Model from NOTEBOOK")
        print(f"    Model_init()_    -- Crtierion  :       {criterion}")
        print(f"    Model_init()_    -- temperature :      {temperature}")
        print(f"    Model_init()_    -- temperature LR:    {temperatureLR}")
        print(f"    Model_init()_    -- mode:              {mode}")
        print(f"    Model_init()_    -- unsupervised :     {unsupervised}")
        print(f"    Model_init()_    -- use_snnl :         {use_snnl}")
        
        if unsupervised is None:
            self.unsupervised = self._unsupervised_supported_modes.get(self.mode)
            print(f" for {self.mode} support for unsupervised is {self.unsupervised}")
        else:
            self.unsupervised = unsupervised
            
        if self.use_snnl:
            if temperature is not None:
                self.temperature = torch.nn.Parameter(data=torch.tensor([temperature]), requires_grad=True)
                self.register_parameter(name="temperature", param=self.temperature)
            else:
                self.temperature = temperature
            # self.temperature = temperature
            self.use_annealing = use_annealing
            self.use_sum = use_sum
            self.snnl_criterion = SNNLoss(
                mode=self.mode,
                criterion=self.primary_criterion,
                factor=self.factor,
                temperature=self.temperature,
                use_annealing=self.use_annealing,
                use_sum=self.use_sum,
                code_units=self.code_units,
                sample_size = self.sample_size,
                stability_epsilon=self.stability_epsilon,
                unsupervised=self.unsupervised,
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

    def sanity_check(
        self,
        data_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        show_every: int = 2,
    ):
        """
        Trains the model on a subset of the dataset.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
            The data loader that consists of the data pipeline.
        epochs: int
            The number of epochs to train the model.
        show_every:
            The epoch interval between progress displays.
        """
        batch_size = data_loader.batch_size
        subset = len(data_loader.dataset.data) * 0.10
        subset = int(subset)
        assert subset > batch_size, "[subset] must be greater than [batch_size]."
        features = data_loader.dataset.data[:subset] / 255.0
        labels = data_loader.dataset.targets[:subset]
        dataset = torch.utils.data.TensorDataset(features, labels)
        data_loader = create_dataloader(
            dataset=dataset, batch_size=batch_size, num_workers=0
        )
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_features, batch_labels in data_loader:
                if self.name in ["Autoencoder", "DNN"]:
                    batch_features = batch_features.view(batch_features.shape[0], -1)
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(features=batch_features)
                train_loss = self.criterion(
                    outputs,
                    batch_labels if self.name in ["CNN", "DNN"] else batch_features,
                )
                epoch_loss += train_loss.item()
                train_loss.backward()
                self.optimizer.step()
            epoch_loss /= len(data_loader)
            if (epoch + 1) % show_every == 0:
                print(f"epoch {epoch + 1}/{epochs}")
                print(f"mean loss = {epoch_loss:4f}")

    # def epoch_train(
    #     self, data_loader: torch.utils.data.DataLoader, epoch: int = None
    # ) -> Tuple:
    #     """
    #     Trains a model for one epoch.

    #     Parameters
    #     ----------
    #     data_loader: torch.utils.dataloader.DataLoader
    #         The data loader object that consists of the data pipeline.
    #     epoch: int
    #         The current epoch training index.

    #     Returns
    #     -------
    #     epoch_loss: float
    #         The epoch loss.
    #     epoch_snn_loss: float
    #         The soft nearest neighbor loss for an epoch.
    #     epoch_xent_loss: float
    #         The cross entropy loss for an epoch.
    #     epoch_accuracy: float
    #         The epoch accuracy.
    #     """
    #     if self.use_snnl:
    #         epoch_primary_loss = 0
    #         epoch_snn_loss = 0
    #     if self.name == "DNN" or self.name == "CNN":
    #         epoch_accuracy = 0
    #     epoch_loss = 0
    #     batch_count = 0
    #     for batch_features, batch_labels in data_loader:
    #         if self.name in ["Autoencoder", "DNN"]:
    #             batch_features = batch_features.view(batch_features.shape[0], -1)
    #         batch_features = batch_features.to(self.device)
    #         batch_labels = batch_labels.to(self.device)
            
    #         self.optimizer.zero_grad()
        
    #         outputs = self.forward(features=batch_features)
            
    #         if self.use_snnl:
    #             train_loss, primary_loss, snn_loss = self.snnl_criterion(
    #                 model=self,
    #                 outputs=outputs,
    #                 features=batch_features,
    #                 labels=batch_labels,
    #                 epoch=epoch,
    #             )
    #             epoch_loss += train_loss.item()
    #             epoch_snn_loss += snn_loss.item()
    #             epoch_primary_loss += primary_loss.item()
    #         else:
    #             train_loss = self.criterion(
    #                 outputs,
    #                 batch_labels
    #                 if self.name == "DNN" or self.name == "CNN"
    #                 else batch_features,
    #             )
    #             epoch_loss += train_loss.item()
            
    #         if self.name == "DNN" or self.name == "CNN":
    #             train_accuracy = (outputs.argmax(1) == batch_labels).sum().item() / len(
    #                 batch_labels
    #             )
    #             epoch_accuracy += train_accuracy
            
    #         train_loss.backward()
    #         self.optimizer.step()
            
    #         if self.use_snnl and self.temperature is not None:
    #             self.optimize_temperature()

    #         batch_count +=1
        
    #     epoch_loss /= batch_count ## len(data_loader)
        
    #     if self.name in ["DNN", "CNN"]:
    #         epoch_accuracy /= batch_count ## len(data_loader)
    #     if self.use_snnl:
    #         epoch_snn_loss /= batch_count ## len(data_loader)
    #         epoch_primary_loss /= batch_count  ## len(data_loader)
    #         if self.name == "DNN" or self.name == "CNN":
    #             print(f" SNNLoss() - epoch_loss {epoch_loss:.6f}, epoch_snn_loss, {epoch_snn_loss:.6f},  epoch_primary_loss, {epoch_primary_loss:.6f}, epoch_accuracy,  {epoch_accuracy:.6f}")
    #             if self.verbose:
    #                 print(f" SNNLoss() - epoch_loss {epoch_loss}, epoch_snn_loss, {epoch_snn_loss},  epoch_primary_loss, {epoch_primary_loss}, epoch_accuracy,  {epoch_accuracy}")
    #             return epoch_loss, epoch_snn_loss, epoch_primary_loss, epoch_accuracy
    #         else:
    #             if self.verbose:
    #                 print(f"  SNNLoss() - epoch_loss {epoch_loss}, epoch_snn_loss, {epoch_snn_loss},  epoch_primary_loss, {epoch_primary_loss}")
    #             return epoch_loss, epoch_snn_loss, epoch_primary_loss
    #     else:
    #         if self.name == "DNN" or self.name == "CNN":
    #             if self.verbose:
    #                 print(f"  SNNLoss() - epoch_loss {epoch_loss}, epoch_snn_loss, {epoch_snn_loss}")
    #             return epoch_loss, epoch_accuracy
    #         else:
    #             if self.verbose:
    #                 print(f"  SNNLoss() - epoch_loss {epoch_loss}")
    #             return epoch_loss

        
    def epoch_train(self, data_loader: torch.utils.data.DataLoader, epoch: int = None, factor: int = 1, verbose: bool = False) -> Tuple:
        epoch_loss = 0
        epoch_ttl_loss = 0
        epoch_primary_loss = 0
        epoch_snn_loss = 0
        epoch_accuracy = 0
        
        if (factor is not None) and (factor != self.snnl_criterion.factor):
            # print(f" model.snnl_criterion.factor {model.snnl_criterion.factor}")
            self.snnl_criterion.factor = factor 
            print(f" model.snnl_criterion.factor set to {factor}")
    
        for self.batch_count, (batch_features, batch_labels, _, batch_compounds, _) in enumerate(data_loader):
 
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            # self.temp_optimizer.zero_grad()
            outputs = self.forward(features=batch_features)
        
            if self.use_snnl:
                train_loss, primary_loss, snn_loss = self.snnl_criterion(
                    model=self,
                    outputs=outputs,
                    features=batch_features,
                    labels=batch_labels,
                    epoch=epoch,
                )
                epoch_loss += train_loss.item()
                epoch_snn_loss += snn_loss.item()
                epoch_primary_loss += primary_loss.item()
            else:
                print(f"Model not using SNNL")
                train_loss = model.criterion(outputs, batch_labels if self.name in ["DNN", "CNN"]  else batch_features)
                epoch_loss += train_loss.item()
            
            # if self.name in ["DNN","CNN"]:
            # train_accuracy = (outputs.argmax(1) == batch_labels).sum().item() / len(batch_labels)
            # epoch_accuracy += train_accuracy
            # else:
            # train_accuracy =  0
            # epoch_accuracy += train_accuracy
            
            train_loss.backward()
            self.optimizer.step()
            
            if self.use_snnl and self.temperature is not None:
                # self.temp_optimizer.step()
                self.optimize_temperature(verbose=True)
            
            if verbose:
                print(f" batch:{batch_count:3d} - ttl loss:  {train_loss:10.6f}  XEntropy: {primary_loss:10.6f}    SNN: {snn_loss*self.snnl_criterion.factor:10.6f}" 
                      f" (loss: {snn_loss:10.6f} * {self.snnl_criterion.factor})   temp: {self.temperature.item():16.12f}   temp.grad: {self.temperature.grad.item():16.12f}")        
        
        #### End of dataloader loop
        # if verbose:
        #     print('-'*100)
        #     print(f" epoch {epoch+1} ended - batch_count: {batch_count}")
        #     print(' Total loss   : ', epoch_loss, epoch_loss / batch_count)
        #     print(' SNN loss     : ', epoch_snn_loss * self.snnl_criterion.factor, (epoch_snn_loss * self.snnl_criterion.factor) / batch_count )
        #     print(' XEntropy_loss: ', epoch_primary_loss, epoch_primary_loss / batch_count)
        #     print(' Accuracy     : ', epoch_accuracy / batch_count)
        #     print('-'*100)
           
        epoch_loss /= self.batch_count
    
        # if model.name in ["DNN", "CNN"]:
        # epoch_accuracy /= batch_count ## len(data_loader)
        # else:
            # epoch_accuracy = 0
            
        # if model.use_snnl:
        epoch_snn_loss = (epoch_snn_loss * self.snnl_criterion.factor) / self.batch_count
        epoch_primary_loss /=  self.batch_count
        
        # if self.name == "DNN" or self.name == "CNN":
        # print(f" epoch_loss: {epoch_loss},  epoch_snn_loss: {epoch_snn_loss}, epoch_primary_loss: {epoch_primary_loss}, accuracy: {epoch_accuracy} ")
        # return (epoch_loss, epoch_snn_loss, epoch_primary_loss), epoch_accuracy
        # else:
        #     print(f" epoch_loss: {epoch_loss},  epoch_snn_loss: {epoch_snn_loss}, epoch_primary_loss: {epoch_primary_loss} ")
        #     a, b =  (epoch_loss, epoch_snn_loss, epoch_primary_loss), 0 
        #     return a
        # else:
        #     # if self.name == "DNN" or self.name == "CNN":
        #     print(f" epoch_loss: {epoch_loss}, accuracy: {epoch_accuracy} ")
        #     a , b = epoch_loss, epoch_accuracy
        #     # else:
        #         # print(f" epoch_loss: {epoch_loss},")
        #         # a =  epoch_loss
                  # return a    
        return (epoch_loss, epoch_snn_loss, epoch_primary_loss), epoch_accuracy
    
    
    def optimize_temperature(self, verbose = False):
        """
        Learns an optimized temperature parameter.
        """
        
        torch.nn.utils.clip_grad_value_(self.temperature, clip_value=1.0)
        self.temperature_gradients.append(self.temperature.grad.item())

        # use the with torch.no_grad(): context manager to prevent PyTorch from tracking the gradients
        # of the parameter, so that you can update it without affecting the training process.
        # without using no_grad() it gives error: 
        # cannot assign 'torch.cuda.FloatTensor' as parameter 'temperature' (torch.nn.Parameter or None expected)
        
        with torch.no_grad():
            self.temperature.copy_(self.temperature - (self.temperatureLR * self.temperature.grad))

        ## original way this was being done in original code
        # updated_temperature = self.temperature - (self.temperatureLR * self.temperature.grad)
        # self.temperature.data = updated_temperature
        
    # if torch.isnan(temperature_gradient):
            # print(f" optimize_temp:  temp_gradients: {temperature_gradient}  temp: {self.temperature.data}   ")
        # else:
            # if verbose:
                # print(f" optimize_temp:  temp_gradients:{unclipped_temperature_gradient:.7f} - {temperature_gradient.item():.7f}  Temp: before: {before_temp}   updated: {updated_temperature.item()}")
    """Feed-forward Neural Network"""

#-------------------------------------------------------------------------------------------------------------------
#  Model DNN class  
#-------------------------------------------------------------------------------------------------------------------

class DNN(Model):
    """
    A feed-forward neural network that optimizes
    softmax cross entropy using Adam optimizer.

    An optional soft nearest neighbor loss
    regularizer can be used with the softmax cross entropy.
    """

    _criterion = torch.nn.CrossEntropyLoss()

    def __init__(
        self,
        units: List or Tuple = [(784, 500), (500, 500), (500, 10)],
        learning_rate: float = 1e-3,
        use_snnl: bool = False,
        factor: float = 100.0,
        temperature: int = None,
        use_annealing: bool = True,
        unsupervised: bool = None,
        use_sum: bool = False,
        stability_epsilon: float = 1e-5,
        # batch_size: int = 0,
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
            mode="classifier",
            criterion=DNN._criterion.to(device),
            device=device,
            use_snnl=use_snnl,
            factor=factor,
            temperature=temperature,
            unsupervised = unsupervised,
            use_annealing=use_annealing,
            use_sum=use_sum,
            # batch_size = batch_size, 
            sample_size = sample_size,
            stability_epsilon=stability_epsilon,
            verbose=verbose,
        )
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=in_features, out_features=out_features)
                for in_features, out_features in units
            ]
        )

        for index, layer in enumerate(self.layers):
            if index < (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif index == (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            else:
                pass

        self.name = "DNN"
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        # self.temp_optimizer = torch.optim.SGD(params = self.temperature, lr=self.temperatureLR, momentum=0.9)
        if not use_snnl:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.train_accuracy = []
        self.to(self.device)
        print(f" Building DNN from NOTEBOOK")
        print(f"    DNN _init()_    -- mode:              {self.mode}")
        print(f"    DNN _init()_    -- unsupervised :     {self.unsupervised}")
        print(f"    DNN _init()_    -- use_snnl :         {self.use_snnl}")
        print(f"    DNN _init()_    -- temperature :      {self.temperature}")
        

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
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = torch.relu(layer(features))
            elif index == len(self.layers) - 1:  
                activations[index] = layer(activations.get(index - 1))
            else:
                activations[index] = torch.relu(layer(activations.get(index - 1)))
        # logits = activations.get(len(activations) - 1)
        return activations

    def fit(
        self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 2
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
        if self.use_snnl:
            self.train_snn_loss = []
            self.train_xent_loss = []

        for epoch in range(epochs):
            if self.use_snnl:
                *epoch_loss, epoch_accuracy = self.epoch_train(data_loader, epoch)
                self.train_loss.append(epoch_loss[0])
                self.train_snn_loss.append(epoch_loss[1])
                self.train_xent_loss.append(epoch_loss[2])
                self.train_accuracy.append(epoch_accuracy)
                if (epoch + 1) % show_every == 0:
                    print(f"epoch {epoch + 1}/{epochs}" \
                          f"\tmean loss = {self.train_loss[-1]:.6f}\t|\tmean acc = {self.train_accuracy[-1]:.6f}" \
                          f"\tXEntropy loss = {self.train_xent_loss[-1]:.6f}\t|\tsnn loss = {self.train_snn_loss[-1]:.6f}"
                    )
                          # )
                    # print(
                    # print(
                        # f"\txent loss = {self.train_xent_loss[-1]:.6f}\t|\tsnn loss = {self.train_snn_loss[-1]:.6f}"
                    # )
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
        outputs = self.forward(features)
        return outputs
        # predictions, classes = torch.max(outputs.data, dim=1)
        # return (predictions, classes) if return_likelihoods else classes


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
        input_shape: int,
        code_dim: int,
        learning_rate: float = 1e-3,
        use_snnl: bool = False,
        factor: float = 100.0,
        temperature: int = None,
        use_annealing: bool = True,

        use_sum: bool = False,
        mode: str = "autoencoding",
        code_units: int = 0,
        stability_epsilon: float = 1e-5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
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
        super().__init__(
            mode=mode,
            criterion=Autoencoder._criterion.to(device),
            device=device,
            use_snnl=use_snnl,
            factor=factor,
            code_units=code_units,
            temperature=temperature,
            use_annealing=use_annealing,
            use_sum=use_sum,
            stability_epsilon=stability_epsilon,
        )
        print(f" Building Autoencoder from NOTEBOOK")
        if mode not in Autoencoder._supported_modes:
            raise ValueError(f"Mode {mode} is not supported.")
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=input_shape, out_features=500),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=500, out_features=500),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=500, out_features=2000),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=2000, out_features=code_dim),
                torch.nn.Sigmoid(),
                torch.nn.Linear(in_features=code_dim, out_features=2000),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=2000, out_features=500),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=500, out_features=500),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=500, out_features=input_shape),
                torch.nn.Sigmoid(),
            ]
        )

        for index, layer in enumerate(self.layers):
            if (index == 6 or index == 14) and isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                pass

        self.name = "Autoencoder"
        self.to(self.device)
        if not use_snnl:
            self.criterion = Autoencoder._criterion.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

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
        if features.ndim > 2:
            features = features.view(features.shape[0], -1)
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = layer(features)
            else:
                activations[index] = layer(activations.get(index - 1))
        reconstruction = activations.get(len(activations) - 1)
        return reconstruction

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
        self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 2
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
                self.train_loss.append(epoch_loss[0])
                self.train_snn_loss.append(epoch_loss[1])
                self.train_recon_loss.append(epoch_loss[2])
                if (epoch + 1) % show_every == 0:
                    print(
                        f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                    )
                    print(
                        f"\trecon loss = {self.train_recon_loss[-1]:.6f}\t|\tsnn loss = {self.train_snn_loss[-1]:.6f}"
                    )
            else:
                self.train_loss.append(epoch_loss)
                if (epoch + 1) % show_every == 0:
                    print(
                        f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                    )

#-------------------------------------------------------------------------------------------------------------------
#  stand_alone epoch_train
#-------------------------------------------------------------------------------------------------------------------

def epoch_train(model, data_loader: torch.utils.data.DataLoader, epoch: int = None, factor: int = None) -> Tuple:
    epoch_loss = 0
    epoch_ttl_loss = 0
    epoch_primary_loss = 0
    epoch_snn_loss = 0
    epoch_accuracy = 0
    
    if (factor is not None) and (factor != self.snnl_criterion.factor):
        # print(f" model.snnl_criterion.factor {model.snnl_criterion.factor}")
        model.snnl_criterion.factor = factor 
        print(f" model.snnl_criterion.factor set to {factor}")

    for batch_count, (batch_features, batch_labels, _, _, _) in enumerate(data_loader):
        
        batch_features = batch_features.to(model.device)
        batch_labels = batch_labels.to(model.device)
        
        model.optimizer.zero_grad()
        
        outputs = model.forward(features=batch_features)
        # break
    
        # if model.use_snnl:
        train_loss, primary_loss, snn_loss = model.snnl_criterion(
            model=model,
            outputs=outputs,
            features=batch_features,
            labels=batch_labels,
            epoch=epoch,
        )
        epoch_loss += train_loss.item()
        epoch_snn_loss += snn_loss.item()
        epoch_primary_loss += primary_loss.item()
        # else:
        #     print(f"Model not using SNNL")
        #     train_loss = model.criterion(outputs, batch_labels if model.name == "DNN" or model.name == "CNN"  else batch_features,)
        #     epoch_loss += train_loss.item()
        
        # if model.name == "DNN" or model.name == "CNN":
        # train_accuracy = (outputs.argmax(1) == batch_labels).sum().item() / len(batch_labels)
        # epoch_accuracy += train_accuracy
        # else:
        train_accuracy =  0
        epoch_accuracy += train_accuracy
        
        train_loss.backward()
        model.optimizer.step()
        
        if model.use_snnl and model.temperature is not None:
            model.optimize_temperature()
        
        print(f" batch:{batch_count:3d} - train : loss    {train_loss:10.6f}   XEntropy loss: {primary_loss:10.6f}    SNN loss: {snn_loss*model.snnl_criterion.factor:10.6f}"
              f" (loss: {snn_loss:10.6f} * factor: {model.snnl_criterion.factor})  "
              f"  temp: {model.temperature.data} snnl_temp: {model.snnl_criterion.temperature.data}")    
    
    #### End of dataloader loop
 
    print(f" loop ended - batch_count: {batch_count}")
    print(' epoch_loss        : ', epoch_loss, epoch_loss / batch_count)
    print(' epoch_snn_loss    : ', epoch_snn_loss * model.snnl_criterion.factor, (epoch_snn_loss * model.snnl_criterion.factor) / batch_count )
    print(' epoch_primary_loss: ', epoch_primary_loss, epoch_primary_loss / batch_count)
    print(' epoch_accuracy    : ', epoch_accuracy / batch_count)
       
    epoch_loss /= batch_count

    # if model.name in ["DNN", "CNN"]:
    epoch_accuracy /= batch_count ## len(data_loader)
    # else:
        # epoch_accuracy = 0
        
    # if model.use_snnl:
    epoch_snn_loss = (epoch_snn_loss * model.snnl_criterion.factor) / batch_count
    epoch_primary_loss /=  batch_count
    
    # if model.name == "DNN" or model.name == "CNN":
    # print(f" epoch_loss: {epoch_loss},  epoch_snn_loss: {epoch_snn_loss}, epoch_primary_loss: {epoch_primary_loss}, accuracy: {epoch_accuracy} ")
    # return (epoch_loss, epoch_snn_loss, epoch_primary_loss), epoch_accuracy
    # else:
    #     print(f" epoch_loss: {epoch_loss},  epoch_snn_loss: {epoch_snn_loss}, epoch_primary_loss: {epoch_primary_loss} ")
    #     a, b =  (epoch_loss, epoch_snn_loss, epoch_primary_loss), 0 
    #     return a
    # else:
    #     # if model.name == "DNN" or model.name == "CNN":
    #     print(f" epoch_loss: {epoch_loss}, accuracy: {epoch_accuracy} ")
    #     a , b = epoch_loss, epoch_accuracy
    #     # else:
    #         # print(f" epoch_loss: {epoch_loss},")
    #         # a =  epoch_loss
              # return a    
    return (epoch_loss, epoch_snn_loss, epoch_primary_loss), epoch_accuracy



#-------------------------------------------------------------------------------------------------------------------
#  CellpaintingDataset
#-------------------------------------------------------------------------------------------------------------------
class CellpaintingDataset(torch.utils.data.IterableDataset ):

    def __init__(self, 
                 train : bool = True,
                 dataset_path : str = None, 
                 batch_size: int = None,
                 sample_size : int = None, 
                 chunksize :int  = None, 
                 conversions : Dict = None, 
                 train_start :int = None, 
                 train_end :int = None,
                 test_start :int = None, 
                 test_end :int = None,
                 names : List = None, 
                 usecols : List = None, 
                 iterator : bool =False, 
                 verbose : bool = False,
                 compounds_per_batch: int = 1,
                 **misc, 
    ):
        # print("Cellpainting __init__ routine", flush=True)
        #Store the filename in object's memory
        self.filename = dataset_path
        self.names = names
        self.dtype = conversions
        self.usecols = usecols
        self.compounds_per_batch = compounds_per_batch
        self.iterator = iterator
        # chunksize should be a mulitple of sample_size
        # self.batch_size = batch_size
        self.sample_size = sample_size
        if chunksize is None:
            self.chunksize = self.sample_size * self.compounds_per_batch
        self.start = train_start if train else test_start
        self.end = train_end if train else test_end
        self.numrows = self.end-self.start
        # self.group_labels = np.arange(self.batch_size * self.sample_size, dtype = np.int64) // self.sample_size
        # print(f" Dataset batch_size: {self.batch_size}" )
        # print(self.group_labels)
        # And that's it, we no longer need to store the contents in the memory

    def preprocess(self, 
                   text):
        ### Do something with data here
        print(f" Running preprocess data \n")
        text_pp = text.lower().strip()
        print(len(text_pp))
        ###
        return text_pp

    def line_mapper(self, 
                    line):
        # print(f" Running line_mapper \n")
        #Splits the line into text and label and applies preprocessing to the text
        # data  = line.to_numpy()
        # text = self.preprocess(text)
        # print(f" compound: {compound.shape}  -  {data}")
        return line

    def __iter__(self):
        #Create an iterator
        # print("Cellpainting __iter__ routine", flush=True)
        self.file_iterator =  pd.read_csv(self.filename, 
                                names = self.names,
                                header = 0,
                                skiprows = self.start,
                                nrows = self.numrows,
                                dtype = self.dtype,
                                usecols = self.usecols,
                                iterator = self.iterator,
                                chunksize = self.chunksize,
                                )
        # df_ps = dd.read_csv(profile_file, header=header, names = names, usecols = usecols, dtype = dtype) 
        # Map each element using the line_mapper
        # self.mapped_itr = map(self.line_mapper, self.file_iterator)
        return self.file_iterator
        
    # def __next__(self):
        # print("Cellpainting __next__ funtion called")
        # super().__next__()

    
    def __len__(self):
        # print("Cellpainting __len__ funtion called")
        return self.numrows 

#-------------------------------------------------------------------------------------------------------------------
#  InfiniteDataLoader
#-------------------------------------------------------------------------------------------------------------------    

class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.

    def __iter__(self):
        # print("InfiniteDataLoader __iter__  routine")
        self.dataset_iterator = super().__iter__()
        return self

    def __next__(self):
        try:
            # print(f"InfiniteDataLoader __next__  routine ")
            batch = next(self.dataset_iterator)
            # print(f"InfiniteDataLoader __next__  routine - next batch is {type(batch)}  {len(batch)} {type(batch[0])}")
        except StopIteration:
            # print("InfiniteDataLoader __next__  routine  -- End of dataset encountered!!!")
            # Dataset exhausted, use a new fresh iterator.
            # self.dataset_iterator = super().__iter__()
            # batch = next(self.dataset_iterator)
            raise StopIteration
        else:
            return batch


def custom_collate_fn(batch):
    batch_numpy = np.concatenate(batch)
    plates  = batch_numpy[:,:4]
    compounds = batch_numpy[:,4]
    cmphash = batch_numpy[:,5:7].astype(np.int64)
    labels = torch.from_numpy(batch_numpy[:,10].astype(np.int64))
    data = torch.from_numpy(batch_numpy[:, 11:].astype(np.float32))
    
    # print(f"  Running custom_collate: {type(batch)} {len(batch)} {type(batch[0])} {len(batch[0])}")
    # print(f"  Batch_numpy.shape:   {batch_numpy.shape}")
    # print(f"  Batch type: {type(batch)}  Length Batch {len(batch)}  type element {type(batch[0])}  length element: {(batch[0].shape)}" )
    # print(f" plates: {plates.shape}  compounds: {compounds.shape}  hash: {cmphash.shape} labels: {labels.shape} data: {data.shape}")
    # print(f" plates: {type(plates)}  compounds: {type(compounds)}  hash: {type(cmphash)} labels: {type(labels)} data: {type(data)}")
    
    return  data,  labels, plates, compounds, cmphash
        
#-------------------------------------------------------------------------------------------------------------------
#  InfiniteDataLoader
#-------------------------------------------------------------------------------------------------------------------     
def parse_args(input = None):
    parser = argparse.ArgumentParser(description="DNN classifier with SNNL")
    group = parser.add_argument_group("Parameters")
    group.add_argument(
        "-s",
        "--seed",
        required=False,
        default=1234,
        type=int,
        help="the random seed value to use, default: [1234]",
    )
    group.add_argument(
        "-m",
        "--model",
        required=False,
        default="baseline",
        type=str,
        help="the model to use, options: [baseline (default) | snnl]",
    )
    group.add_argument(
        "-c",
        "--configuration",
        required=False,
        default="examples/hyperparameters/dnn.json",
        type=str,
        help="the path to the JSON file containing the hyperparameters to use",
    )
    arguments = parser.parse_args(input)
    return arguments


def display_cellpainting_batch(batch_id, batch):
    data, labels, plates, compounds, cmphash,  = batch
    print("-"*80)
    print(f" Batch Id: {batch_id}   {type(batch)}  Rows returned {len(batch[0])} features: {len(data[0])}  ")
    print("-"*80)
    print(f"{len(labels)}")
    for i in range(len(labels)):
        print(f" {i:3d} | {plates[i,0]:9s} | {compounds[i]:12s} | {cmphash[i,0]} | {cmphash[i,1]:2d} | {labels[i]:1d} | {data[i,:3]}")
