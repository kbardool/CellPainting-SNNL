from typing import Dict, Tuple

# import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"

import numpy as np
import pandas as pd
import torch
import argparse
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from types import SimpleNamespace
from datetime import datetime
from pt_datasets import create_dataloader
from typing import Dict, List, Tuple
from collections import defaultdict
from snnl.utils import binary_accuracy, binary_f1_score
from sklearn.metrics import accuracy_score, f1_score, roc_curve, precision_recall_curve, precision_recall_fscore_support, \
                            roc_auc_score
from dev_snnloss import SNNLoss
#-------------------------------------------------------------------------------------------------------------------
#  Model Base class  
#-------------------------------------------------------------------------------------------------------------------

class Model(torch.nn.Module):

    _unsupervised_supported_modes = {
        "classifier"  : True,
        # "classifier"  : False,
        "resnet"      : False,
        "autoencoding": True,
        "latent_code" : True,
        "sae"         : True,
        "custom"      : False,
        "moe"         : False,
    }    
    _ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau
    def __init__(
        self,
        mode: str,
        criterion: object = None,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        use_snnl: bool = False,
        loss_factor: float = 1.0,
        snnl_factor: float = 100.0,
        temperature: float = 100.0,
        temperatureLR: float = None,
        use_annealing: bool = False,
        use_sum: bool = False,
        unsupervised: bool = None,
        code_units: int = 0,
        embedding_layer: int = None,
        stability_epsilon: float = 1e-5,
        # batch_size: int = 0,
        sample_size: int = 1,
        verbose: bool = False,
    ):
        """
        Constructs the Base Model.

        Parameters
        ----------
        mode: str
            The mode in which the soft nearest neighbor loss
            will be used. Default: [classifier]
        criterion: object
            The primary loss to use.
        #     Default: [torch.nn.CrossEntropyLoss()]
        snnl_factor: float
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
 
        """        
        super().__init__()
        mode = mode.lower()

        self.mode = mode
        self.device = device
        self.train_loss = []
        self.use_snnl = use_snnl
        self.loss_factor = loss_factor
        self.snnl_factor = snnl_factor
        self.code_units = code_units
        self.embedding_layer = embedding_layer
        self.stability_epsilon = stability_epsilon
        self.verbose = verbose
        # self.criterion = criterion
        self.sample_size = sample_size
        self.temperatureLR = temperatureLR
        self.primary_criterion = criterion
        self.training_history = dict()
        self.training_history['trn'] = defaultdict(list)
        self.training_history['val'] = defaultdict(list)

        print('\n'+'-'*60)
        print(f" Building Base Model from NOTEBOOK")
        print('-'*60)
        print(f"    Model_init()_    -- mode:              {mode}")
        print(f"    Model_init()_    -- unsupervised :     {unsupervised}")
        print(f"    Model_init()_    -- Criterion:         {self.primary_criterion}")
        print(f"    Model_init()_    -- use_snnl :         {use_snnl}")
        print(f"    Model_init()_    -- temperature :      {temperature}")
        print(f"    Model_init()_    -- temperature LR:    {temperatureLR}")
        
        if unsupervised is None:
            self.unsupervised = self._unsupervised_supported_modes.get(self.mode)
            print(f"    Model_init()_    -- {self.name} - support for unsupervised in {self.mode} mode is {self.unsupervised}")
        else:
            self.unsupervised = unsupervised
        if (self.mode == "latent_code"):
            if (embedding_layer is None):
                raise ValueError("[Embedding_layer]  must be specified when self.mode = 'latent_code'." )
            elif  (type(embedding_layer) == int) and (embedding_layer <=0):
                raise ValueError("[Embedding_layer]  must be specified when self.mode = 'latent_code'." )
               
        if self.use_snnl:
            self.temperature = torch.nn.Parameter(data=torch.tensor([temperature]), requires_grad=True)
            self.register_parameter(name="temperature", param=self.temperature)
            # self.temperature = temperature
            self.use_annealing = use_annealing
            self.use_sum = use_sum
            self.snnl_criterion = SNNLoss(
                mode=self.mode,
                temperature=self.temperature,
                use_annealing=self.use_annealing,
                use_sum=self.use_sum,
                code_units=self.code_units,
                sample_size = self.sample_size,
                stability_epsilon=self.stability_epsilon,
                unsupervised=self.unsupervised )
        else:
            self.snnl_criterion = None
            self.temperature = 0
            

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

    
    def sanity_check(self,
                     data_loader: torch.utils.data.DataLoader,
                     epochs: int = 10,
                     show_every: int = 2):
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


    
    def epoch_train(self, 
                    data_loader: torch.utils.data.DataLoader, 
                    epoch: int = None, 
                    loss_factor: float = None,
                    snnl_factor: float = None, 
                    verbose: bool = False) -> Tuple:
        self.training = True
        # epoch_loss = 0
        # epoch_ttl_loss = 0
        # epoch_primary_loss = 0
        # epoch_snn_loss  = 0
        epoch_losses = SimpleNamespace()
        epoch_losses.ttl_loss = 0
        epoch_losses.snn_loss = 0
        epoch_losses.primary_loss = 0
        
        epoch_metrics = SimpleNamespace()
        if not self.unsupervised:
            epoch_metrics.accuracy  = 0
            epoch_metrics.f1        = 0
            epoch_metrics.precision = 0
            epoch_metrics.recall    = 0
            epoch_metrics.roc_auc   = 0
        
        if (snnl_factor is not None) and (snnl_factor != self.snnl_factor):
            # print(f" model.snnl_criterion.factor {model.snnl_criterion.factor}")
            self.snnl_.factor = snnl_factor 
            self.snnl_criterion.factor = snnl_factor 
            print(f" model.snnl_criterion.factor set to {snnl_factor}")
            
        if (loss_factor is not None) and (loss_factor != self.loss_factor):
            # print(f" model.loss_factor {model.loss_factor}")
            self.loss_factor = loss_factor 
            print(f" model.loss_factor set to {loss_factor}")    
            
        for self.batch_count, (batch_features, batch_labels, _, _, _) in enumerate(data_loader):
 
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            if self.use_snnl:
                self.temp_optimizer.zero_grad()
 
            _, logits = self.forward(features=batch_features)
 
            snn_loss = self.compute_snnl_loss(batch_labels, logits, batch_features)
            snn_loss =  torch.mul(self.snnl_factor, snn_loss)
            primary_loss = self.compute_primary_loss(batch_labels, logits, batch_features)
            primary_loss = torch.mul(self.loss_factor, primary_loss)        
            total_loss = torch.add(primary_loss, snn_loss)
            
            epoch_losses.snn_loss += snn_loss.item()
            epoch_losses.primary_loss += primary_loss.item() 
            epoch_losses.ttl_loss += total_loss.item() 
             
            if  not self.unsupervised:
                epoch_metrics = self.classification_metrics(batch_labels,logits, epoch_metrics)
            
            # train_loss.backward()
            primary_loss.backward()                            
            if self.use_snnl:
                snn_loss.backward()
                # print(f" temp gradient: {self.temperature.grad.item():.6e}    new temp: {self.temperature.item():.6e}")
                
            self.optimizer.step()
            if self.use_snnl:
                self.temp_optimizer.step()
                torch.nn.utils.clip_grad_value_(self.temperature, clip_value=1.0)
                # self.training_history['trn']['temp_grads'].append(self.temperature.grad.item())  
                # print(f" training batch {self.batch_count} primary_loss: {primary_loss:.7f}  epoch: {epoch_primary_loss:.7f}   snn_loss: {snn_loss:.7f}   epoch: {epoch_snn_loss:.7f} "                                
                #       f" temp gradient: {self.temperature.grad.item():.6e}    new temp: {self.temperature.item():.6e}")

            if self.monitor_grads_layer is not None:
                self.training_history['trn']['layer_grads'].append( self.layers[self.monitor_grads_layer].weight.grad.sum().item() + 
                                                                      self.layers[self.monitor_grads_layer].bias.grad.sum().item())     
        
        ## End of dataloader loop
 
        total_batches = self.batch_count +1 
 
        epoch_losses.ttl_loss /= total_batches
        epoch_losses.primary_loss /=  total_batches
        epoch_losses.snn_loss /= total_batches
        if  not self.unsupervised:
            epoch_metrics.accuracy  /= total_batches
            epoch_metrics.f1        /= total_batches
            epoch_metrics.precision /= total_batches
            epoch_metrics.recall    /= total_batches
            epoch_metrics.roc_auc   /= total_batches
        self.update_training_history('trn', epoch_losses, epoch_metrics)              
        self.training = False        
        return epoch_losses 


    def epoch_validate(self, 
                    data_loader: torch.utils.data.DataLoader, 
                    epoch: int = None, 
                    verbose: bool = False) -> Tuple:

        epoch_losses = SimpleNamespace()
        epoch_losses.ttl_loss = 0
        epoch_losses.snn_loss = 0
        epoch_losses.primary_loss = 0
                
        epoch_metrics = SimpleNamespace()
        if not self.unsupervised:
            epoch_metrics.accuracy  = 0
            epoch_metrics.f1        = 0
            epoch_metrics.precision = 0
            epoch_metrics.recall    = 0
            epoch_metrics.roc_auc   = 0        
        
        self.eval()
        
        ## begin dataloader loop
        for batch_count, (batch_features, batch_labels, _, _, _) in enumerate(data_loader): 
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            _, logits = self.forward(features=batch_features)
                
            snn_loss = self.compute_snnl_loss(batch_labels, logits, batch_features)
            snn_loss =  torch.mul(self.snnl_factor, snn_loss)          
            primary_loss = self.compute_primary_loss(batch_labels,logits,batch_features)
            primary_loss = torch.mul(self.loss_factor, primary_loss)        
            total_loss = torch.add(primary_loss, snn_loss)
            
            epoch_losses.snn_loss += snn_loss.item()
            epoch_losses.primary_loss += primary_loss.item()
            epoch_losses.ttl_loss += total_loss.item()
            if not self.unsupervised:
                epoch_metrics = self.classification_metrics(batch_labels,logits, epoch_metrics)
        ## end of dataloader loop
                
        total_batches = batch_count +1     
        epoch_losses.ttl_loss /= total_batches
        epoch_losses.primary_loss /=  total_batches
        epoch_losses.snn_loss /= total_batches
        if not self.unsupervised:
            epoch_metrics.accuracy  /= total_batches
            epoch_metrics.f1        /= total_batches
            epoch_metrics.precision /= total_batches
            epoch_metrics.recall    /= total_batches
            epoch_metrics.roc_auc   /= total_batches
            
        self.update_training_history('val', epoch_losses, epoch_metrics)
        self.train()
        self.validation = False
        return epoch_losses

    
    def classification_metrics(self, batch_labels, logits, metrics): 
        y_true = batch_labels.detach().cpu().numpy()
        y_prob = logits.squeeze().detach().cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(np.int32)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        # print(f" {ttl} : Acc: {accuracy:.5f}  F1: {f1:.5f}   Prec: {precision:.5f}   Recall: {recall:.5f}  "
                # f"  pos lables: {y_true.sum()}  preds: {y_pred.sum()}  matching: {(y_pred == y_true).sum()}" )
        metrics.accuracy  += accuracy_score(y_true, y_pred)
        metrics.roc_auc   += roc_auc_score(y_true, y_prob)
        metrics.f1        += f1
        metrics.precision += precision
        metrics.recall    += recall
        return metrics

    def update_training_history(self, key, losses, metrics):
        assert  key in ['trn', 'val'], f" invalid history type {key} - must be {{'trn', 'val'}} "
        self.training_history[key][f"{key}_time"].append(datetime.now().strftime('%H:%M:%S'))
        self.training_history[key][f"{key}_ttl_loss"].append(losses.ttl_loss)
        self.training_history[key][f"{key}_prim_loss"].append(losses.primary_loss)
        self.training_history[key][f"{key}_snn_loss"].append(losses.snn_loss)
        if not self.unsupervised:
            self.training_history[key][f"{key}_accuracy"].append(metrics.accuracy)
            self.training_history[key][f"{key}_f1"].append(metrics.f1)
            self.training_history[key][f"{key}_precision"].append(metrics.precision)
            self.training_history[key][f"{key}_recall"].append(metrics.recall)
            self.training_history[key][f"{key}_roc_auc"].append(metrics.roc_auc)
        if key == 'trn':        
            self.training_history['trn']['trn_lr'].append(self.optimizer.param_groups[0]['lr'])
            if self.use_snnl:
                self.training_history['trn']['temp_hist'].append(self.temperature.item()) 
                self.training_history['trn']['temp_grad_hist'].append(self.temperature.grad.item())    
            if self.use_temp_scheduler:
                self.training_history['trn']['temp_lr'].append(self.temp_optimizer.param_groups[0]['lr'])
    
    def compute_snnl_loss(self, labels, outputs, features ):
        if self.use_snnl:
            snn_loss = self.snnl_criterion(
                model=self,
                outputs=outputs,
                features=features,
                labels=labels)
        else:
            snn_loss = 0.0
        return snn_loss

    def compute_primary_loss(self, labels, outputs, features ):
        # print(f" compute primary loss - {self.primary_criterion} unsupervised: {self.unsupervised}")
        # print(f" {outputs.squeeze().shape} \n {outputs.squeeze()}")
        # print(f"{labels.shape}  \n {labels}")
        if self.unsupervised:
             if self.name in ["DNN","CNN"] :
                loss = torch.tensor(0, requires_grad = True, dtype=torch.float32, device = self.device)
             else:
                loss = self.primary_criterion(outputs.squeeze(), features)
        else:
            loss = self.primary_criterion(outputs.squeeze(), labels)
            # self.primary_loss = self.primary_criterion(outputs.squeeze, features if self.unsupervised else labels)
        return loss

    def scheduling(self, loss):
        if self.use_scheduler:
            self.scheduler.step(loss.primary_loss)
            if  self.training_history['trn']['trn_lr'][-1] != self.optimizer.param_groups[0]['lr']:
                print(f" Optimizer learning rate reduced to {self.scheduler._last_lr}")
                
        if self.use_temp_scheduler:
            self.scheduler.step(loss.snn_loss)
            if  self.training_history['trn']['temp_lr'][-1] != self.temp_optimizer.param_groups[0]['lr']:
                print(f" Temperature optimizer learning rate reduced to {self.temp_scheduler._last_lr}")
                
        
    def optimize_temperature(self, verbose = False):
        """
        Learns an optimized temperature parameter.
        """
        torch.nn.utils.clip_grad_value_(self.temperature, clip_value=1.0)

        # use the with torch.no_grad(): context manager to prevent PyTorch from tracking the gradients
        # of the parameter, so that you can update it without affecting the training process.
        # without using no_grad() it gives error: 
        # cannot assign 'torch.cuda.FloatTensor' as parameter 'temperature' (torch.nn.Parameter or None expected)
        # 
        # original way this was being done (original code)
        #     updated_temperature = self.temperature - (self.temperatureLR * self.temperature.grad)
        #     self.temperature.data = updated_temperature
        
        with torch.no_grad():
            self.temperature.copy_(self.temperature - (self.temperatureLR * self.temperature.grad))

    # if torch.isnan(temperature_gradient):
            # print(f" optimize_temp:  temp_gradients: {temperature_gradient}  temp: {self.temperature.data}   ")
        # else:
            # if verbose:
                # print(f" optimize_temp:  temp_gradients:{unclipped_temperature_gradient:.7f} - {temperature_gradient.item():.7f}  Temp: before: {before_temp}   updated: {updated_temperature.item()}")

    
    def epoch_train_old(self, data_loader: torch.utils.data.DataLoader, epoch: int = None) -> Tuple:
        """
        Trains a model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epoch: int
            The current epoch training index.
     
        Returns
        -------
        epoch_loss: float
            The epoch loss.
        epoch_snn_loss: float
            The soft nearest neighbor loss for an epoch.
        epoch_xent_loss: float
            The cross entropy loss for an epoch.
        epoch_accuracy: float
            The epoch accuracy.
        """
        if self.use_snnl:
            epoch_primary_loss = 0
            epoch_snn_loss = 0
        if self.name == "DNN" or self.name == "CNN":
            epoch_accuracy = 0
        epoch_loss = 0
        batch_count = 0
        for batch_features, batch_labels in data_loader:
            if self.name in ["Autoencoder", "DNN"]:
                batch_features = batch_features.view(batch_features.shape[0], -1)
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
        
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
                train_loss = self.criterion(
                    outputs,
                    batch_labels
                    if self.name == "DNN" or self.name == "CNN"
                    else batch_features,
                )
                epoch_loss += train_loss.item()
            
            if self.name == "DNN" or self.name == "CNN":
                train_accuracy = (outputs.argmax(1) == batch_labels).sum().item() / len(
                    batch_labels
                )
                epoch_accuracy += train_accuracy
            
            train_loss.backward()
            self.optimizer.step()
            
            if self.use_snnl and self.temperature is not None:
                self.optimize_temperature()

            batch_count +=1
        
        epoch_loss /= batch_count ## len(data_loader)
        
        if self.name in ["DNN", "CNN"]:
            epoch_accuracy /= batch_count ## len(data_loader)
        if self.use_snnl:
            epoch_snn_loss /= batch_count ## len(data_loader)
            epoch_primary_loss /= batch_count  ## len(data_loader)
            if self.name == "DNN" or self.name == "CNN":
                print(f" SNNLoss() - epoch_loss {epoch_loss:.6f}, epoch_snn_loss, {epoch_snn_loss:.6f},  epoch_primary_loss, {epoch_primary_loss:.6f}, epoch_accuracy,  {epoch_accuracy:.6f}")
                if self.verbose:
                    print(f" SNNLoss() - epoch_loss {epoch_loss}, epoch_snn_loss, {epoch_snn_loss},  epoch_primary_loss, {epoch_primary_loss}, epoch_accuracy,  {epoch_accuracy}")
                return epoch_loss, epoch_snn_loss, epoch_primary_loss, epoch_accuracy
            else:
                if self.verbose:
                    print(f"  SNNLoss() - epoch_loss {epoch_loss}, epoch_snn_loss, {epoch_snn_loss},  epoch_primary_loss, {epoch_primary_loss}")
                return epoch_loss, epoch_snn_loss, epoch_primary_loss
        else:
            if self.name == "DNN" or self.name == "CNN":
                if self.verbose:
                    print(f"  SNNLoss() - epoch_loss {epoch_loss}, epoch_snn_loss, {epoch_snn_loss}")
                return epoch_loss, epoch_accuracy
            else:
                if self.verbose:
                    print(f"  SNNLoss() - epoch_loss {epoch_loss}")
                return epoch_loss