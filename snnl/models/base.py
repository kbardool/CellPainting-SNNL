from typing import Dict, Tuple

# import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"
import logging
import numpy as np
import pandas as pd
import torch
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from types import SimpleNamespace
from datetime import datetime
from pt_datasets import create_dataloader
from typing import Dict, List, Tuple
from collections import defaultdict
import sklearn.metrics as skm
import torcheval.metrics.functional as tev
from snnl.losses import SNNLoss
logger = logging.getLogger(__name__) 
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
        device: torch.device = None,
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
        
        self.sample_size = sample_size
        self.primary_criterion = criterion
        self.best_metric = 0
        self.best_epoch  = 0
        self.training_history = dict()
        self.training_history['gen'] = {'trn_best_metric' : 0, 'trn_best_metric_ep' : 0, 'trn_best_loss': np.inf, 'trn_best_loss_ep' : 0 ,
                                        'val_best_metric' : 0, 'val_best_metric_ep' : 0, 'val_best_loss': np.inf, 'val_best_loss_ep' : 0 }
        self.training_history['trn'] = defaultdict(list)
        self.training_history['val'] = defaultdict(list)
        
        if unsupervised is None:
            self.unsupervised = self._unsupervised_supported_modes.get(self.mode)
        else:
            self.unsupervised = unsupervised

        self.calculate_metrics =  self.regression_metrics if self.unsupervised else self.classification_metrics
            
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
                device = self.device,
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
            
        if verbose:
            print('\n'+'-'*60)
            print(f" Building Base Model from NOTEBOOK")
            print('-'*60)
            print(f"    Model_init()_    -- mode:              {self.mode}")
            print(f"    Model_init()_    -- Unsupervised :     {self.unsupervised}")
            print(f"    Model_init()_    -- Support for unsupervised training  in '{self.mode}' mode is {self.unsupervised}")
            print(f"    Model_init()_    -- Criterion:         {self.primary_criterion}")
            print(f"    Model_init()_    -- use_snnl :         {self.use_snnl}")
            print(f"    Model_init()_    -- temperature :      {self.temperature}")
            print(f"    Model_init()_    -- temperature LR:    {self.temperatureLR}")
        
            

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
                    DEBUG_COUNT: int = 0, 
                    verbose: bool = False) -> Tuple:
        self.train()
        self.epoch = epoch
        epoch_losses, epoch_metrics = self.init_losses_and_metrics()
        
        if (snnl_factor is not None) and (snnl_factor != self.snnl_factor):
            # print(f" model.snnl_criterion.factor {model.snnl_criterion.factor}")
            self.snnl_.factor = snnl_factor 
            self.snnl_criterion.factor = snnl_factor 
            print(f" model.snnl_criterion.factor set to {snnl_factor}")
            
        if (loss_factor is not None) and (loss_factor != self.loss_factor):
            # print(f" model.loss_factor {model.loss_factor}")
            self.loss_factor = loss_factor 
            print(f" model.loss_factor set to {loss_factor}")    
        
        if self.use_annealing:
            # temp_before = self.temperature.item()
            with torch.no_grad():
                self.temperature.copy_(1.0 / ((1.0 + epoch) ** 0.55))
            # print(f" {epoch} - anneal temp  -  before: {temp_before:10.6f}     new_temp: {self.temperature.item():10.6f} ")
 
                
        for self.batch_count, (batch_features, batch_labels, _, _, _) in enumerate(data_loader):
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            _, logits = self.forward(features=batch_features)
            
            # temp_before = self.temperature.item()
            # snnl_temp_before = self.snnl_criterion.temperature.item()

            snn_loss = self.compute_snnl_loss(batch_labels, logits, batch_features)

            # temp_after = self.temperature.item()
            # snnl_temp_after = self.snnl_criterion.temperature.item()

            primary_loss = self.compute_primary_loss(batch_labels, logits, batch_features)
           
            snn_loss =  torch.mul(self.snnl_factor, snn_loss)
            primary_loss = torch.mul(self.loss_factor, primary_loss)        
            total_loss   = torch.add(primary_loss, snn_loss)
            
            epoch_losses.snn_loss += snn_loss.item()
            epoch_losses.primary_loss += primary_loss.item() 
            epoch_losses.ttl_loss += total_loss.item() 
             
            self.calculate_metrics(batch_features, logits, epoch_metrics)

            if self.use_single_loss:  
                self.optimizers['prim'].zero_grad()
                total_loss.backward()
                self.optimizers['prim'].step()
            else:        
                # if self.batch_count < DEBUG_COUNT:
                #     self.display_gradients('After forward pass, before zero grad')
 
                for k,v in self.optimizers.items():
                    # self.optimizers[k].zero_grad()
                    v.zero_grad()
                    # if self.batch_count < DEBUG_COUNT:
                    #     self.display_gradients(f" Optimizer {k} after zero_grad()")
                    #     self.display_values(f" Optimizer {k} after zero_grad()")

                primary_loss.backward( retain_graph = self.use_temp_optimizer)
                if 'temp' in self.optimizers: 
                    snn_loss.backward()     
                # else:   ## self.use_annealing
                #     pass

                # for k,v in self.optimizers.items():
                    # if self.batch_count < DEBUG_COUNT:
                    #     self.display_gradients(f" Optimizer {k} after {v}.backward()")
                    #     self.display_values(f" Optimizer {k} after {v}.backward()")
                    
                for k,v in self.optimizers.items():
                    v.step()
                    # if self.batch_count < DEBUG_COUNT:
                    #     self.display_gradients(f" Optimizer {k} after step()")
                    #     self.display_values(f" Optimizer {k} after step()")
                        
                # self.temperature = torch.clamp(self.temperature,1.0e-6,None)
            if self.monitor_grads_layer is not None:
                self.training_history['trn']['layer_grads'].append( self.layers[self.monitor_grads_layer].weight.grad.sum().item() + 
                                                                      self.layers[self.monitor_grads_layer].bias.grad.sum().item())     
            # temp_final = self.temperature.item()
            # snnl_temp_final = self.snnl_criterion.temperature.item()
            # temp_grad = 0.0 if self.temperature.grad is None else self.temperature.grad.item
            # print(f" {self.epoch}/{self.batch_count} - temp- bef: {temp_before:.6f}  aft: {temp_after:.6f}   "
                #   f" fin: {temp_final:.6f}      delta: {temp_final - temp_after:10.6f}     grad: {temp_grad:10.6f}   "
                #   f" grad * LR : {temp_grad*self.temperatureLR:10.6f}")
                #   f" snnl-temp : {snnl_temp_before:10.6f}  {snnl_temp_after:10.6f}  {snnl_temp_final:10.6f} ")
 
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
        else:
            epoch_metrics.R2_score  /= total_batches
            epoch_metrics.R2_score_tev  /= total_batches
            
        self.update_training_history('trn', epoch, epoch_losses, epoch_metrics)              
 
        return epoch_losses 


    def epoch_validate(self, 
                    data_loader: torch.utils.data.DataLoader, 
                    epoch: int = None, 
                    verbose: bool = False) -> Tuple:
        self.new_best = False
        self.epoch = epoch
        epoch_losses, epoch_metrics = self.init_losses_and_metrics()        
        
        self.eval()
        
        ## begin dataloader loop
        for self.batch_count, (batch_features, batch_labels, _, _, _) in enumerate(data_loader): 
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            _, logits = self.forward(features=batch_features)
                
            snn_loss = self.compute_snnl_loss(batch_labels, logits, batch_features)
            primary_loss = self.compute_primary_loss(batch_labels,logits,batch_features)

            snn_loss = torch.mul(self.snnl_factor, snn_loss)          
            primary_loss = torch.mul(self.loss_factor, primary_loss)        
            total_loss= torch.add(primary_loss, snn_loss)
            
            epoch_losses.snn_loss += snn_loss.item()
            epoch_losses.primary_loss += primary_loss.item()
            epoch_losses.ttl_loss += total_loss.item()
            
            # if not self.unsupervised:
            #     epoch_metrics = self.classification_metrics(batch_labels, logits, epoch_metrics)
            # else:
            #     epoch_metrics = self.regression_metrics(batch_features, logits, epoch_metrics)                
            self.calculate_metrics(batch_features, logits, epoch_metrics)
        ## end of dataloader loop
                
        total_batches = self.batch_count +1     
        epoch_losses.ttl_loss /= total_batches
        epoch_losses.primary_loss /=  total_batches
        epoch_losses.snn_loss /= total_batches
        
        if not self.unsupervised:
            epoch_metrics.accuracy  /= total_batches
            epoch_metrics.f1        /= total_batches
            epoch_metrics.precision /= total_batches
            epoch_metrics.recall    /= total_batches
            epoch_metrics.roc_auc   /= total_batches
        else:
            epoch_metrics.R2_score  /= total_batches
            epoch_metrics.R2_score_tev  /= total_batches
            
        self.update_training_history('val', epoch, epoch_losses, epoch_metrics)
        self.update_best_metric()
        return epoch_losses

    def init_losses_and_metrics(self):
        losses = SimpleNamespace()
        metrics = SimpleNamespace()
        losses.ttl_loss = 0
        losses.snn_loss = 0
        losses.primary_loss = 0
        
        if self.unsupervised:
            metrics.R2_score  = 0 
            metrics.R2_score_tev  = 0 
        else:
            metrics.accuracy  = 0
            metrics.f1        = 0
            metrics.precision = 0
            metrics.recall    = 0
            metrics.roc_auc   = 0
        return losses, metrics 
            
    def update_best_metric(self):
        if self.best_metric <  self.training_history['gen'][f'val_best_metric']:
            self.best_metric =  self.training_history['gen'][f'val_best_metric']
            self.best_epoch = self.training_history['gen'][f'val_best_metric_ep']
            self.new_best = True
        
    def update_training_history(self, key, epoch, losses, metrics):
 
        assert  key in ['trn', 'val'], f" invalid history type {key} - must be {{'trn', 'val'}} "
        
        self.training_history[key][f"{key}_time"].append(datetime.now().strftime('%H:%M:%S'))
        self.training_history[key][f"{key}_ttl_loss"].append(losses.ttl_loss)
        self.training_history[key][f"{key}_prim_loss"].append(losses.primary_loss)
        self.training_history[key][f"{key}_snn_loss"].append(losses.snn_loss)
        if losses.ttl_loss < self.training_history['gen'][f'{key}_best_loss']:
            self.training_history['gen'][f'{key}_best_loss'] = losses.ttl_loss
            self.training_history['gen'][f'{key}_best_loss_ep'] = epoch
            
        if not self.unsupervised:
            self.training_history[key][f"{key}_accuracy"].append(metrics.accuracy)
            self.training_history[key][f"{key}_f1"].append(metrics.f1)
            self.training_history[key][f"{key}_precision"].append(metrics.precision)
            self.training_history[key][f"{key}_recall"].append(metrics.recall)
            self.training_history[key][f"{key}_roc_auc"].append(metrics.roc_auc)
        else:
            self.training_history[key][f"{key}_R2_score"].append(metrics.R2_score)
            self.training_history[key][f"{key}_R2_score_tev"].append(metrics.R2_score_tev )
            if metrics.R2_score > self.training_history['gen'][f'{key}_best_metric']:
                self.training_history['gen'][f'{key}_best_metric'] = metrics.R2_score
                self.training_history['gen'][f'{key}_best_metric_ep'] = epoch
            
        if key == 'trn':        
            self.training_history['trn']['trn_lr'].append(self.optimizers['prim'].param_groups[0]['lr'])
            if self.use_snnl:
                self.training_history['trn']['temp_hist'].append(self.temperature.item()) 
                temp_grad = 0.0 if self.temperature.grad is None else self.temperature.grad.item()
                self.training_history['trn']['temp_grad_hist'].append(temp_grad)   
                if self.use_temp_optimizer:
                    self.training_history['trn']['temp_lr'].append(self.optimizers['temp'].param_groups[0]['lr'])
                else:
                    self.training_history['trn']['temp_lr'].append(0.0)
    
    def compute_snnl_loss(self, labels, outputs, features ):
        if self.use_snnl:
            snn_loss = self.snnl_criterion(
                model=self,
                outputs=outputs,
                features=features,
                labels=labels,
                epoch = self.epoch,
                batch = self.batch_count)
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


    def regression_metrics(self, y_true, y_pred, metrics): 
        r2_score_skm = skm.r2_score(y_pred = y_pred.detach().cpu().numpy(), y_true = y_true.detach().cpu().numpy())
        r2_score_tev = torch.tensor(0)
        if torch.isinf(r2_score_tev):
            print(f" r2_score_tev is inf ")
        metrics.R2_score +=  r2_score_skm.item()
        metrics.R2_score_tev  += r2_score_tev.item()
        return metrics
        # return metrics, torch.tensor(batch_r2_score, requires_grad = True, dtype=torch.float32, device = self.device)

        
    def classification_metrics(self, batch_labels, logits, metrics): 
        y_true = batch_labels.detach().cpu().numpy()
        y_prob = logits.squeeze().detach().cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(np.int32)
        
        precision, recall, f1, _ = skm.precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        # print(f" {ttl} : Acc: {accuracy:.5f}  F1: {f1:.5f}   Prec: {precision:.5f}   Recall: {recall:.5f}  "
                # f"  pos lables: {y_true.sum()}  preds: {y_pred.sum()}  matching: {(y_pred == y_true).sum()}" )
        metrics.accuracy  += skm.accuracy_score(y_true, y_pred)
        metrics.roc_auc   += skm.roc_auc_score(y_true, y_prob)
        metrics.f1        += f1
        metrics.precision += precision
        metrics.recall    += recall
        return metrics

                   
    def scheduling_step(self, loss):
        
        if 'prim' in self.schedulers:
            self.schedulers['prim'].step(loss.primary_loss)
            if  self.training_history['trn']['trn_lr'][-1] != self.optimizers['prim'].param_groups[0]['lr']:
                logger.info(f" Main learning rate reduced to {self.schedulers['prim']._last_lr}")
                
        if 'temp' in self.schedulers:
            self.schedulers['temp'].step(loss.snn_loss)
            if  self.training_history['trn']['temp_lr'][-1] != self.optimizers['temp'].param_groups[0]['lr']:
                logger.info(f" Temperature optimizer learning rate reduced to {self.schedulers['temp']._last_lr}")    

    
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
        #     if verbose:
        #         print(f" optimize_temp:  temp_gradients:{unclipped_temperature_gradient:.7f} - {temperature_gradient.item():.7f}"
        #               f"  Temp: before: {before_temp}   updated: {updated_temperature.item()}")

    
    def display_gradients(self, msg):
        print()
        print(f" {self.epoch}/ {self.batch_count} - {msg}")
        for p in self.named_parameters():
            if p[1].grad is not None:
                print(f"  Name: {p[0]:30s}  Grad:  Min: {p[1].grad.min():15.12f}  Max: {p[1].grad.max():15.12f}  Sum: {p[1].grad.sum():15.12f} "
                                       f" - Parm:  Min: {p[1].min():15.12f}  Max: {p[1].max():15.12f}  Sum: {p[1].sum():18.12f}")
            else:
                print(f"  Name: {p[0]:30s}  Gradient: {p[1].requires_grad}   Min: {'None':12s}  Max: {'None':12s}  Sum:  {'None':12s} ")
                
    def display_values(self, msg):
        print()
        print(f" {self.epoch}/ {self.batch_count} - {msg}")
        for k in ['temperature', 'snnl_criterion.temperature','layers.0.weight', 'layers.0.bias','layers.2.weight',
                  'layers.4.weight','layers.4.bias','layers.5.weight','layers.5.bias','layers.7.weight','layers.9.weight','layers.9.bias',]:
            if self.state_dict()[k].ndim > 1:
                print(f" {k+' - '+str(self.state_dict()[k].shape):45s} - {self.state_dict()[k].sum():20.15f} - {self.state_dict()[k][:3,:3].reshape((-1)).data}")
            else:
                print(f" {k+' - '+str(self.state_dict()[k].shape):45s} - {self.state_dict()[k].sum():20.15f} - {self.state_dict()[k][:9].data}")
        print()