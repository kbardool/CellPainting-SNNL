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
    def __init__(
        self,
        mode: str,
        criterion: object,
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
        self.training_history['train'] = defaultdict(list)
        self.training_history['val']   = defaultdict(list)
        # self.temperature_gradients = [] 
        # self.train_snn_loss = []
        # self.train_xent_loss = []
        # self.train_temp_hist = []
        # self.train_temp_grad_hist = []        
        # self.train_accuracy = []
        # self.train_f1 = []
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
            print(f" for {self.mode} support for unsupervised is {self.unsupervised}")
        else:
            self.unsupervised = unsupervised
            
        if self.use_snnl:
            self.temperature = torch.nn.Parameter(data=torch.tensor([temperature]), requires_grad=True)
            self.register_parameter(name="temperature", param=self.temperature)
            # self.temperature = temperature
            self.use_annealing = use_annealing
            self.use_sum = use_sum
            self.snnl_criterion = SNNLoss(
                mode=self.mode,
                # criterion=self.primary_criterion,
                factor=self.snnl_factor,
                temperature=self.temperature,
                use_annealing=self.use_annealing,
                use_sum=self.use_sum,
                code_units=self.code_units,
                embedding_layer = self.embedding_layer,
                sample_size = self.sample_size,
                stability_epsilon=self.stability_epsilon,
                unsupervised=self.unsupervised )
        else:
            self.snnl_criterion = None

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

    
    def epoch_train(self, 
                    data_loader: torch.utils.data.DataLoader, 
                    epoch: int = None, 
                    loss_factor: float = None,
                    snnl_factor: float = None, 
                    verbose: bool = False) -> Tuple:
        epoch_loss = 0
        epoch_ttl_loss = 0
        epoch_primary_loss = 0
        epoch_snn_loss  = 0
        epoch_metrics     =  SimpleNamespace()
        if not self.unsupervised:
            epoch_metrics.accuracy  = 0
            epoch_metrics.f1        = 0
            epoch_metrics.precision = 0
            epoch_metrics.recall    = 0
            epoch_metrics.fbeta     = 0
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
            self.temp_optimizer.zero_grad()
 
            outputs, logits = self.forward(features=batch_features)
                             
            if self.use_snnl:
                # train_loss, primary_loss, snn_loss = self.snnl_criterion(
                snn_loss = self.snnl_criterion(
                    model=self,
                    outputs=logits,
                    features=batch_features,
                    labels=batch_labels,
                    # temperature = self.temperature,
                    epoch=epoch,
                )
 
            primary_loss = self.compute_primary_loss(batch_labels,logits, batch_features)
            
            primary_loss = torch.mul(self.loss_factor, primary_loss)        
            snn_loss =  torch.mul(self.snnl_factor, snn_loss)
            train_loss = torch.add(primary_loss, snn_loss)
            
            epoch_snn_loss += snn_loss.item()
            epoch_primary_loss += primary_loss.item() 
            epoch_loss += train_loss.item() 

            # if self.mode != "moe":
                # self.train_loss = torch.add(self.primary_loss, torch.mul(self.snnl_criterion.factor, self.snn_loss))
            # else:
            #     return self.primary_loss, self.snn_loss
            
            if self.name in ["DNN","CNN"] and not self.unsupervised:
                self.classification_metrics(batch_labels,logits, epoch_metrics)

                        
            # train_loss.backward()
            primary_loss.backward()
            
            if self.use_snnl:
                snn_loss.backward()
                # print(f" temp gradient: {self.temperature.grad.item():.6e}    new temp: {self.temperature.item():.6e}")
                
            self.optimizer.step()
            if self.use_snnl:
                self.temp_optimizer.step()
                # print(f" training batch {self.batch_count} primary_loss: {primary_loss:.7f}  epoch: {epoch_primary_loss:.7f}   snn_loss: {snn_loss:.7f}   epoch: {epoch_snn_loss:.7f} "                                
                #       f" temp gradient: {self.temperature.grad.item():.6e}    new temp: {self.temperature.item():.6e}")
            # llg = self.layers[-2].weight.grad 
            # print(f" training batch {self.batch_count:4d}  -  primary_loss: {primary_loss:.7f}     ll grad min:{llg.min():.7e}     max: {llg.max():.7e}    sum: {llg.sum():.7f}   std: {llg.std():.7f}")
             
            self.training_history['train']['temp_grads'].append(self.temperature.grad.item())  
            if self.monitor_grads_layer is not None:
                self.training_history['train']['layer_grads'].append( self.layers[self.monitor_grads_layer].weight.grad.sum().item() + 
                                                                      self.layers[self.monitor_grads_layer].bias.grad.sum().item())   
            
            # self.temperature_gradients.append(self.temperature.grad.item())
            # if self.use_snnl and self.temperature is not None:
            #     # self.temp_optimizer.step()
            #     self.optimize_temperature(verbose=True)
            
            # if verbose:
                # print(f" batch:{batch_count:3d} - ttl loss:  {train_loss:10.6f}  XEntropy: {primary_loss:10.6f}    SNN: {snn_loss*self.snnl_criterion.factor:10.6f}" 
                      # f" (loss: {snn_loss:10.6f} * {self.snnl_criterion.factor})   temp: {self.temperature.item():16.12f}   temp.grad: {self.temperature.grad.item():16.12f}")        
        
        ## End of dataloader loop
 
        self.training_history['train']['temp_hist'].append(self.temperature.item()) 
        self.training_history['train']['temp_grad_hist'].append(self.temperature.grad.item()) 
        total_batches = self.batch_count +1 
    
        if self.name in ["DNN", "CNN"] and not self.unsupervised:
            self.add_metrics_to_history('train', epoch_metrics, total_batches)

            
        epoch_loss /= total_batches
        if self.use_snnl:
            epoch_snn_loss /= total_batches
            epoch_primary_loss /=  total_batches
        
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
        
        return (epoch_loss, epoch_snn_loss, epoch_primary_loss) 


    def epoch_validate(self, 
                    data_loader: torch.utils.data.DataLoader, 
                    epoch: int = None, 
                    verbose: bool = False) -> Tuple:
        epoch_loss = 0
        epoch_ttl_loss = 0
        epoch_primary_loss = 0
        epoch_snn_loss = 0
        epoch_metrics = SimpleNamespace()
        if not self.unsupervised:
            epoch_metrics.accuracy  = 0
            epoch_metrics.f1        = 0
            epoch_metrics.precision = 0
            epoch_metrics.recall    = 0
            epoch_metrics.fbeta     = 0
            epoch_metrics.roc_auc   = 0        
        
        self.eval()
        for batch_count, (batch_features, batch_labels, _, _, _) in enumerate(data_loader): 
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            outputs, logits = self.forward(features=batch_features)

            if self.use_snnl:
                # train_loss, primary_loss, snn_loss = self.snnl_criterion(
                snn_loss = self.snnl_criterion(
                    model=self,
                    outputs=logits,
                    features=batch_features,
                    labels=batch_labels,
                    epoch=epoch,
                )
                                
            primary_loss = self.compute_primary_loss(batch_labels,logits,batch_features)
            
            primary_loss = torch.mul(self.loss_factor, primary_loss)        
            snn_loss =  torch.mul(self.snnl_factor, snn_loss)          
            train_loss = torch.add(primary_loss, snn_loss)
            
            epoch_snn_loss += snn_loss.item()
            epoch_primary_loss += primary_loss.item()
            epoch_loss += train_loss.item()
            
            # print(f"validation batch loss:  primary_loss: {primary_loss:.7f}  epoch: {epoch_primary_loss:.7f}   snn_loss: {snn_loss:.7f}   epoch: {epoch_snn_loss:.7f} ")      
            
            if self.name in ["DNN","CNN"] and not self.unsupervised:
                self.classification_metrics(batch_labels,logits, epoch_metrics, 'validation')
        
        self.training_history['val']['temp_hist'].append(self.temperature.item()) 
        self.training_history['val']['temp_grad_hist'].append(self.temperature.grad.item())                 
        total_batches = batch_count +1 
    
        if self.name in ["DNN", "CNN"] and not self.unsupervised:
            self.add_metrics_to_history('val', epoch_metrics, total_batches)

        epoch_loss /= total_batches
        epoch_primary_loss /=  total_batches
        if self.use_snnl:
            epoch_snn_loss /= total_batches

        self.train()
        return (epoch_loss, epoch_snn_loss, epoch_primary_loss) 

    
    def classification_metrics(self, batch_labels, logits, metrics, ttl = 'training'): 
        y_true = batch_labels.detach().cpu().numpy()
        y_prob = logits.squeeze().detach().cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(np.int32)
 
        accuracy  = accuracy_score(y_true, y_pred)
        roc_auc   = roc_auc_score(y_true, y_prob)
        precision, recall, fbeta, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        # print(f" {ttl} : Acc: {accuracy:.5f}  F1: {f1:.5f}   Prec: {precision:.5f}   Recall: {recall:.5f}   fbeta: {fbeta:.5f}  pos lables: {y_true.sum()}  preds: {y_pred.sum()}  matching: {(y_pred == y_true).sum()}" )
        metrics.accuracy  += accuracy
        metrics.f1        += fbeta
        metrics.precision += precision
        metrics.recall    += recall
        metrics.fbeta     += fbeta        
        metrics.roc_auc   += roc_auc

    def add_metrics_to_history(self,key , metrics, total_batches):
            # print(f"total epoch accuracy: {epoch_accuracy:.5f}   batch_count: {self.batch_count+1}  epoch_accuracy: {epoch_accuracy / (self.batch_count+1):.6f}")            
            self.training_history[key]['accuracy'].append(metrics.accuracy / total_batches )
            self.training_history[key]['f1'].append(metrics.f1 / total_batches)
            self.training_history[key]['precision'].append(metrics.precision / total_batches)
            self.training_history[key]['recall'].append(metrics.recall / total_batches)
            self.training_history[key]['fbeta'].append(metrics.fbeta / total_batches)
            self.training_history[key]['roc_auc'].append(metrics.roc_auc/ total_batches)

    
    def compute_primary_loss(self, labels, outputs, features ):
        if self.unsupervised:
             if self.name in ["DNN","CNN"] :
                loss = torch.tensor(0, requires_grad = True, dtype=torch.float32, device = self.device)
             else:
                loss = self.primary_criterion(outputs.squeeze(), features)
        else:
            loss = self.primary_criterion(outputs.squeeze(), labels)
            # self.primary_loss = self.primary_criterion(outputs.squeeze, features if self.unsupervised else labels)
 
        return loss

    def model_history(self,key, losses):
        self.training_history[key]['time'].append(datetime.now().strftime('%H:%M:%S'))
        self.training_history[key]['loss'].append(losses[0])
        self.training_history[key]['snn_loss'].append(losses[1])
        self.training_history[key]['xent_loss'].append(losses[2])
 
        
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

# # -------------------------------------------------------------------------------------------------------------------
# #  Model DNN class  
# # -------------------------------------------------------------------------------------------------------------------

# class DNN(Model):
#     """
#     Feed-forward Neural Network
    
#     A feed-forward neural network that optimizes
#     softmax cross entropy using Adam optimizer.

#     An optional soft nearest neighbor loss
#     regularizer can be used with the softmax cross entropy.
#     """

#     # _criterion = torch.nn.CrossEntropyLoss()

#     def __init__(
#         self,
#         mode="classifier",
#         units: List or Tuple = [(784, 500), (500, 500), (500, 10)],
#         activations: List = ["relu", "relu", "softmax"],
#         learning_rate: float = 1e-3,
#         use_snnl: bool = False,
#         loss_factor: float = 1.0,
#         snnl_factor: float = 100.0,
#         temperature: int = None,
#         temperatureLR: float = None,
#         adam_weight_decay: float = 0.0,
#         SGD_weight_decay: float = 0.0,
#         use_annealing: bool = True,
#         unsupervised: bool = None,
#         use_sum: bool = False,
#         stability_epsilon: float = 1e-3,

#         sample_size: int = 1,        
#         criterion=torch.nn.CrossEntropyLoss(),
#         device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#         verbose: bool = False,
#     ):
#         """
#         Constructs a feed-forward neural network classifier.

#         Parameters
#         ----------
#         units: list or tuple
#             An iterable that consists of the number of units in each hidden layer.
#         learning_rate: float
#             The learning rate to use for optimization.
#         use_snnl: bool
#             Whether to use soft nearest neighbor loss or not.
#         factor: float
#             The balance between SNNL and the primary loss.
#             A positive factor implies SNNL minimization,
#             while a negative factor implies SNNL maximization.
#         temperature: int
#             The SNNL temperature.
#         use_annealing: bool
#             Whether to use annealing temperature or not.
#         use_sum: bool
#             Use summation of SNNL across hidden layers if True,
#             otherwise get the minimum SNNL.
#         stability_epsilon: float
#             A constant for helping SNNL computation stability
#         device: torch.device
#             The device to use for model computations.
#         """
#         super().__init__(
#             mode=mode,
#             criterion=criterion.to(device),
#             device=device,
#             use_snnl=use_snnl,
#             loss_factor=loss_factor,
#             snnl_factor=snnl_factor,
#             temperature=temperature,
#             temperatureLR=temperatureLR,
#             unsupervised = unsupervised,
#             use_annealing=use_annealing,
#             use_sum=use_sum,
#             # batch_size = batch_size, 
#             sample_size = sample_size,
#             stability_epsilon=stability_epsilon,
#             verbose=verbose,
#         )
#         print(f" Building DNN from NOTEBOOK")
#         self.primary_weight = torch.tensor([2])

#         self.layers = torch.nn.ModuleList()
#         self.layer_type = []
#         for idx, (type,in_features, out_features) in enumerate(units):
#             type = type.lower()
#             if type =='linear':
#                 self.layers.append( torch.nn.Linear(in_features=in_features, out_features=out_features))
#                 self.layer_type.append('linear')
#             elif type == 'dropout':
#                 self.layers.append( torch.nn.dropout(p=dropout_p))
#                 self.layer_type.append('dropout')
#         self.layer_activations = activations
#         print(f" layer_types      : {self.layer_type}")
#         print(f" layer_activations: {self.layer_activations}")
#         # self.layers = torch.nn.ModuleList(
#         #     [
#         #         torch.nn.Linear(in_features=in_features, out_features=out_features)
#         #         for in_features, out_features in units
#         #     ]
#         # )
        
#         for index, layer in enumerate(self.layers):
#             if index < (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
#                 torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
#             elif index == (len(self.layers) - 1) and isinstance(layer, torch.nn.Linear):
#                 torch.nn.init.xavier_uniform_(layer.weight)
#             else:
#                 pass

#         self.name = "DNN"
 
#         parameter_dict = defaultdict(list)
#         for name, parm in self.named_parameters():
#             if name in ['temperature']:
#                 parameter_dict['SGD'].append(parm)
#             else:
#                 parameter_dict['Adam'].append(parm)
#         self.optimizer = torch.optim.Adam(params=parameter_dict['Adam'], lr=learning_rate, weight_decay = adam_weight_decay)
#         self.temp_optimizer = torch.optim.SGD(params=parameter_dict['SGD'], lr=self.temperatureLR, momentum=0.9, weight_decay = SGD_weight_decay)
        
#         # self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        
#         if not use_snnl:
#             # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
#             self.criterion = criterion

#         self.to(self.device)
        
#         print(f"    DNN _init()_    -- mode:              {self.mode}")
#         print(f"    DNN _init()_    -- unsupervised :     {self.unsupervised}")
#         print(f"    DNN _init()_    -- use_snnl :         {self.use_snnl}")
#         if not use_snnl:
#             print(f"    DNN _init()_    -- Crtierion  :       {self.criterion}")
#         else:
#             print(f"    DNN _init()_    -- Crtierion  :       {self.snnl_criterion}")
#         print(f"    DNN _init()_    -- temperature :      {temperature}")
#         print(f"    DNN _init()_    -- temperature LR:    {temperatureLR}")
 

#     def forward_old(self, features: torch.Tensor) -> torch.Tensor:
#         """
#         Defines the forward pass by the model.

#         Parameter
#         ---------
#         features: torch.Tensor
#             The input features.

#         Returns
#         -------
#         logits: torch.Tensor
#             The model output.
#         """
#         # features = features.view(features.shape[0], -1)
#         activations = {}
#         num_layers = len(self.layers)
#         for index, layer in enumerate(self.layers):
#             if index == 0:
#                 activations[index] = torch.relu(layer(features))
#             elif index == num_layers - 1 :     ## last layer
#                 activations[index] = layer(activations.get(index - 1))
#             else:                              ## middle layers
#                 activations[index] = torch.relu(layer(activations.get(index - 1)))
        
#         logits = torch.sigmoid(activations.get(len(activations) - 1))
        
#         return activations, logits 



#     def forward(self, features: torch.Tensor) -> torch.Tensor:
#         """
#         Defines the forward pass by the model.

#         Parameter
#         ---------
#         features: torch.Tensor
#             The input features.

#         Returns
#         -------
#         logits: torch.Tensor
#             The model output.
#         """
#         # features = features.view(features.shape[0], -1)
#         activations = {}
#         num_layers = len(self.layers)
#         for index, layer in enumerate(self.layers):
#             if index == 0:
#                 activations[index] = layer(features)
#             else:                              ## middle & last layer
#                 activations[index] = layer(activations[index - 1])
                
#         assert (len(self.layers) == len(activations)) , f" lengths of self.layers {len(self.layers)} and activations {len(activations)} do not match"
        
#         for index, non_linearity in enumerate(self.layer_activations):
#             non_linearity = non_linearity.lower()
#             if non_linearity == 'relu':
#                 activations[index] = torch.relu(activations[index])
#             elif non_linearity == 'sigmoid':
#                 activations[index] = torch.sigmoid(activations[index])
#             elif non_linearity == 'none':
#                 pass
#             else :
#                 raise Exception(f"Unrecognized type found in activations {non_linearity}")
                
#         logits = activations [len(activations) - 1]
#         return activations, logits 
    
#     def fit(
#         self, data_loader: torch.utils.data.DataLoader, epochs: int, show_every: int = 1
#     ) -> None:
#         """
#         Trains the DNN model.

#         Parameters
#         ----------
#         data_loader: torch.utils.dataloader.DataLoader
#             The data loader object that consists of the data pipeline.
#         epochs: int
#             The number of epochs to train the model.
#         show_every: int
#             The interval in terms of epoch on displaying training progress.
#         """
#         header = True
#         for epoch in range(epochs):
#             if self.use_snnl:
#                 train_loss = self.epoch_train( train_loader, epoch, factor = snnl_factor, verbose = False)
#                 self.model_history('train', train_loss)

#                 val_loss  = self.epoch_validate( val_loader, epoch, factor = snnl_factor, verbose = False)
#                 self.model_history('val', val_loss)

#                 display_epoch_metrics(self, epoch, epochs, header)
#                 header = False            
                
#             else:
#                 epoch_loss, epoch_accuracy = self.epoch_train(data_loader)
#                 self.train_loss.append(epoch_loss)
#                 self.train_accuracy.append(epoch_accuracy)
#                 if (epoch + 1) % show_every == 0:
#                     print(f"epoch {epoch + 1}/{epochs}")
#                     print(
#                         f"\tmean loss = {self.train_loss[-1]:.6f}\t|\tmean acc = {self.train_accuracy[-1]:.6f}"
#                     )

#     def predict(
#         self, features: torch.Tensor, return_likelihoods: bool = False
#     ) -> torch.Tensor:
#         """
#         Returns model classifications

#         Parameters
#         ----------
#         features: torch.Tensor
#             The input features to classify.
#         return_likelihoods: bool
#             Whether to return classes with likelihoods or not.

#         Returns
#         -------
#         predictions: torch.Tensor
#             The class likelihood output by the model.
#         classes: torch.Tensor
#             The class prediction by the model.
#         """
#         outputs, logits = self.forward(features)
#         return outputs, logits
#         # predictions, classes = torch.max(outputs.data, dim=1)
#         # return (predictions, classes) if return_likelihoods else classes



