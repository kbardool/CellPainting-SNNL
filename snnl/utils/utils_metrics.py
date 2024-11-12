"""Utility functions / Metrics """
import os
import sys
from datetime import datetime
from collections import defaultdict
import logging
import random
from typing import List, Tuple
from types import SimpleNamespace
import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt
import scipy.stats as sps 
import sklearn.metrics as skm 
from scipy.spatial.distance import pdist, squareform, euclidean
# from .utils import display_dist_metrics
logger = logging.getLogger(__name__) 

    
# -------------------------------------------------------------------------------------------------------------------
#  Metric routines
# -------------------------------------------------------------------------------------------------------------------
def accuracy(y_true, y_pred) -> float:
    """
    Returns the classification accuracy of the model.
    """
    correct = (y_pred == y_true).sum().item()
    accuracy = correct / len(y_true)
    accuracy *= 100.0
    return accuracy


def binary_accuracy(y_true, y_prob):
    assert (y_true.ndim == 1 and y_true.size == y_prob.size), f"binary accuracy:  y_true: {y_true.ndim}   {y_true.shape}  y_prob: {y_prob.ndim}   {y_prob.shape}   "
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum() / y_true.size


def binary_f1_score(y_true, y_prob):
    assert (y_true.ndim == 1 and y_true.size == y_prob.size), f"binary f1 score:  y_true: {y_true.ndim}   {y_true.shape}  y_prob: {y_prob.ndim}   {y_prob.shape}   "
    y_prob = y_prob > 0.5
    return skm.f1_score(y_true, y_prob)


def compute_regression_metrics(mo):
    rm = dict()
    rm['epoch'] = mo.epoch
    rm['R2_score']   = skm.r2_score(y_true = mo.y_true, y_pred = mo.y_pred)
    rm['mse_score']  = skm.mean_squared_error(y_true = mo.y_true, y_pred = mo.y_pred)
    rm['rmse_score'] = skm.root_mean_squared_error(y_true = mo.y_true, y_pred = mo.y_pred)
    rm['mae_score']  = skm.mean_absolute_error(y_true = mo.y_true, y_pred = mo.y_pred)
    # pearson_corr, pearson_p = sps.pearsonr(mo.y_true, mo.y_pred)
    return rm


def compute_classification_metrics(mo):
    """
    mo : model outpust from 'run_model_on_test_data'
    """
    cm = dict()
    print(f" Compute Regression metrics for epoch {mo.epochs}")
    cm['epoch'] = mo.epoch
    cm['metrics'] = mo.metrics
    cm['accuracy'] = skm.accuracy_score(mo.labels, mo.y_pred)
    cm['roc_auc']  = skm.roc_auc_score(mo.labels, mo.logits)
    cm['precision'], cm['recall'], cm['f1'], _ = skm.precision_recall_fscore_support(mo.labels, mo.y_pred, average='binary', zero_division = 0)
    cm['cls_report'] = skm.classification_report(mo.labels, mo.y_pred)
    (mo.labels == mo.y_pred).sum()

    # cm.test_accuracy = binary_accuracy(y_true=mo.labels, y_prob=mo.logits)
    # cm.test_f1 = binary_f1_score(y_true=mo.labels, y_prob=mo.logits)
    return cm


def display_classification_metrics(cm):
    print(f" metrics at epoch {cm['epoch']:^4d}")
    print('-'*22)
    print(f" F1 Score:  {cm['f1']:.7f}")
    print(f" Accuracy:  {cm['accuracy']*100:.2f}%")
    print(f" Precision: {cm['precision']*100:.2f}%")
    print(f" Recall:    {cm['recall']:.7f}")
    print(f" ROC_AUC:   {cm['roc_auc']:.7f}")
    print()
    print(cm.cls_report)


def display_regr_metrics(rm):
    print(f" metrics at epoch {rm['epoch']:^4d}")
    print('-'*22)
    print(f"RMSE Score : {rm['rmse_score']:9.6f}")
    print(f" MSE Score : {rm['mse_score']:9.6f}")
    print(f" MAE Score : {rm['mae_score']:9.6f}")
    print(f"  R2 Score : {rm['R2_score']:9.6f} ")

# -------------------------------------------------------------------------------------------------------------------
#  Run Model on Test Dataloader
# -------------------------------------------------------------------------------------------------------------------
def run_model_on_test_data(model, data_loader, embedding_layer, verbose = False):
    """
    embedding layer: layer that contains embedding (for encoding models)
    """
    out = SimpleNamespace()
    out.labels = np.empty((0))
    out.logits = np.empty((0))
    out.compounds = np.empty((0))
    out.tpsa = np.empty((0))
    out.tpsa_rest = np.empty((0,2))
    out.embeddings = {}

    for k, layer in enumerate(model.layers):
        if hasattr(layer,"out_features"):
            out.embeddings[k] = np.empty((0,layer.out_features))
        else:
            out.embeddings[k] = np.empty((0,model.layers[k-1].out_features))
        if verbose:
            print(f" layer {k:2d}: - {layer} {out.embeddings[k].shape}")

    out.y_true = np.empty((0,model.layers[0].in_features))

    if hasattr(model.layers[-1],"out_features"):
        out.y_pred = np.empty((0,model.layers[-1].out_features))
        if verbose:
            print(f" y_pred picked up from layer {k} of {len(model.layers)} layers")

    elif hasattr(model.layers[-2],"out_features"):
        out.y_pred = np.empty((0,model.layers[-2].out_features))
        if verbose:
            print(f" y_pred picked up from layer {k-1} of {len(model.layers)} layers")
    else:
        raise ValueError("last two layers are not linear")

    ###- Pass Test Dataset through network
    for idx, (batch_features, batch_labels, batch_wellinfo, batch_compound_ids, batch_hashbin, batch_tpsa) in enumerate(data_loader):
        #    (batch_features, batch_labels, plate_well, compound, hash) 

        batch_features = batch_features.to(model.device)
        batch_labels = batch_labels.to(model.device)
        output_activations, reconstruction = model.forward(batch_features)

        for k,v in output_activations.items():
            out.embeddings[k] = np.concatenate((out.embeddings[k], v.detach().cpu().numpy()))
        out.compounds = np.concatenate((out.compounds, batch_compound_ids))
        out.y_true = np.concatenate((out.y_true, batch_features.detach().cpu().numpy()))
        out.y_pred = np.concatenate((out.y_pred, reconstruction.detach().cpu().numpy()))
        out.labels = np.concatenate((out.labels, batch_labels.detach().cpu().numpy()))
        out.tpsa   = np.concatenate((out.tpsa,  batch_tpsa[:,0] ))
        out.tpsa_rest = np.concatenate((out.tpsa_rest,  batch_tpsa[:,1:] ))
        if verbose:
            print(f"   output :  {idx:2d} - Labels:{out.labels.shape[0]:5d}   y_true:{out.y_true.shape}   y_pred:{out.y_pred.shape} ")
    ###- end

    out.labels = np.array(out.labels.astype(np.int32))
    out.comp_labels = np.arange(out.labels.shape[0])//3 
    out.latent_embedding = out.embeddings[embedding_layer]
    # logits = np.concatenate((logits, out_logits.detach().numpy()[:,0]))
    if verbose:
        print(f" out.latent_embedding shape: {out.latent_embedding.shape}")
        print(f" out.y_true    :      shape: {out.y_true.shape}")
        print(f" out.y_pred    :      shape: {out.y_pred.shape}")
        print(f" out.compounds :      shape: {out.compounds.shape}")
        print(f" out.labels    :      shape: {out.labels.shape} - Pos Labels {out.labels.sum()}")
        print(f" out.tpsa      :      shape: {out.tpsa.shape} - {out.tpsa[:5]}")
        print(f" out.tpsa_rest :      shape: {out.tpsa_rest.shape} - {out.tpsa_rest[0,:]}")
        print(f" out.comp_labels:     shape: {out.comp_labels.shape} - {out.comp_labels[:25]}")
        for k,v in out.embeddings.items():
            print(f" out.embeddings[{k:2d}]  :   shape: {v.shape}  {' <---- embedding layer' if k == embedding_layer else ''}")

    return out


def get_latent_representation(model, data_loader, embedding_layer, verbose = False):
    """
    embedding layer: layer that contains embedding (for encoding models)
    """
    print(f" embedding layer: {embedding_layer} - model.layer: in: {model.layers[embedding_layer].in_features} out: {model.layers[embedding_layer].out_features}")

    # ##- Pass Test Dataset through network
    for idx, (batch_features, batch_labels, batch_wellinfo, batch_compounds, batch_hashbin, batch_tpsa_info) in enumerate(data_loader):
        # print(f" batch_features : {batch_features.shape}")
        # print(f" batch_labels   : {batch_labels.shape}  ")
        # print(f" batch_wellinfo : {batch_wellinfo.shape}")
        # print(f" batch_compounds: {batch_compounds.shape}")
        # print(f" batch_hashbin  : {batch_hashbin.shape}")
        # print(f" latent_code    : {latent_code.shape}")

        latent_code = model.compute_latent_code(batch_features.to(model.device))
        output_batch = np.hstack((batch_wellinfo, batch_compounds, batch_hashbin, batch_tpsa_info, batch_labels, latent_code.detach().cpu().numpy()))

        output = output_batch if idx == 0 else np.vstack((output,output_batch))

        print(output_batch.shape, output.shape)

    # if verbose:
    #     print(f" out.latent_embedding shape: {out.embeddings.shape}")
    #     print(f" out.labels    :      shape: {out.labels.shape} - Pos Labels {out.labels.sum()}")
    #     print(f" out.wellinfo  :      shape: {out.wellinfo.shape}")
    #     print(f" out.compounds :      shape: {out.compounds.shape}")
    #     print(f" out.hashbin   :      shape: {out.hashbin.shape}")
    return output


def pairwise_cosine_distance(features: torch.Tensor) -> torch.Tensor:
    """
    Returns the pairwise cosine distance between two copies
    of the features matrix.

    Parameter
    ---------
    features: torch.Tensor                The input features.

    Returns
    -------
    distance_matrix: torch.Tensor         The pairwise cosine distance matrix.

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
    normalized_a = torch.nn.functional.normalize(features, dim=1, p=2)
    return 1.0 - torch.matmul(normalized_a, normalized_a.T)

    # a, b = features.clone(), features.clone()
    # normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
    # normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
    # normalized_b = torch.conj(normalized_b).T
    # product = torch.matmul(normalized_a, normalized_b.T)
    # distance_matrix = torch.sub(torch.tensor(1.0), product)
    # return torch.sub(torch.tensor(1.0), product)
    # return distance_matrix


def pairwise_euclidean_distance(features: torch.Tensor ) -> torch.Tensor:
    """
    Returns the pairwise Euclidean distance between two copies
    of the features matrix.

    Parameter
    ---------
    features: torch.Tensor               The input features.

    Returns
    -------
    distance_matrix: torch.Tensor        The pairwise Euclidean distance matrix.

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

    # distance_matrix = torch.Tensor(squareform(pdist(features)))
    # a, b = features.clone(), features.clone()
    # normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
    # normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
    # # normalized_b = torch.conj(normalized_b).T
    # product = torch.matmul(normalized_a, normalized_b.T)
    # distance_matrix = torch.sub(torch.tensor(1.0), product)
    return torch.sqrt(((features[:, :, None] - features[:, :, None].T) ** 2).sum(1))


def get_sample_indices(dataloader, num_samples = 10,  seed = 1236, verbose= True):
    num_inputs  = len(dataloader) // dataloader.dataset.sample_size
    np.random.seed(seed)

    sampled_compounds = np.sort(np.random.randint(0,num_inputs-1, num_samples))
    # sampled_compounds = np.array(sorted(sample_inds))
    sampled_compounds *= dataloader.dataset.sample_size
    if verbose:
        print(f" Sampled indices sorted - shape:   {sampled_compounds.shape} - {sampled_compounds}")
    return sampled_compounds


def get_sample_data(model_data, compound_ids,  spc = 3, verbose = False):
    '''
    spc: samples per compound
    '''
    indicies = [(compound_ids + i) for i in range(spc)]
    sorted_indicies = np.sort(np.concatenate(indicies))

    samp_input  = model_data.y_true[[sorted_indicies],:].squeeze()
    samp_embed  = model_data.latent_embedding[[sorted_indicies],:].squeeze()
    samp_output = model_data.y_pred[[sorted_indicies],:].squeeze()
    samp_cmpnd  = model_data.compounds[sorted_indicies].squeeze()

    if verbose:
        print(f" Sampled compounds      - shape:   {compound_ids.shape} - {compound_ids}")
        print(f" Compound indices       - shape:   {len(indicies)} - {indicies}")
        for i in indicies:
            print(f"\t\t i: {i}")
        print(f" Sampled indicies sorted - shape:   {sorted_indicies.shape} - {sorted_indicies}")
        print()
        print(f" Sample Input features  - shape:   {samp_input.shape}")
        print(f" Sample Latent features - shape:   {samp_embed.shape}")
        print(f" Sample Output features - shape:   {samp_output.shape}")
        print(f" Sample Compounds       - shape:   {samp_cmpnd.shape}")

    return [samp_input, samp_embed, samp_output, samp_cmpnd]


def distance_metrics_sample_set(model_data, num_samples = 10, cps = 3, seed = 1236, display = True):
    """
    selects a random set of samples that have been passed through the model
    prepares the necessary datasets for distance metric computation 

    cps: Compounds per Sample

    """
    print(f" model_data y_true.shape: {model_data.y_true.shape}")
    num_inputs  = model_data.y_true.shape[0] // cps

    np.random.seed(seed)

    sample = np.random.randint(0,num_inputs-1, num_samples)
    sample = np.array(sorted(sample))
    sample *= cps

    sample_indices = [ (sample + i) for i in range(cps)]
    print(f" Sample indices - len :   {len(sample_indices)}")
    for i in sample_indices:
        print(f" i: {i}")

    sorted_sample = np.concatenate(sample_indices)
    sorted_sample.sort()
    print(f" Sampled indices sorted - shape:   {sorted_sample.shape} - {sorted_sample}")

    samp_cmpnd  = model_data.compounds[sorted_sample]
    samp_input  = model_data.y_true[[sorted_sample],:].squeeze()
    samp_embed  = model_data.latent_embedding[[sorted_sample],:].squeeze()
    samp_output = model_data.y_pred[[sorted_sample],:].squeeze()

    if display:
        print(f" Number of unique compounds    :   {num_inputs}")
        print(f" Sampled compounds      - shape:   {sample.shape} - {sample}")
        print(f" Compound indices       - shape:   {sorted_sample.shape} - {sorted_sample}")
        print(f" Sample Input features  - shape:   {samp_input.shape}")
        print(f" Sample Latent features - shape:   {samp_embed.shape}")
        print(f" Sample Output features - shape:   {samp_output.shape}")

    # emb_cos_distance = squareform(pdist(samp_embed , "cosine"))
    # out_cos_distance = squareform(pdist(samp_output , "cosine"))
    # inp_euc_distance = squareform(pdist(samp_input))
    # emb_euc_distance = squareform(pdist(samp_embed))
    # out_euc_distance = squareform(pdist(samp_output))

    samp_act = [samp_input, samp_embed, samp_output, samp_cmpnd]
    return samp_act


# def distance_metric_1(measurement, cps = 3, epochs = 0):
#     CPS = cps
#     titles = [f'INPUT features - Euclidean distances - {epochs} epochs',
#               f'EMBEDDED features - Euclidean distances - {epochs} epochs',
#               f'RECONSTURCTED features -  Euclidean distances - {epochs} epochs']
#     sub_titles = ["ALL group distances", "INTRA-group (same compound) distances", "INTER-group (diff compound) distances"]
#     k1_list = ['all_grp', 'same_grp', 'diff_grp']
#     k2_list = ['_mean',  '_median', '_stddev', '_min', '_max' ]
#     num_samples = measurement.shape[0]//CPS
#     grp_level = {0 : np.zeros((num_samples,num_samples))}
#     dm = dict()
#     non_diag_elmnts = CPS *(CPS-1)
#     for i in range(0, num_samples*CPS, CPS):
#         for j in range(0, num_samples*CPS, CPS):
#             if i == j:
#                 # print(f" compound sample {1+i//3:4d}   --  sum of dist errs: {measurement[i:i+3, j:j+3].sum():.4f}")
#                 grp_level[0][i//CPS,j//CPS] = measurement[i:i+CPS, j:j+CPS].sum() / non_diag_elmnts  ### 6.0 needs to be parameterized 
#             else:
#                 grp_level[0][i//3,j//3] = measurement[i:i+CPS, j:j+CPS].mean()

#     grp_level[1] = grp_level[0].diagonal()   ## same group
#     grp_level[2] = grp_level[0][np.triu_indices(num_samples,k=1)] ## other_group

#     # same_group  = grp_level[0].diagonal()   ## same group
#     # other_group = grp_level[0][np.triu_indices(num_samples,k=1)]


#     for idx,(subtitle, k1) in enumerate(zip(sub_titles, k1_list)):
#         dm[k1 + k2_list[0]] = grp_level[idx].sum() / grp_level[idx].size
#         dm[k1 + k2_list[1]] = np.median(grp_level[idx])
#         dm[k1 + k2_list[2]] = np.std(grp_level[idx]) 
#         dm[k1 + k2_list[3]] = grp_level[idx].min()
#         dm[k1 + k2_list[4]] = grp_level[idx].max()
#         # print(grp_level[idx])
#         # print()
#         # print(dm[k1+'_mean'])
#         # print()
#         # print(dm[k1+'_median'])
#         # print()
#         # print(dm[k1+'_stddev'])
#         # print()
#         print(f"{idx} {subtitle:39s} :\t mean {dm[k1+'_mean']:8.4f}    median: {dm[k1+'_median']:8.4f}    stddev: {dm[k1+'_stddev'] :8.4f}"
#               f"    min: {dm[k1+'_min'] :8.4f}    max: {dm[k1+'_max'] :8.4f}")
#     print()
#     return dm


def compute_distance_metrics(sample_inputs, epochs = 0, metric ='euclidian',
                             display = False,
                             verbose = False):
    '''
    Input
    ------
    sample_inputs:      List of [ sample X_features, sample embedded representation, sample output (reconstruction)]

    same_comp distances:     s1    s2    s3
                        s1  a11   a12   a13     aij == aji
                        s2  a21   a22   a23 
                        s3  a31   a32   a33


    '''
    assert metric in ['euclidian', 'cosine', 'correlation'], "metric must be one of ['euclidian', 'cosine', 'correlation']"

    if display:
        print(f" Compute Distance {metric} distance metrics for epoch {epochs}") 

    titles = [f'INPUT features - {metric} distances - {epochs} epochs',
              f'EMBEDDED features - {metric} distances - {epochs} epochs',
              f'RECONSTURCTED features -  {metric} distances - {epochs} epochs']
    sub_titles = ["ALL group distances", "INTRA-group (same compound) dist ", "INTER-group (diff compound) dist "]
    k0_list = ['inp', 'emb', 'out']
    k1_list = ['all', 'same', 'diff']
    k2_list = ['_min', '_max', '_avg', '_std' ]
    compounds = sample_inputs[-1]
    grp_level = dict()
    smpl_level = dict()
    dm = dict()
    smpl_mtrcs = dict()

    # Following is done for input representation, embedding rep, and output representation

    for idx0, (k0, sample_input) in enumerate(zip(k0_list, sample_inputs)):
        # 1. Create pair-wise distance matrix for all samples (each compound has 3 samples)
        pairwise_distance = squareform(pdist(sample_input, metric))

        # print(sample_input.shape, pairwise_distance.shape)
        num_samples = pairwise_distance.shape[0]//3

        # Group level distances - Distances averaged at compound level
        grp_level[k0] = {0 : np.zeros((num_samples,num_samples)),  # pair-wise avg distances
                         1 : np.zeros((num_samples,num_samples)),  # pair-wise avg distances (same compound)
                         2 : np.zeros((num_samples,num_samples))}  # pair-wise avg distances (diff. compounds)

        # Sample level distances - Intra compound level - samples of the SAME compound
        smpl_level[k0] = {0 : np.empty(0),  # ## all distances
                          1 : np.empty(0),  # ## Same compound sample/sample distnaces
                          2 : np.empty(0)}  # ## Different compound sample-sample distances 

        for i in range(0,num_samples*3,3):
            for j in range(0,num_samples*3,3):
                # Take a 3x3 region of the pairiwse distance (Compounds i and j)
                tile = pairwise_distance[i:i+3, j:j+3]

                # i==j this tile contains pairwise distances of samples for the same compound
                # in this case, use the mean of distances as group level distance

                if i == j:
                    upper_triang = tile[np.triu_indices(3,k=1)]
                    smpl_level[k0][1] = np.hstack((smpl_level[k0][1], upper_triang))
                    grp_level[k0][0][i//3,j//3] = upper_triang.mean()
                    if verbose:
                        print(f"\n k0: {k0}  row: {i}  col: {j} - Row: {compounds[i]}  Col: {compounds[j]} \n {'-'*70}")
                        print(f" {tile} \n")
                        print(f" Whole Tile     :  mean: {tile.mean():.6f}")
                        print(f" Upper Triangle :  mean: {upper_triang.mean():.6f} {upper_triang}")
                        print(f" Stack          :  mean: {smpl_level[k0][1].mean():.6f} {smpl_level[k0][1][:9]} . . . " )
                        # print(f" Stack[-10:]    :                 {smpl_level[k0][1][-9:]}" )

                # if i <> j we have a distance matrix between samples of compound i and j
                else:
                    smpl_level[k0][2] = np.hstack((smpl_level[k0][2], tile.reshape(-1)))
                    grp_level[k0][0][i//3,j//3] = tile.mean()
                    # if verbose:
                    #     print(f"\n k0: {k0}  row: {i}  col: {j} - Row: {compounds[i]}  Col: {compounds[j]} \n {'-'*50}")
                    #     print(f" Pairwise distance matrix : Row: {compounds[i]}  Col: {compounds[j]} \n {tile} \n")
                    #     print(f" Tile mean     :  mean: {tile.mean():.6f} - {tile.reshape(-1)}")
                    #     print(f" Stack[:10]    :  mean: {cmp_level[k0][2].mean():.6f} {cmp_level[k0][2][:9]}" )
                    #     print(f" Stack[-10:]   :                 {cmp_level[k0][2][-9:]}" )


        triupper_indices = np.triu_indices(num_samples, k=1)
        trilower_indices = np.tril_indices(num_samples, k=-1)

        smpl_level[k0][0] = np.hstack((smpl_level[k0][1],smpl_level[k0][2]))
        ##  distances on diagonal (same compound group)
        grp_level[k0][1] = grp_level[k0][0].diagonal()
        ##  distances on non-diagonal (different compound groups)
        grp_level[k0][2] = np.hstack((grp_level[k0][0][triupper_indices],grp_level[k0][0][trilower_indices]))   

        if verbose:
            print(f"\n {'-'*80} \n")
            print(f" sample_level[{k0}][0] - Samplelevel Pairwise distances - All Compounds  : mean: {smpl_level[k0][0].mean():.4f}   shape: {smpl_level[k0][0].shape} ")
            print(f" sample_level[{k0}][0] - Samplelevel Pairwise distances - Same Compounds : mean: {smpl_level[k0][1].mean():.4f}   shape: {smpl_level[k0][1].shape} ")
            print(f" sample_level[{k0}][0] - Samplelevel Pairwise distances - Diff Compounds : mean: {smpl_level[k0][2].mean():.4f}   shape: {smpl_level[k0][2].shape} \n")
            print(f" grp_level[{k0}][0] - Grouplevel Pairwise distances - All Compounds : mean: {grp_level[k0][0].mean():.4f}   shape: {grp_level[k0][0].shape} ")
            print(f" grp_level[{k0}][1] - Grouplevel Pairwise distances - Same Compounds: mean: {grp_level[k0][1].mean():.4f}   shape: {grp_level[k0][1].shape}  (diagonal)")
            print(f" grp_level[{k0}][2] - Grouplevel Pairwise distances - Diff Compounds: mean: {grp_level[k0][2].mean():.4f}   shape: {grp_level[k0][2].shape}\n")
            print(f"\n {'-'*80} \n")
            # print(f"{grp_level[k0][0]}")
            # print(f"{grp_level[k0][1]}")
            # print(f"{grp_level[k0][2]}")

        # grp_level[4] = grp_level[1].diagonal()   ## stds on same group
        # print(grp_level[0], '\n\n\n', grp_level[0].diagonal(), '\n\n\n',grp_level[0][triupper_indices], '\n\n\n' , grp_level[0][trilower_indices], '\n\n\n') 
        # grp_level[6] = grp_level[0][triupper_indices] ## avg. distances on diff_group
        # print( grp_level[6], '\n\n\n' )
        # grp_level[7] = np.hstack((grp_level[1][triupper_indices],grp_level[1][trilower_indices])) ## std devs on diff_group  
        # print(f" upper: {grp_level[0][triupper_indices].std()}  lower: {grp_level[0][trilower_indices].std()}    upper+lower: {grp_level[6].std()} ")

        # for row, k0 in enumerate(k0_list):
        for row, (subtitle, k1) in enumerate(zip(sub_titles, k1_list)):
            # key K0: ['inp','emb','out'] & K1: ['all', 'same', 'diff']
            # row: 0 (all groups) , 1: same compound group, 2: diff compound groups
            key = k0 + '_' + k1
            # print(f" row: {row}     k0: {k0}   key: {key}")
            dm[key + '_min'] = grp_level[k0][row].min()               # min(distances)
            dm[key + '_max'] = grp_level[k0][row].max()               # max(avg_distances)
            dm[key + '_avg'] = grp_level[k0][row].mean()              # averages(distances)
            dm[key + '_std'] = grp_level[k0][row].std()               # std_dev(distances)

            smpl_mtrcs[key + '_min'] = smpl_level[k0][row].min()      # averages(ALL distances)
            smpl_mtrcs[key + '_max'] = smpl_level[k0][row].max()      # averages(ALL distances)
            smpl_mtrcs[key + '_avg'] = smpl_level[k0][row].mean()     # averages(ALL distances)
            smpl_mtrcs[key + '_std'] = smpl_level[k0][row].std()      # std_dev(ALL distances)

    if display:
        display_dist_metrics(dm, smpl_mtrcs, epochs, metric)

    return dm, smpl_mtrcs,  grp_level


# def display_dist_metrics_old(group_metrics, sample_metrics, epochs, metric = 'euclidian'):
#     titles = [f'INPUT features - {metric} distances - {epochs} epochs',
#               f'EMBEDDED features - {metric} distances - {epochs} epochs',
#               f'RECONSTURCTED features -  {metric} distances - {epochs} epochs']
#     sub_titles = ["ALL group distances", "INTRA-group (same compound) distances", "INTER-group (diff compound) distances"]
#     k0_list = ['inp', 'emb', 'out']
#     k1_list = ['all', 'same', 'diff']
#     k2_list = ['_min', '_max', '_avg', '_std', ]

#     for idx0,(title, k0) in enumerate(zip(titles, k0_list)):
#         print()
#         print(title)
#         print('-'*len(title))

#         for idx1,(subtitle, k1) in enumerate(zip(sub_titles, k1_list)):
#             # key ['inp','emb','out'] & ['all', 'same', 'diff']
#             key = k0+'_'+k1
#             row = idx1   # # 0 (all groups) , 1: same grou, 2: diff_groups
#             print(f"  {key:10s} {subtitle:37s} - Group Lvl  -  min/max: {group_metrics[key+'_min']:8.4f}   {group_metrics[key+'_max']:8.4f}"
#                   f"  | Compound Grp - mean: {group_metrics[key+'_avg']:8.4f}   std: {group_metrics[key+'_std']:8.4f}"
#                   f"  || Sample Lvl -  min/max: {sample_metrics[key+'_min']:8.4f}   {sample_metrics[key+'_max']:8.4f}"
#                   f"  | Indv/sample level distances - mean: {sample_metrics[key+'_avg']:8.4f}   std: {sample_metrics[key+'_std']:8.4f}")

#     # f"  | Indv/sample level distances - mean: {sample_metrics['CL_'+key+ '_avg']:8.4f}   std: {sample_metrics['CL_'+key+ '_std']:8.4f}"
#     print()


def display_dist_metrics(group_metrics, sample_metrics, epochs, metric = 'euclidian'):
    titles = [f'INPUT features ',
              f'EMBEDDED features',
              f'RECONSTRUCTED features ']
    sub_titles = ["ALL compounds ", "INTRA-group (same compound) distances", "INTER-group (diff compound) distances"]
    k0_list = ['inp', 'emb', 'out']
    k1_list = ['all', 'same', 'diff']

    print(f"  {metric} distances - {epochs} epochs")
    print(f"  {'':50s}         Compound Grp Lvl Averages                          All Samples    ")
    print(f"  {'':50s}    min      max          mean       std      |    min       max         mean       std")

    for idx0,(title, k0) in enumerate(zip(titles, k0_list)):
        print(f"  {title:50s} {'-'*85 if idx0 == 0 else ''}")

        for idx1,(subtitle, k1) in enumerate(zip(sub_titles, k1_list)):
            key = k0+'_'+k1    # key ['inp','emb','out'] & ['all', 'same', 'diff']
            print(f"  {key:10s} {subtitle:38s}"
                  f"  {group_metrics[key+'_min']:7.4f} - {group_metrics[key+'_max']:7.4f}   "
                  f"  {group_metrics[key+'_avg']:8.4f}   {group_metrics[key+'_std']:8.4f}"
                  f"    |"
                  f"  {sample_metrics[key+'_min']:7.4f} - {sample_metrics[key+'_max']:7.4f}  "
                  f"  {sample_metrics[key+'_avg']:8.4f}   {sample_metrics[key+'_std']:8.4f}")
        print()
    print()



def plot_distance_matrices(grp_level_dict, data = None, layers = ['inp', 'emb', 'out'], epochs = [50,150,250,350],annot = True):
    import seaborn as sns
    assert data in ['train', 'val', 'test' ]
    layer_str = {'inp': 'input', 
                 'emb': 'embedding layer',
                 'out': 'output/reconstruction'}
    # plotting the heatmap 
    # del fig, axs
    len_dict = len(grp_level_dict)
    ## figsize =(width, height)
    cols = len(layers)
    rows = len(epochs) 
    fig, axs = plt.subplots(rows,cols, sharey=False, tight_layout=True, figsize=(cols*7,rows*7) )

    row_id = 0
    col_id = 0
    for k in epochs:
        for col_id, layer in enumerate(layers):
            hm = sns.heatmap(data=grp_level_dict[data][k][layer][0], annot=annot, ax = axs[row_id,col_id], vmin = 0.0, vmax = 1.6)
            hm.set_title(f" {data.upper()} Data - epoch {k} - {layer_str[layer]}")
        # hm = sns.heatmap(data=grp_level_dict[data][k]['emb'][0],annot=annot, ax = axs[row_id,1], vmin = 0.0, vmax = 1.6)
        # hm.set_title(f"epoch {k} embedding")
        # hm = sns.heatmap(data=grp_level_dict[data][k]['out'][0],annot=annot, ax = axs[row_id,2], vmin = 0.0, vmax = 1.6)
        # hm.set_title(f"epoch {k} output")
        # if col_id == 3:
        #     row_id += 1
        #     col_id = 0
        # else:
        #     col_id += 1
        row_id += 1
    # displaying the plotted heatmap 
    plt.show()


def plot_distance_change_heatmap(grp_level_dict, data = None, epochs = None,annot = True):
    import seaborn as sns
    assert data in ['train', 'val', 'test' ]
    layer_str = {'inp': 'input', 
                 'emb': 'embedding layer',
                 'out': 'output/reconstruction'}
    # plotting the heatmap 
    # del fig, axs
    len_dict = len(grp_level_dict)
    ## figsize =(width, height)
    # cols = len(layers)
    # rows = len(epochs) 
    fig, axs = plt.subplots(1,1, sharey=False, tight_layout=True, figsize=(cols*7,rows*7) )

    row_id = 0
    col_id = 0
    delta = np.abs(grp_level_dict[data][epoch[-1]]['emb'][0] - grp_level_dict[data][epoch[0]]['emb'][0] )
    hm = sns.heatmap(data=delta,
                     annot=annot,
                     ax = axs,
                     vmin = 0.0,
                     vmax = 1.6)
    hm.set_title(f" {data.upper()} - Distance change between epochs {epochs[0]} and  {epochs[-1]} ")
 
    plt.show()



def plot_R2_rmse_scores(reg_metrics, latent_dim = 0 ):
    fig, axs = plt.subplots(1,2, sharey=False, tight_layout=True, figsize=(10,5))
    reg_metrics_grouping = reg_metrics["all"].groupby('ds')
    for grp in reg_metrics_grouping:
        _ = grp[1].plot(x = 'epoch', y='rmse_score', kind='line', subplots=False, title = f"RMSE for AE with {latent_dim} dim embedding layer", label = grp[0], ax = axs[0])
        _ = grp[1].plot(x = 'epoch', y='R2_score', kind='line', subplots=False, title = f"R2 score for AE with {latent_dim} dim embedding layer", label = grp[0], ax = axs[1])
# df_reg_metrics["all"].plot(x = 'epochs', y=['rmse_score_train','rmse_score_val','rmse_score_test',], kind='line', subplots=False, title = f"RMSE for AE with {LATENT_DIM} dim embedding layer")


def plot_distances(dist_metrics, layer = None, latent_dim = 0, verbose = False):
    assert layer in ['inp','emb', 'out'], f" Invalid layer {layer} valid vlaues: {{'inp', 'emb', 'out'}}"
    fig, axs = plt.subplots(1,3, sharey=False, tight_layout=True, figsize=(15,5))
    # fig, axs = plt.subplots(1,2, sharey=False, tight_layout=True, figsize=(10,5))
    if layer == 'emb':
        layer_str = f'embedding layer (dim{latent_dim})'
    elif layer == 'inp':
        layer_str = 'input layer'
    else:
        layer_str = 'output layer'

    sample_set_columns = ['epoch'] + [f'{layer}_all_avg', f'{layer}_same_avg', f'{layer}_diff_avg']
    sample_set_columns
    for id, ds in enumerate(['train','val','test']):
        sample_set = (dist_metrics[ds][dist_metrics[ds]['epoch'] > 0])[sample_set_columns]
        if verbose:
            print(ds, sample_set)
        _ = sample_set.plot(x = 'epoch', y = f'{layer}_all_avg', kind='line',  label = 'All Samples', subplots=False, ax = axs[id])
        _ = sample_set.plot(x = 'epoch', y = f'{layer}_same_avg', kind='line', label = 'Same Cmpnds', subplots=False, ax = axs[id])
        _ = sample_set.plot(x = 'epoch', y = f'{layer}_diff_avg', kind='line', label = 'Diff Cmpnds', subplots=False, ax = axs[id])
        axs[id].set_title(f"Avg Sample distances on {layer_str} - {ds} data", fontsize = 9)
