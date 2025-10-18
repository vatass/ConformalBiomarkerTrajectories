'''
Conformal Prediction for Randomly-Timed Biomarker Trajectories with Deep Kernel GP predictor
'''
import pandas as pd
import numpy as np
from pandas.core.arrays.sparse import dtype
from pandas.core.indexes.base import default_index 
import sys
import torch
import gpytorch
from functions import *
import pickle
from models import SingleTaskDeepKernelNonLinear, SingleTaskDeepKernel
import argparse
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
import wandb
import time 
import json
import math 
from utils import * 
import random

##### SET RANDOM SEED ####
# Set random seeds
seed = 42  
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# If using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
##################################

sns.set_style("white", {'axes.grid' : False})
# Plot Controls 
# sns.set_theme(context="paper",style="whitegrid", rc={'axes.grid' : True, 'font.serif': 'Times New Roman'})
# wandb.init(project="HMUSEDeepSingleTask", entity="vtassop", save_code=True)
parser = argparse.ArgumentParser(description='Temporal Deep Kernel Single Task GP model for a single Biomarker')
## Data Parameters 
parser.add_argument("--gpuid", help="GPUs", default=0)
parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='adniblsa') 
parser.add_argument("--file", help="Identifier for the data", default="data/conformal_longitudinal_adniblsa_data.csv")
parser.add_argument("--conformalsplitpercentage", help="Percentage of spliting Population/Calibration", default=0.04)
parser.add_argument("--alpha", help="Alpha", default=0.1)


## Kernel Parameters ##
parser.add_argument('--kernel_choice', help='Indicates the selection of GP kernel', default='RBF', type=str)
parser.add_argument('--mean', help='Selection of Mean function', default='Constant', type=str)
## Deep Kernel Parameters 
parser.add_argument("--depth", help='indicates the depth of the Deep Kernel', default='')
parser.add_argument("--activ", help='indicates the activation function', default='relu')
parser.add_argument("--dropout", help='indicates the dropout rate', default=0.2, type=float)
## Training and Data Parameters
parser.add_argument("--iterations", help="Epochs", default=50)
parser.add_argument("--optimizer", help='Optimizer', default='adam')
parser.add_argument("--learning_rate", help='Learning Rate', type=float, default=0.02)    # 0.01844 is in hmuse rois 
parser.add_argument("--roi_idx", type=int, default=14)


t0= time.time()
args = parser.parse_args()
gpuid = int(args.gpuid)
expID = args.experimentID
file = args.file
kernel = args.kernel_choice
mean = args.mean
depth = args.depth
activ = args.activ
dropout = args.dropout
iterations = args.iterations
optimizer = args.optimizer
learning_rate = args.learning_rate
task = "MUSE"
roi_idx = args.roi_idx
alpha = float(args.alpha) 

mae_TempGP_list, mae_TempGP_list_comp = [], []  

population_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': []}

population_mae_kfold = {'kfold': [], 'mae': [], 'mse': [], 'rmse': [], 'r2': [], 'interval': [], 'coverage': []}

calibration_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': []}
conformalized_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': []}
nonnormalized_conformalized_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': []}
mtv_conformalized_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': []}

external_dkgp_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'study': [], 'winkler': []}
external_conf_dkgp_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'study': [], 'winkler': []}
external_mtv_conf_dkgp_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'study': [], 'winkler': []}

qhat_dict = {'qhat': [], 'calibration_set_size': [], 'fold': [], 'qhat_mvt': [], 'mtr': [] }
strat_qhat_dict = {'qhat': [], 'covariate': [], 'calibration_set_size': [], 'fold': []} 

mae_MTGP_list, coverage_MTGP_list, interval_MTGP_list = [], [], [] 

datasamples = pd.read_csv(file)
longitudinal_covariates = pd.read_csv('data/longitudinal_covariates_conformal.csv')
print(longitudinal_covariates['Diagnosis'].unique())
# 0: CN, 1: MCI, 2: AD, -1: UKN
# map the diagnosis to integers
diagnosis_map = {0: 'CN', 1: 'MCI', 2: 'AD', -1: 'UKN'}
longitudinal_covariates['Diagnosis'] = longitudinal_covariates['Diagnosis'].map(diagnosis_map)

f = open('data/roi_to_idx.json')
roi_to_idx = json.load(f)

index_to_roi = {v: k for k, v in roi_to_idx.items()}
list_index = roi_idx
# roi_idx is the index list from 0 to 144 
roi_index = index_to_roi[roi_idx]
expID = expID + '_conformal_' + str(args.conformalsplitpercentage) + '_alpha_' + str(alpha)

print('MUSE Volume ', roi_index)
for fold in range(10): 
    print('FOLD::', fold)
    train_ids, test_ids = [], []     

    with (open("data/conformal_train_adniblsa_subjects_fold_" + str(fold) +  ".pkl", "rb")) as openfile:
        while True:
            try:
                train_ids.append(pickle.load(openfile))
            except EOFError:
                break 
      
    with (open("data/conformal_test_adniblsa_subjects_fold_" + str(fold) + ".pkl", "rb")) as openfile:
        while True:
            try:
                test_ids.append(pickle.load(openfile))
            except EOFError:
                break
    
    train_ids = train_ids[0]
    test_ids = test_ids[0]

    print('Train the Deep Kernel Regressor')
    print('Train IDs', len(train_ids))
    print('Test IDs', len(test_ids))

    for t in test_ids: 
        if t in train_ids: 
            raise ValueError('Test Samples belong to the train!')

    ### SET UP THE TRAIN/TEST DATA FOR THE MULTITASK GP### 
    train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']    
    test_x = datasamples[datasamples['PTID'].isin(test_ids)]['X']
    test_y = datasamples[datasamples['PTID'].isin(test_ids)]['Y']

    corresponding_test_ids = datasamples[datasamples['PTID'].isin(test_ids)]['PTID'].to_list()
    corresponding_train_ids = datasamples[datasamples['PTID'].isin(train_ids)]['PTID'].to_list() 
    assert len(corresponding_test_ids) == test_x.shape[0]
    assert len(corresponding_train_ids) == train_x.shape[0]

    train_x, train_y, test_x, test_y = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

    if torch.cuda.is_available():
        train_x = train_x.cuda(gpuid) 
        train_y = train_y.cuda(gpuid)
        test_x = test_x.cuda(gpuid) 
        test_y = test_y.cuda(gpuid)

    ### Check if there is negative time
    time_ = train_x[:, -1].cpu().detach().numpy().tolist()

    print('Train Time', len(time_))
    print('Train Subjectes', len(corresponding_train_ids))

    if task == 'MUSE':
        list_index = roi_idx
    elif task == 'SPARE_AD':
        list_index = 0
    elif task == 'SPARE_BA':
        list_index =1

    test_y = test_y[:, list_index]
    train_y = train_y[:, list_index]

    train_y = train_y.squeeze() 
    test_y = test_y.squeeze()

    #### DEFINE GP MODEL #### 
    depth = [(train_x.shape[1], int(train_x.shape[1]/2) )]
    dr = args.dropout
    activ = 'relu'

    lat_dim = int(train_x.shape[1]/2)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    deepkernelmodel = SingleTaskDeepKernel(input_dim=train_x.shape[1], train_x=train_x, train_y=train_y, likelihood=likelihood, depth=depth, dropout=dr, activation=activ, kernel_choice=kernel, mean=mean,
     pretrained=False, feature_extractor=None, latent_dim=int(train_x.shape[1]/2), gphyper=None) 

    if torch.cuda.is_available(): 
        likelihood = likelihood.cuda(gpuid) 
        deepkernelmodel = deepkernelmodel.cuda(gpuid)

    training_iterations  =  iterations 
        
    # set up train mode 
    deepkernelmodel.feature_extractor.train()
    deepkernelmodel.train()
    deepkernelmodel.likelihood.train()

    if args.optimizer == 'adam': 
        optimizer = torch.optim.Adam([
        {'params': deepkernelmodel.feature_extractor.parameters(), 'lr': args.learning_rate},
        {'params': deepkernelmodel.covar_module.parameters(), 'lr': args.learning_rate },
        {'params': deepkernelmodel.mean_module.parameters(), 'lr': args.learning_rate},
        {'params': deepkernelmodel.likelihood.parameters(),  'lr': args.learning_rate} ], weight_decay=0.1) ## try more reg 
    elif args.optimizer == 'sgd': 
        optimizer = torch.optim.SGD([
        {'params': deepkernelmodel.feature_extractor.parameters(), 'lr': args.learning_rate},
        {'params': deepkernelmodel.covar_module.parameters(), 'lr': args.learning_rate },
        {'params': deepkernelmodel.mean_module.parameters(), 'lr': args.learning_rate},
        {'params': deepkernelmodel.likelihood.parameters(),  'lr': args.learning_rate} ])

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, deepkernelmodel)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20, 50, 70], gamma=0.1)

    train_loss, val_loss = [], [] 
    for i in tqdm(range(50)):
        deepkernelmodel.train()
        likelihood.train()
        optimizer.zero_grad()
        output = deepkernelmodel(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        # wandb.log({"mll_train_loss": loss})
        train_loss.append(loss.item())
        optimizer.step()

    # Set into eval mode
    deepkernelmodel.eval()
    likelihood.eval()

    print('Run Inference on Test Subjects')
    for id_ in test_ids: 
        winkler_scores = [] 
        # print('Subject ID', id_)
        subject_data = datasamples[datasamples['PTID'] == id_]
        subject_covariates = longitudinal_covariates[longitudinal_covariates['PTID'] == id_]
        subject_x = subject_data['X']
        subject_y = subject_data['Y']

        subject_dict = {'y': [], 'score': [], 'id': [], 'time': [], 'age': []}

        subject_x, subject_y, _, _ = process_temporal_singletask_data(train_x=subject_x, train_y=subject_y, test_x=subject_x, test_y=subject_y)

        # verify that time is increasing 
        time_ = subject_x[:, -1].cpu().detach().numpy().tolist()
        assert np.all(np.diff(time_) > 0)
        
        if torch.cuda.is_available():
            subject_x = subject_x.cuda(gpuid) 
            subject_y = subject_y.cuda(gpuid)

        # print('Subject X', subject_x.shape)
        if roi_idx != -1:
            task = str(roi_idx)
            subject_y = subject_y[:, roi_idx]

        else: 
            subject_y = subject_y.squeeze() 
        # print('Subject Y', subject_y.shape)
        subject_x.requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            f_preds = deepkernelmodel(subject_x)
            y_preds = likelihood(f_preds)
            variance = y_preds.variance
            mean = y_preds.mean
            #lower, upper = y_preds.confidence_region()
            if alpha == 0.1:
                std = variance.sqrt()
                lower = mean - 1.645 * std
                upper = mean + 1.645 * std
            elif alpha == 0.05:
                std = variance.sqrt()
                lower = mean - 1.96 * std
                upper = mean + 1.96 * std
            elif alpha == 0.01:
                std = variance.sqrt()
                lower = mean - 2.576 * std
                upper = mean + 2.576 * std

        # Extract the derivative w.r.t. the time dimension (assumed to be the last column)

        mae_pop, ae_pop = mae(subject_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
        mse_pop, rmse_pop, se_pop = mse(subject_y.cpu().detach().numpy(), mean.cpu().detach().numpy())  
        rsq = R2(subject_y.cpu().detach().numpy(), mean.cpu().detach().numpy()) 

        coverage, interval_width, mean_coverage, mean_interval_width  = calc_coverage(predictions=mean.cpu().detach().numpy(), groundtruth=subject_y.cpu().detach().numpy(),
            intervals=[lower.cpu().detach().numpy(), upper.cpu().detach().numpy()])  
        coverage, interval_width, mean_coverage, mean_interval_width = coverage.numpy().astype(int), interval_width.numpy(), mean_coverage.numpy(), mean_interval_width.numpy() 

        mae_MTGP_list.append(mae_pop)
        coverage_MTGP_list.append(mean_coverage)
        interval_MTGP_list.append(mean_interval_width)

        # uncertainty eval
        acp = np.mean(coverage)
        ncp = acp - 0.5 

        # get the 50% width 
        posterior_std = y_preds.variance.sqrt()  # Standard deviation at each test point
        inverse_width = 1/(upper.cpu().detach().numpy()-lower.cpu().detach().numpy())

        ### Calculate the Winkler Score for this Subject ### 
        ### For the %95 confidence interval, alpha = 0.05 ###
        bayesian_alpha = alpha
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = lower[i].cpu().detach().numpy()
            u = upper[i].cpu().detach().numpy()

            if l <= y_true <= u:
                winkler_score = u - l  # Width of the interval if y_true is within the interval
            elif y_true < l:
                winkler_score = (u - l) + (2 / bayesian_alpha) * (l - y_true)  # Penalty for underprediction
            else:  # y_true > u
                winkler_score = (u - l) + (2 / bayesian_alpha) * (y_true - u)  # Penalty for overprediction

            winkler_scores.append(winkler_score)

        population_results['id'].extend([id_ for c in range(subject_x.shape[0])]) 
        population_results['kfold'].extend([fold for c in range(subject_x.shape[0])])
        population_results['score'].extend(mean.cpu().detach().numpy().tolist())
        population_results['lower'].extend(lower.cpu().detach().numpy().tolist())
        population_results['upper'].extend(upper.cpu().detach().numpy().tolist())
        population_results['y'].extend(subject_y.cpu().detach().numpy().tolist())
        population_results['variance'].extend(variance.cpu().detach().numpy().tolist())
        time_ = subject_x[:, -1].cpu().detach().numpy().tolist()
        population_results['time'].extend(time_)
        ae = np.abs(mean.cpu().detach().numpy() - subject_y.cpu().detach().numpy())
        population_results['ae'].extend(ae.tolist())
        population_results['winkler'].extend(winkler_scores)

    print('Count of Model Parameters')
    # Count feature extractor parameters
    feature_extractor_params = sum(p.numel() for p in deepkernelmodel.feature_extractor.parameters())
    print(f'Feature Extractor Parameters: {feature_extractor_params}')
    
    # Count covariance module parameters
    covar_params = sum(p.numel() for p in deepkernelmodel.covar_module.parameters())
    print(f'Covariance Module Parameters: {covar_params}')
    
    # Count mean module parameters
    mean_params = sum(p.numel() for p in deepkernelmodel.mean_module.parameters())
    print(f'Mean Module Parameters: {mean_params}')
    
    # Count likelihood parameters
    likelihood_params = sum(p.numel() for p in deepkernelmodel.likelihood.parameters())
    print(f'Likelihood Parameters: {likelihood_params}')
    
    # Total parameters
    total_params = feature_extractor_params + covar_params + mean_params + likelihood_params
    print(f'Total Parameters: {total_params}')

    print('Train the Conformalized Deep Kernel Regressor')
    ## split the train_ids into train and calibration at random

    conformal_split_percentage = args.conformalsplitpercentage
    # convert to float
    conformal_split_percentage = float(conformal_split_percentage)


    ## Select Randomly the Calibration Set
    calibration_ids = np.random.choice(train_ids, int(conformal_split_percentage*len(train_ids)), replace=False)
    train_ids = [x for x in train_ids if x not in calibration_ids]

    print('Train IDs', len(train_ids))
    print('Calibration IDs', len(calibration_ids))

    for t in calibration_ids:
        if t in train_ids: 
            raise ValueError('Calibration Samples belong to the train!')
        
    ### SET UP THE TRAIN/TEST DATA FOR THE MULTITASK GP###
    train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']
    calibration_x = datasamples[datasamples['PTID'].isin(calibration_ids)]['X']
    calibration_y = datasamples[datasamples['PTID'].isin(calibration_ids)]['Y']

    corresponding_train_ids = datasamples[datasamples['PTID'].isin(train_ids)]['PTID'].to_list()
    corresponding_calibration_ids = datasamples[datasamples['PTID'].isin(calibration_ids)]['PTID'].to_list()

    assert len(corresponding_train_ids) == train_x.shape[0]
    assert len(corresponding_calibration_ids) == calibration_x.shape[0]

    train_x, train_y, calibration_x, calibration_y = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=calibration_x, test_y=calibration_y)

    if torch.cuda.is_available():
        train_x = train_x.cuda(gpuid) 
        train_y = train_y.cuda(gpuid)
        calibration_x = calibration_x.cuda(gpuid)
        calibration_y = calibration_y.cuda(gpuid)


    calibration_y = calibration_y[:, list_index]
    train_y = train_y[:, list_index]

    train_y = train_y.squeeze()
    calibration_y = calibration_y.squeeze()

    #### DEFINE GP MODEL ####
    depth = [(train_x.shape[1], int(train_x.shape[1]/2))]
    dr = args.dropout
    activ = 'relu'

    lat_dim = int(train_x.shape[1]/2)
    conformal_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    conformal_deepkernelmodel = SingleTaskDeepKernel(input_dim=train_x.shape[1], train_x=train_x, train_y=train_y, likelihood=likelihood, depth=depth, dropout=dr, activation=activ, kernel_choice=kernel, mean=mean,
        pretrained=False, feature_extractor=None, latent_dim=int(train_x.shape[1]/2), gphyper=None)
    
    if torch.cuda.is_available():
        conformal_likelihood = conformal_likelihood.cuda(gpuid)
        conformal_deepkernelmodel = conformal_deepkernelmodel.cuda(gpuid)

    training_iterations  =  iterations

    # set up train mode
    conformal_deepkernelmodel.feature_extractor.train()
    conformal_deepkernelmodel.train()
    conformal_deepkernelmodel.likelihood.train()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([
        {'params': conformal_deepkernelmodel.feature_extractor.parameters(), 'lr': args.learning_rate},
        {'params': conformal_deepkernelmodel.covar_module.parameters(), 'lr': args.learning_rate },
        {'params': conformal_deepkernelmodel.mean_module.parameters(), 'lr': args.learning_rate},
        {'params': conformal_deepkernelmodel.likelihood.parameters(),  'lr': args.learning_rate} ], weight_decay=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(conformal_likelihood, conformal_deepkernelmodel)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20, 50, 70], gamma=0.1)

    train_loss, val_loss = [], []
    for i in tqdm(range(50)): # it was 200 
        conformal_deepkernelmodel.train()
        conformal_likelihood.train()
        optimizer.zero_grad()
        output = conformal_deepkernelmodel(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        # wandb.log({"mll_train_loss": loss})
        train_loss.append(loss.item())
        optimizer.step()

    # Set into eval mode
    conformal_deepkernelmodel.eval()
    conformal_likelihood.eval()

    print('Calculate the Train Residuals')
    absolute_residuals_list = []
    for id_ in train_ids:

        subject_data = datasamples[datasamples['PTID'] == id_]
        subject_covariates = longitudinal_covariates[longitudinal_covariates['PTID'] == id_]
        subject_x = subject_data['X']
        subject_y = subject_data['Y']

        subject_dict = {'y': [], 'score': [], 'id': [], 'time': [], 'age': []}

        subject_x, subject_y, _, _ = process_temporal_singletask_data(train_x=subject_x, train_y=subject_y, test_x=subject_x, test_y=subject_y)

        # verify that time is increasing 
        time_ = subject_x[:, -1].cpu().detach().numpy().tolist()
        assert np.all(np.diff(time_) > 0)
        
        if torch.cuda.is_available():
            subject_x = subject_x.cuda(gpuid) 
            subject_y = subject_y.cuda(gpuid)

        subject_y = subject_y[:, list_index]
        subject_y = subject_y.squeeze()

        # print('Subject Y', subject_y.shape)
        subject_x.requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            f_preds = conformal_deepkernelmodel(subject_x)
            y_preds = conformal_likelihood(f_preds)
            variance = y_preds.variance
            mean = y_preds.mean
         
        absolute_residuals = np.abs(mean.cpu().detach().numpy() - subject_y.cpu().detach().numpy()).tolist()
        absolute_residuals_list.extend(absolute_residuals)

    print('Max Train Residuals', np.max(absolute_residuals_list))
    max_train_residual = np.max(absolute_residuals_list)

    qhat_dict['mtr'].append(max_train_residual)

    print('Run Inference on Calibration Subjects')
    for id_ in calibration_ids:
        # print('Subject ID', id_)
        subject_data = datasamples[datasamples['PTID'] == id_]
        subject_covariates = longitudinal_covariates[longitudinal_covariates['PTID'] == id_]
        subject_x = subject_data['X']
        subject_y = subject_data['Y']

        subject_dict = {'y': [], 'score': [], 'id': [], 'time': [], 'age': []}

        subject_x, subject_y, _, _ = process_temporal_singletask_data(train_x=subject_x, train_y=subject_y, test_x=subject_x, test_y=subject_y)

        # verify that time is increasing 
        time_ = subject_x[:, -1].cpu().detach().numpy().tolist()
        assert np.all(np.diff(time_) > 0)
        
        if torch.cuda.is_available():
            subject_x = subject_x.cuda(gpuid) 
            subject_y = subject_y.cuda(gpuid)

        subject_y = subject_y[:, list_index]
        subject_y = subject_y.squeeze()
        
        # print('Subject Y', subject_y.shape)
        subject_x.requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            f_preds = conformal_deepkernelmodel(subject_x)
            y_preds = conformal_likelihood(f_preds)
            variance = y_preds.variance
            mean = y_preds.mean
            #lower, upper = y_preds.confidence_region()
            # 90% confidence interval
            std = variance.sqrt()

            if alpha == 0.1:
                std = variance.sqrt()
                lower = mean - 1.645 * std
                upper = mean + 1.645 * std
            elif alpha == 0.05:
                std = variance.sqrt()
                lower = mean - 1.96 * std
                upper = mean + 1.96 * std
            elif alpha == 0.01:
                std = variance.sqrt()
                lower = mean - 2.576 * std
                upper = mean + 2.576 * std

        mae_pop, ae_pop = mae(subject_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
        mse_pop, rmse_pop, se_pop = mse(subject_y.cpu().detach().numpy(), mean.cpu().detach().numpy())  
        rsq = R2(subject_y.cpu().detach().numpy(), mean.cpu().detach().numpy()) 

        mae_MTGP_list.append(mae_pop)
        coverage_MTGP_list.append(mean_coverage)
        interval_MTGP_list.append(mean_interval_width)

        # get the 50% width 
        posterior_std = y_preds.variance.sqrt()  # Standard deviation at each test point
        inverse_width = 1/(upper.cpu().detach().numpy()-lower.cpu().detach().numpy())

        calibration_results['id'].extend([id_ for c in range(subject_x.shape[0])]) 
        calibration_results['kfold'].extend([fold for c in range(subject_x.shape[0])])
        calibration_results['score'].extend(mean.cpu().detach().numpy().tolist())
        calibration_results['lower'].extend(lower.cpu().detach().numpy().tolist())
        calibration_results['upper'].extend(upper.cpu().detach().numpy().tolist())
        calibration_results['y'].extend(subject_y.cpu().detach().numpy().tolist())
        calibration_results['variance'].extend(variance.cpu().detach().numpy().tolist())
        time_ = subject_x[:, -1].cpu().detach().numpy().tolist()
        calibration_results['time'].extend(time_)

        ae = np.abs(mean.cpu().detach().numpy() - subject_y.cpu().detach().numpy())
        calibration_results['ae'].extend(ae.tolist())

    calibration_results_df = pd.DataFrame(data=calibration_results)

    print('Calculate the Non-Conformity Scores')
    conformity_scores_per_subject = {'id': [], 'conformal_scores': [], 'nonnorm_conformal_scores': [], 'conformal_scores_mtv': []}
    for subject in calibration_results_df['id'].unique():
        subject_df = calibration_results_df[calibration_results_df['id'] == subject]

        std = np.sqrt(subject_df['variance'])

        conformal_scores = np.abs(subject_df['score'] - subject_df['y'])/std #
        nonnorm = np.abs(subject_df['score'] - subject_df['y'])

        conformal_scores_mtv = np.abs(subject_df['score'] - subject_df['y'])/max_train_residual


        conformity_scores_per_subject['id'].append(subject)
        conformity_scores_per_subject['conformal_scores'].append(np.max(conformal_scores))
        conformity_scores_per_subject['nonnorm_conformal_scores'].append(np.max(nonnorm))
        conformity_scores_per_subject['conformal_scores_mtv'].append(np.max(conformal_scores_mtv))


    conformity_scores_per_subject_df = pd.DataFrame(data=conformity_scores_per_subject)
    conformity_scores_per_subject_df.to_csv('conformal_scores_fold_' + str(fold) + '_'+ expID + '.csv', index=False)

    # gather all the conformity scores
    conformity_scores = np.array(conformity_scores_per_subject['conformal_scores'])
    sorted_conformity_scores = np.sort(conformity_scores)
    # n is the number of the validation subjects 
    n = conformity_scores.shape[0]
    conformal_alpha = alpha
    # alpha is the user-chosen error rate. In w ords, the probability that the prediction set contains the correct label is almost exactly 1 􀀀 ; we call
    # this property marginal coverage, since the probability is marginal (averaged) over the randomness in the
    # calibration and test points

    k = int(np.ceil((n + 1) * (1 - conformal_alpha)))
    # Ensure k does not exceed n
    k = min(k, n)
    # Get the (n - k + 1)-th smallest value since we want the k-th largest value
    qhat = sorted_conformity_scores[k-1]
    print('Qhat', qhat)
    print('Calibration Set Size',len(calibration_ids))

    qhat_dict['qhat'].append(qhat)
    qhat_dict['calibration_set_size'].append(len(calibration_results_df['id'].unique()))
    qhat_dict['fold'].append(fold)

    #### Calculate the non-normalized non-conformity scores ####
    nonnorm_conformity_scores = np.array(conformity_scores_per_subject['nonnorm_conformal_scores'])
    sorted_nonnorm_conformity_scores = np.sort(nonnorm_conformity_scores)
    # n is the number of the validation subjects
    n = nonnorm_conformity_scores.shape[0]
    # alpha is the user-chosen error rate. In w ords, the probability that the prediction set contains the correct label is almost exactly 
    k = int(np.ceil((n + 1) * (1 - conformal_alpha)))
    # Ensure k does not exceed n
    k = min(k, n)
    # Get the (n - k + 1)-th smallest value since we want the k-th largest value
    qhat_nonnorm = sorted_nonnorm_conformity_scores[k-1 ]

    # Calculate the quantile with mtv non-conformity scores 
    conformity_scores_mtv = np.array(conformity_scores_per_subject['conformal_scores_mtv'])
    sorted_conformity_scores_mtv = np.sort(conformity_scores_mtv)
    # n is the number of the validation subjects
    n = conformity_scores_mtv.shape[0]
    # alpha is the user-chosen error rate. In w ords, the probability that the prediction set contains the correct label is almost exactly
    k = int(np.ceil((n + 1) * (1 - conformal_alpha)))
    # Ensure k does not exceed n
    k = min(k, n)
    # Get the (n - k + 1)-th smallest value since we want the k-th largest value
    qhat_mtv = sorted_conformity_scores_mtv[k-1]

    qhat_dict['qhat_mvt'].append(qhat_mtv)


    print('Conformalized Inference on Test Subjects')
    print('Run Inference on Test Subjects')
    for id_ in test_ids: 
        # print('Subject ID', id_)
        winkler_scores = []
        subject_data = datasamples[datasamples['PTID'] == id_]
        subject_covariates = longitudinal_covariates[longitudinal_covariates['PTID'] == id_]
        subject_x = subject_data['X']
        subject_y = subject_data['Y']

        subject_dict = {'y': [], 'score': [], 'id': [], 'time': [], 'age': []}

        subject_x, subject_y, _, _ = process_temporal_singletask_data(train_x=subject_x, train_y=subject_y, test_x=subject_x, test_y=subject_y)

        # verify that time is increasing 
        time_ = subject_x[:, -1].cpu().detach().numpy().tolist()
        assert np.all(np.diff(time_) > 0)
        
        if torch.cuda.is_available():
            subject_x = subject_x.cuda(gpuid) 
            subject_y = subject_y.cuda(gpuid)

        subject_y = subject_y[:, list_index]
        subject_y = subject_y.squeeze()
        
        # print('Subject Y', subject_y.shape)
        subject_x.requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            f_preds = conformal_deepkernelmodel(subject_x)
            y_preds = conformal_likelihood(f_preds)
            variance = y_preds.variance
            mean = y_preds.mean

        mae_pop, ae_pop = mae(subject_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
        mse_pop, rmse_pop, se_pop = mse(subject_y.cpu().detach().numpy(), mean.cpu().detach().numpy())  
        rsq = R2(subject_y.cpu().detach().numpy(), mean.cpu().detach().numpy()) 

        mae_MTGP_list.append(mae_pop)

        std = variance.sqrt()
        conformal_lower = mean - qhat * std
        conformal_upper = mean + qhat * std

        ### Calculate the Winkler Score for this Subject ### 
        ### 
        winkler_scores = []
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = conformal_lower[i].cpu().detach().numpy()
            u = conformal_upper[i].cpu().detach().numpy()

            if l <= y_true <= u:
                winkler_score = u - l  # Width of the interval if y_true is within the interval
            elif y_true < l:
                winkler_score = (u - l) + (2 / conformal_alpha) * (l - y_true)  # Penalty for underprediction
            else:  # y_true > u
                winkler_score = (u - l) + (2 / conformal_alpha) * (y_true - u)  # Penalty for overprediction

            winkler_scores.append(winkler_score) 

        conformalized_results['id'].extend([id_ for c in range(subject_x.shape[0])]) 
        conformalized_results['kfold'].extend([fold for c in range(subject_x.shape[0])])
        conformalized_results['score'].extend(mean.cpu().detach().numpy().tolist())
        conformalized_results['lower'].extend(conformal_lower.cpu().detach().numpy().tolist())
        conformalized_results['upper'].extend(conformal_upper.cpu().detach().numpy().tolist())
        conformalized_results['y'].extend(subject_y.cpu().detach().numpy().tolist())
        conformalized_results['variance'].extend(variance.cpu().detach().numpy().tolist())
        time_ = subject_x[:, -1].cpu().detach().numpy().tolist()
        conformalized_results['time'].extend(time_)

        ae = np.abs(mean.cpu().detach().numpy() - subject_y.cpu().detach().numpy())
        conformalized_results['ae'].extend(ae.tolist())
        conformalized_results['winkler'].extend(winkler_scores)

        # print('Calculate the Conformalized Results with the Non-Normalized Non-Conformity Scores')
        std = variance.sqrt()
        non_conformal_lower = mean - qhat_nonnorm * std
        non_conformal_upper = mean + qhat_nonnorm * std

        #### Calculate the Winkler Score for this Subject ####
        winkler_scores = []
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = non_conformal_lower[i].cpu().detach().numpy()
            u = non_conformal_upper[i].cpu().detach().numpy()

            if l <= y_true <= u:
                winkler_score = u - l  # Width of the interval if y_true is within the interval
            elif y_true < l:
                winkler_score = (u - l) + (2 / conformal_alpha) * (l - y_true)  # Penalty for underprediction
            else:  # y_true > u
                winkler_score = (u - l) + (2 / conformal_alpha) * (y_true - u)  # Penalty for overprediction

            winkler_scores.append(winkler_score) 

        nonnormalized_conformalized_results['id'].extend([id_ for c in range(subject_x.shape[0])])
        nonnormalized_conformalized_results['kfold'].extend([fold for c in range(subject_x.shape[0])])
        nonnormalized_conformalized_results['score'].extend(mean.cpu().detach().numpy().tolist())
        nonnormalized_conformalized_results['lower'].extend(non_conformal_lower.cpu().detach().numpy().tolist())
        nonnormalized_conformalized_results['upper'].extend(non_conformal_upper.cpu().detach().numpy().tolist())
        nonnormalized_conformalized_results['y'].extend(subject_y.cpu().detach().numpy().tolist())
        nonnormalized_conformalized_results['variance'].extend(variance.cpu().detach().numpy().tolist())
        time_ = subject_x[:, -1].cpu().detach().numpy().tolist()
        nonnormalized_conformalized_results['time'].extend(time_)
        ae = np.abs(mean.cpu().detach().numpy() - subject_y.cpu().detach().numpy())
        nonnormalized_conformalized_results['ae'].extend(ae.tolist())
        nonnormalized_conformalized_results['winkler'].extend(winkler_scores)

        # Reults with Max Train Residual 
        mtv_conformal_lower = mean - qhat_mtv * max_train_residual
        mtv_conformal_upper = mean + qhat_mtv * max_train_residual

        #### Calculate the Winkler Score for this Subject ####
        winkler_scores = []
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = mtv_conformal_lower[i].cpu().detach().numpy()
            u = mtv_conformal_upper[i].cpu().detach().numpy()

            if l <= y_true <= u:
                winkler_score = u - l
            elif y_true < l:
                winkler_score = (u - l) + (2 / conformal_alpha) * (l - y_true)
            else:  # y_true > u
                winkler_score = (u - l) + (2 / conformal_alpha) * (y_true - u)
                
            winkler_scores.append(winkler_score)

        mtv_conformalized_results['id'].extend([id_ for c in range(subject_x.shape[0])])
        mtv_conformalized_results['kfold'].extend([fold for c in range(subject_x.shape[0])])
        mtv_conformalized_results['score'].extend(mean.cpu().detach().numpy().tolist())
        mtv_conformalized_results['lower'].extend(mtv_conformal_lower.cpu().detach().numpy().tolist())
        mtv_conformalized_results['upper'].extend(mtv_conformal_upper.cpu().detach().numpy().tolist())
        mtv_conformalized_results['y'].extend(subject_y.cpu().detach().numpy().tolist())
        mtv_conformalized_results['variance'].extend(variance.cpu().detach().numpy().tolist())
        time_ = subject_x[:, -1].cpu().detach().numpy().tolist()
        mtv_conformalized_results['time'].extend(time_)
        ae = np.abs(mean.cpu().detach().numpy() - subject_y.cpu().detach().numpy())
        mtv_conformalized_results['ae'].extend(ae.tolist())
        mtv_conformalized_results['winkler'].extend(winkler_scores)

    print('Save the Results')


    save_model(deepkernelmodel, optimizer, likelihood, filename='./results/population_deep_kernel_gp_'+ str(roi_idx) + '.pth')
    ### Extract the Feature Importance for Interpretability ###
    weights = deepkernelmodel.feature_extractor.final_linear.weight.cpu().detach()
    feature_importance = torch.abs(weights).mean(dim=0)
    feature_importance_np = feature_importance.numpy()

    # store 
    np.save('./conformalresults/feature_importance_' + str(task) + '_' + str(list_index) + '_' + str(fold) + '_.npy', feature_importance_np)

    '''
    Apply to External Studies: OASIS, AIBL, PreventAD, WRAP, Penn, CARDIA
    '''
    print('Do inference on the external clinical studies')
    external_studies = ['oasis', 'penn', 'aibl', 'preventad', 'wrap', 'cardia']

    for study in external_studies:
        print('Study', study)

        if task.startswith('SPARE'):
            external_data = pd.read_csv('conformal_longitudinal_spare_' + study + '.csv')
        else:
            external_data = pd.read_csv('conformal_longitudinal_' + study + '_data.csv')
        
        external_study_ids = external_data['PTID'].unique()
        data_covariates = pd.read_csv('longitudinal_covariates_' + study + '_conformal.csv')

        for id_ in external_study_ids: 
            # print('Subject ID', id_)
            winkler_scores = []
            data = external_data[external_data['PTID'] == id_]
            subject_covariates = data_covariates[data_covariates['PTID'] == id_]
    
            data_x = data['X']
            data_y = data['Y']

            data_x, data_y, _, _ = process_temporal_singletask_data(train_x=data_x, train_y=data_y, test_x=data_x, test_y=data_y)

            if torch.cuda.is_available():
                data_x = data_x.cuda(gpuid) 
                data_y = data_y.cuda(gpuid)

            data_y = data_y[:, list_index]
            data_y = data_y.squeeze()

            data_x.requires_grad_(True)
            with gpytorch.settings.fast_pred_var():
                f_preds = deepkernelmodel(data_x)
                y_preds = likelihood(f_preds)
                variance = y_preds.variance
                mean = y_preds.mean
                if alpha == 0.1:
                    std = variance.sqrt()
                    lower = mean - 1.645 * std
                    upper = mean + 1.645 * std
                elif alpha == 0.05:
                    std = variance.sqrt()
                    lower = mean - 1.96 * std
                    upper = mean + 1.96 * std
                elif alpha == 0.01:
                    std = variance.sqrt()
                    lower = mean - 2.576 * std
                    upper = mean + 2.576 * std

            ## Winkler Score for the Bayesian Bounds ## 
            ## For the %95 confidence interval, alpha = 0.05 ##
            bayesian_alpha = alpha
            for i in range(len(data_y)):

                y_true = data_y[i].cpu().detach().numpy()
                l = lower[i].cpu().detach().numpy()
                u = upper[i].cpu().detach().numpy()

                if l <= y_true <= u:
                    winkler_score = u - l  # Width of the interval if y_true is within the interval
                elif y_true < l:
                    winkler_score = (u - l) + (2 / bayesian_alpha) * (l - y_true)  # Penalty for underprediction
                else:  # y_true > u
                    winkler_score = (u - l) + (2 / bayesian_alpha) * (y_true - u)  # Penalty for overprediction

                winkler_scores.append(winkler_score)


            mae_pop, ae_pop = mae(data_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
            mse_pop, rmse_pop, se_pop = mse(data_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
            rsq = R2(data_y.cpu().detach().numpy(), mean.cpu().detach().numpy())

            coverage, interval_width, mean_coverage, mean_interval_width = calc_coverage(predictions=mean.cpu().detach().numpy(), groundtruth=data_y.cpu().detach().numpy(),
                intervals=[lower.cpu().detach().numpy(), upper.cpu().detach().numpy()])
            coverage, interval_width, mean_coverage, mean_interval_width = coverage.numpy().astype(int), interval_width.numpy(), mean_coverage.numpy(), mean_interval_width.numpy()

            external_dkgp_results['id'].extend([id_ for c in range(data_x.shape[0])])
            external_dkgp_results['kfold'].extend([fold for c in range(data_x.shape[0])])
            external_dkgp_results['score'].extend(mean.cpu().detach().numpy().tolist())
            external_dkgp_results['lower'].extend(lower.cpu().detach().numpy().tolist())
            external_dkgp_results['upper'].extend(upper.cpu().detach().numpy().tolist())
            external_dkgp_results['y'].extend(data_y.cpu().detach().numpy().tolist())
            external_dkgp_results['variance'].extend(variance.cpu().detach().numpy().tolist())
            time_ = data_x[:, -1].cpu().detach().numpy().tolist()
            external_dkgp_results['time'].extend(time_)
            external_dkgp_results['study'].extend([study for c in range(data_x.shape[0])])

            ae = np.abs(mean.cpu().detach().numpy() - data_y.cpu().detach().numpy())
            external_dkgp_results['ae'].extend(ae.tolist())
            external_dkgp_results['winkler'].extend(winkler_scores)

            data_x.requires_grad_(True)
            with gpytorch.settings.fast_pred_var():
                f_preds = conformal_deepkernelmodel(data_x)
                y_preds = conformal_likelihood(f_preds)
                variance = y_preds.variance
                mean = y_preds.mean
                lower, upper = y_preds.confidence_region()

            mae_pop, ae_pop = mae(data_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
            mse_pop, rmse_pop, se_pop = mse(data_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
            rsq = R2(data_y.cpu().detach().numpy(), mean.cpu().detach().numpy())

            mae_MTGP_list.append(mae_pop)

            std = variance.sqrt()
            conformal_lower = mean - qhat * std
            conformal_upper = mean + qhat * std

            ### Calculate the Winkler Score for this Subject ###
            winkler_scores= [] 
            for i in range(len(data_y)):
                y_true = data_y[i].cpu().detach().numpy()
                l = conformal_lower[i].cpu().detach().numpy()
                u = conformal_upper[i].cpu().detach().numpy()

                if l <= y_true <= u:
                    winkler_score = u - l
                elif y_true < l:
                    winkler_score = (u - l) + (2 / conformal_alpha) * (l - y_true)
                else:  # y_true > u
                    winkler_score = (u - l) + (2 / conformal_alpha) * (y_true - u)

                winkler_scores.append(winkler_score)

            external_conf_dkgp_results['id'].extend([id_ for c in range(data_x.shape[0])])
            external_conf_dkgp_results['kfold'].extend([fold for c in range(data_x.shape[0])])

            external_conf_dkgp_results['score'].extend(mean.cpu().detach().numpy().tolist())
            external_conf_dkgp_results['lower'].extend(conformal_lower.cpu().detach().numpy().tolist())
            external_conf_dkgp_results['upper'].extend(conformal_upper.cpu().detach().numpy().tolist())
            external_conf_dkgp_results['y'].extend(data_y.cpu().detach().numpy().tolist())
            external_conf_dkgp_results['variance'].extend(variance.cpu().detach().numpy().tolist())
            time_ = data_x[:, -1].cpu().detach().numpy().tolist()
            external_conf_dkgp_results['time'].extend(time_)
            ae = np.abs(mean.cpu().detach().numpy() - data_y.cpu().detach().numpy())
            external_conf_dkgp_results['ae'].extend(ae.tolist())
            external_conf_dkgp_results['study'].extend([study for c in range(data_x.shape[0])]) 
            external_conf_dkgp_results['winkler'].extend(winkler_scores)

            # MTV Conformalized Results
            std = variance.sqrt()
            mtv_conformal_lower = mean - qhat_mtv * max_train_residual
            mtv_conformal_upper = mean + qhat_mtv * max_train_residual

            ### Calculate the Winkler Score for this Subject ###
            winkler_scores = []
            for i in range(len(data_y)):
                y_true = data_y[i].cpu().detach().numpy()
                l = mtv_conformal_lower[i].cpu().detach().numpy()
                u = mtv_conformal_upper[i].cpu().detach().numpy()

                if l <= y_true <= u:
                    winkler_score = u - l
                elif y_true < l:
                    winkler_score = (u - l) + (2 / conformal_alpha) * (l - y_true)
                else:
                    winkler_score = (u - l) + (2 / conformal_alpha) * (y_true - u)

                winkler_scores.append(winkler_score)

            external_mtv_conf_dkgp_results['id'].extend([id_ for c in range(data_x.shape[0])])
            external_mtv_conf_dkgp_results['kfold'].extend([fold for c in range(data_x.shape[0])])

            external_mtv_conf_dkgp_results['score'].extend(mean.cpu().detach().numpy().tolist())
            external_mtv_conf_dkgp_results['lower'].extend(mtv_conformal_lower.cpu().detach().numpy().tolist())
            external_mtv_conf_dkgp_results['upper'].extend(mtv_conformal_upper.cpu().detach().numpy().tolist())
            external_mtv_conf_dkgp_results['y'].extend(data_y.cpu().detach().numpy().tolist())
            external_mtv_conf_dkgp_results['variance'].extend(variance.cpu().detach().numpy().tolist())
            time_ = data_x[:, -1].cpu().detach().numpy().tolist()
            external_mtv_conf_dkgp_results['time'].extend(time_)
            ae = np.abs(mean.cpu().detach().numpy() - data_y.cpu().detach().numpy())
            external_mtv_conf_dkgp_results['ae'].extend(ae.tolist())
            external_mtv_conf_dkgp_results['study'].extend([study for c in range(data_x.shape[0])])
            external_mtv_conf_dkgp_results['winkler'].extend(winkler_scores)

    print('Save the External Results')

# enhance the expID with the calibration split percentage
print('#### Evaluation of the Singletask Deep Kernel Temporal GP for ROI ' + str(task) + '###')
population_mae_kfold_df = pd.DataFrame(data=population_mae_kfold)
population_mae_kfold_df.to_csv('./conformalresults/singletask_'+ str(task) + '_' + str(roi_idx) + '_dkgp_mae_kfold_'+ expID+'.csv')

population_results_df = pd.DataFrame(data=population_results)
population_results_df.to_csv('./conformalresults/singletask_' + str(task)+'_' + str(roi_idx) + '_dkgp_population_'+ expID+'.csv')

conformalized_predictions_df = pd.DataFrame(data=conformalized_results)
conformalized_predictions_df.to_csv('./conformalresults/singletask_' + str(task)+ '_' + str(roi_idx) + '_dkgp_conformalized_predictions_'+ expID+'.csv')

nonnormalized_conformalized_predictions_df = pd.DataFrame(data=nonnormalized_conformalized_results)
nonnormalized_conformalized_predictions_df.to_csv('./conformalresults/singletask_' + str(task)+ '_' + str(roi_idx) + '_dkgp_nonnormalized_conformalized_predictions_'+ expID+'.csv')

mtv_conformalized_predictions_df = pd.DataFrame(data=mtv_conformalized_results)
mtv_conformalized_predictions_df.to_csv('./conformalresults/singletask_' + str(task)+ '_' + str(roi_idx) + '_dkgp_mtv_conformalized_predictions_'+ expID+'.csv')

# store the external results
external_dkgp_results_df = pd.DataFrame(data=external_dkgp_results)
external_dkgp_results_df.to_csv('./conformalresults/singletask_' + str(task)+ '_' + str(roi_idx) + '_dkgp_external_predictions_'+ expID+'.csv')

external_conf_dkgp_results_df = pd.DataFrame(data=external_conf_dkgp_results)
external_conf_dkgp_results_df.to_csv('./conformalresults/singletask_' + str(task)+ '_' + str(roi_idx) + '_dkgp_external_conformalized_predictions_'+ expID+'.csv')

external_mtv_conf_dkgp_results_df = pd.DataFrame(data=external_mtv_conf_dkgp_results)
external_mtv_conf_dkgp_results_df.to_csv('./conformalresults/singletask_' + str(task)+ '_' + str(roi_idx) + '_dkgp_external_mtv_conformalized_predictions_'+ expID+'.csv')

# store the qhat values
qhat_df = pd.DataFrame(data=qhat_dict)
qhat_df.to_csv('./conformalresults/qhat_' + str(task)+ '_' + str(roi_idx) + '_'+ expID+'.csv')

# store the qhat values
strat_qhat_df = pd.DataFrame(data=strat_qhat_dict)
strat_qhat_df.to_csv('./conformalresults/strat_qhat_' + str(task)+ '_' + str(roi_idx) + '_'+ expID+'.csv')


print('#### Evaluation of the Singletask Deep Kernel Temporal GP for ' + str(task) + '###')
print('POPULATION GP')
print('mean:', np.mean(mae_MTGP_list, axis=0))
print('var:', np.var(mae_MTGP_list, axis=0))
print('Interval', np.mean(interval_MTGP_list), np.var(interval_MTGP_list))
print('Coverage', np.mean(coverage_MTGP_list), np.var(coverage_MTGP_list))

t1 = time.time() - t0 
print("Time elapsed: ", t1)
print('Qhat', qhat)
print('Calibration Set Size',len(calibration_results_df['id'].unique()))

