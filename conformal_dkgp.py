'''
 - Deep Kernel GP Predictor for Randomly-Timed Biomarker Trajectories
 - Conformal DKGP 
'''
import os
import sys
import time
import json
import math
import random
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import gpytorch

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from functions import process_temporal_singletask_data
from models import SingleTaskDeepKernel

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

parser = argparse.ArgumentParser(description='Temporal Deep Kernel Single Task GP model for a single Biomarker')
parser.add_argument("--gpuid", help="GPUs", default=0)
parser.add_argument("--file", help="Identifier for the data", default="./data/data.csv")
parser.add_argument("--conformalsplitpercentage", help="Percentage of spliting Population/Calibration", default=0.04)
parser.add_argument("--alpha", help="Alpha", default=0.1)
parser.add_argument("--biomarker_idx", type=int, default=14)


t0= time.time()
args = parser.parse_args()
gpuid = int(args.gpuid)
file = args.file
biomarker_idx = args.biomarker_idx
alpha = float(args.alpha) 

results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)
print(f"✅ Results directory : {results_dir}")

population_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': []}
calibration_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': []}
conformalized_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': []}
qhat_dict = {'qhat': [], 'calibration_set_size': [], 'fold': []}

datasamples = pd.read_csv(file)

list_index = biomarker_idx

for fold in range(10): 
    print('FOLD::', fold)
    train_ids, test_ids = [], []     

    with (open("./data/folds/fold_" + str(fold) +  "_train.pkl", "rb")) as openfile:
        while True:
            try:
                train_ids.append(pickle.load(openfile))
            except EOFError:
                break 
      
    with (open("./data/folds/fold_" + str(fold) +  "_test.pkl", "rb")) as openfile:
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
    train_x = datasamples[datasamples['anon_id'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['anon_id'].isin(train_ids)]['Y']    
    test_x = datasamples[datasamples['anon_id'].isin(test_ids)]['X']
    test_y = datasamples[datasamples['anon_id'].isin(test_ids)]['Y']

    corresponding_test_ids = datasamples[datasamples['anon_id'].isin(test_ids)]['anon_id'].to_list()
    corresponding_train_ids = datasamples[datasamples['anon_id'].isin(train_ids)]['anon_id'].to_list() 
    assert len(corresponding_test_ids) == test_x.shape[0]
    assert len(corresponding_train_ids) == train_x.shape[0]

    train_x, train_y, test_x, test_y = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

    if torch.cuda.is_available():
        train_x = train_x.cuda(gpuid) 
        train_y = train_y.cuda(gpuid)
        test_x = test_x.cuda(gpuid) 
        test_y = test_y.cuda(gpuid)

    time_ = train_x[:, -1].cpu().detach().numpy().tolist()

    print('Train Time', len(time_))
    print('Train Subjectes', len(corresponding_train_ids))

    test_y = test_y[:, list_index]
    train_y = train_y[:, list_index]

    train_y = train_y.squeeze() 
    test_y = test_y.squeeze()

    #### DEFINE DKGP MODEL #### 
    depth = [(train_x.shape[1], int(train_x.shape[1]/2) )]
    dr = 0.2
    activ = 'relu'
    kernel = 'RBF'
    mean = 'Constant'
    ###########################

    lat_dim = int(train_x.shape[1]/2)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    deepkernelmodel = SingleTaskDeepKernel(input_dim=train_x.shape[1], train_x=train_x, train_y=train_y, likelihood=likelihood, depth=depth, dropout=dr, activation=activ, kernel_choice=kernel, mean=mean,
     pretrained=False, feature_extractor=None, latent_dim=int(train_x.shape[1]/2), gphyper=None) 

    if torch.cuda.is_available(): 
        likelihood = likelihood.cuda(gpuid) 
        deepkernelmodel = deepkernelmodel.cuda(gpuid)
        
    # set up train mode 
    deepkernelmodel.feature_extractor.train()
    deepkernelmodel.train()
    deepkernelmodel.likelihood.train()

    optimizer = torch.optim.Adam([
    {'params': deepkernelmodel.feature_extractor.parameters(), 'lr': 0.02},
    {'params': deepkernelmodel.covar_module.parameters(), 'lr': 0.02 },
    {'params': deepkernelmodel.mean_module.parameters(), 'lr': 0.02},
    {'params': deepkernelmodel.likelihood.parameters(),  'lr': 0.02} ], weight_decay=0.1) ## try more reg 

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, deepkernelmodel)

    train_loss, val_loss = [], [] 
    for i in tqdm(range(50)):
        deepkernelmodel.train()
        likelihood.train()
        optimizer.zero_grad()
        output = deepkernelmodel(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()

    # Set into eval mode
    deepkernelmodel.eval()
    likelihood.eval()

    print('Run Inference on Test Subjects')
    for id_ in test_ids: 
        winkler_scores = [] 
        subject_data = datasamples[datasamples['anon_id'] == id_]
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


        subject_y = subject_y[:, biomarker_idx]
        subject_y = subject_y.squeeze() 

        # print('Subject Y', subject_y.shape)
        subject_x.requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            f_preds = deepkernelmodel(subject_x)
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
    train_x = datasamples[datasamples['anon_id'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['anon_id'].isin(train_ids)]['Y']
    calibration_x = datasamples[datasamples['anon_id'].isin(calibration_ids)]['X']
    calibration_y = datasamples[datasamples['anon_id'].isin(calibration_ids)]['Y']

    corresponding_train_ids = datasamples[datasamples['anon_id'].isin(train_ids)]['anon_id'].to_list()
    corresponding_calibration_ids = datasamples[datasamples['anon_id'].isin(calibration_ids)]['anon_id'].to_list()

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

    #### DEFINE DKGP MODEL ####
    depth = [(train_x.shape[1], int(train_x.shape[1]/2))]
    dr = 0.2
    activ = 'relu'

    lat_dim = int(train_x.shape[1]/2)
    conformal_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    conformal_deepkernelmodel = SingleTaskDeepKernel(input_dim=train_x.shape[1], train_x=train_x, train_y=train_y, likelihood=likelihood, depth=depth, dropout=dr, activation=activ, kernel_choice=kernel, mean=mean,
        pretrained=False, feature_extractor=None, latent_dim=int(train_x.shape[1]/2), gphyper=None)
    
    if torch.cuda.is_available():
        conformal_likelihood = conformal_likelihood.cuda(gpuid)
        conformal_deepkernelmodel = conformal_deepkernelmodel.cuda(gpuid)

    # set up train mode
    conformal_deepkernelmodel.feature_extractor.train()
    conformal_deepkernelmodel.train()
    conformal_deepkernelmodel.likelihood.train()

   
    optimizer = torch.optim.Adam([
    {'params': conformal_deepkernelmodel.feature_extractor.parameters(), 'lr': 0.02},
    {'params': conformal_deepkernelmodel.covar_module.parameters(), 'lr': 0.02 },
    {'params': conformal_deepkernelmodel.mean_module.parameters(), 'lr': 0.02},
    {'params': conformal_deepkernelmodel.likelihood.parameters(),  'lr': 0.02} ], weight_decay=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(conformal_likelihood, conformal_deepkernelmodel)

    train_loss, val_loss = [], []
    for i in tqdm(range(50)): # it was 200 
        conformal_deepkernelmodel.train()
        conformal_likelihood.train()
        optimizer.zero_grad()
        output = conformal_deepkernelmodel(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        train_loss.append(loss.item())
        optimizer.step()

    # Set into eval mode
    conformal_deepkernelmodel.eval()
    conformal_likelihood.eval()

    print('Run Inference on Calibration Subjects')
    for id_ in calibration_ids:
        # print('Subject ID', id_)
        subject_data = datasamples[datasamples['anon_id'] == id_]
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
    conformity_scores_per_subject = {'id': [], 'conformal_scores': []}
    for subject in calibration_results_df['id'].unique():
        subject_df = calibration_results_df[calibration_results_df['id'] == subject]

        std = np.sqrt(subject_df['variance'])

        conformal_scores = np.abs(subject_df['score'] - subject_df['y'])/std #
        conformity_scores_per_subject['id'].append(subject)
        conformity_scores_per_subject['conformal_scores'].append(np.max(conformal_scores))


    conformity_scores_per_subject_df = pd.DataFrame(data=conformity_scores_per_subject)

    # gather all the conformity scores
    conformity_scores = np.array(conformity_scores_per_subject['conformal_scores'])
    sorted_conformity_scores = np.sort(conformity_scores)
    # n is the number of the validation subjects 
    n = conformity_scores.shape[0]
    conformal_alpha = alpha

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

    print('Conformalized Inference on Test Subjects')
    print('Run Inference on Test Subjects')
    for id_ in test_ids: 

        winkler_scores = []
        subject_data = datasamples[datasamples['anon_id'] == id_]
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

        std = variance.sqrt()
        conformal_lower = mean - qhat * std
        conformal_upper = mean + qhat * std

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

       
population_results_df = pd.DataFrame(data=population_results)
population_results_df.to_csv('./results/dkgp_'  + str(biomarker_idx) + '_results.csv')

conformalized_predictions_df = pd.DataFrame(data=conformalized_results)
conformalized_predictions_df.to_csv('./results/conformalized_dkgp_' + str(biomarker_idx) + '_results.csv')
# store the qhat values
qhat_df = pd.DataFrame(data=qhat_dict)
qhat_df.to_csv('./results/dkgp_qhat_' + str(biomarker_idx) + '.csv')
