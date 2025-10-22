'''
Group Conformal Prediction - DKGP Predictor for Diagnosis Covariate
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
from tqdm import tqdm

from functions import process_temporal_singletask_data
from models import SingleTaskDeepKernel

##### SET RANDOM SEED ####
# Set random seeds
seed = 123
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
parser.add_argument("--file", help="Identifier for the data", default="./data/data.csv")
parser.add_argument("--calibrationset", help="Percentage of spliting Population/Calibration", default=0.2)
parser.add_argument("--alpha", help="Significance Level", default=0.1)
parser.add_argument("--biomarker_idx", type=int, default=14)


t0= time.time()
args = parser.parse_args()
gpuid = int(args.gpuid)
file = args.file
biomarker_idx = args.biomarker_idx
alpha = float(args.alpha)  
calibrationset = args.calibrationset

calibration_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': []}
population_conformalized_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': [], 'covariate': [] }
group_conditional_conformalized_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': [] }

qhat_dict = {'qhat': [], 'calibration_set_size': [], 'fold': [], 'covariate': []}
unstrat_qhat_dict = {'qhat': [], 'calibration_set_size': [], 'fold': []}

datasamples = pd.read_csv(file)    


longitudinal_covariates = pd.read_csv('./data/anonymized_covariates.csv')
longitudinal_covariates['Diagnosis'].replace(
    [-1.0, 0.0, 1.0, 2.0], ['UKN', 'CN', 'MCI', 'AD'], inplace=True
)

# Define valid options for each covariate
valid_values = {
    'Diagnosis': {'CN', 'MCI', 'AD'},
}

list_index = biomarker_idx 

for fold in range(10): 
    # print('FOLD::', fold)
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

    # print('Train the Deep Kernel Regressor')
    # print('Train IDs', len(train_ids))
    # print('Test IDs', len(test_ids))

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

    ### Check if there is negative time
    time_ = train_x[:, -1].cpu().detach().numpy().tolist()

    # print('Train Time', len(time_))
    # print('Train Subjectes', len(corresponding_train_ids))

    test_y = test_y[:, list_index]
    train_y = train_y[:, list_index]

    train_y = train_y.squeeze() 
    test_y = test_y.squeeze()

    # print('Split the Train Data into Train and Calibration')
    calibrationset = args.calibrationset
    # convert to float
    calibrationset = float(calibrationset)

    # Random Seletion of Calibration Subjects!
    calibration_ids = np.random.choice(train_ids, int(calibrationset*len(train_ids)), replace=False)
    train_ids = [x for x in train_ids if x not in calibration_ids]

    # print('Train IDs', len(train_ids))
    # print('Calibration IDs', len(calibration_ids))
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
    depth = [(train_x.shape[1], int(train_x.shape[1]/2) )]
    dr = 0.2
    activ = 'relu'
    kernel = 'RBF'
    mean = 'Constant'
    ###########################

    lat_dim = int(train_x.shape[1]/2)
    conformal_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    conformal_deepkernelmodel = SingleTaskDeepKernel(input_dim=train_x.shape[1], train_x=train_x, train_y=train_y, likelihood=conformal_likelihood, depth=depth, dropout=dr, activation=activ, kernel_choice=kernel, mean=mean,
        pretrained=False, feature_extractor=None, latent_dim=int(train_x.shape[1]/2), gphyper=None)
    
    if torch.cuda.is_available():
        conformal_likelihood = conformal_likelihood.cuda(gpuid)
        conformal_deepkernelmodel = conformal_deepkernelmodel.cuda(gpuid)

    training_iterations  =  100

    # set up train mode
    conformal_deepkernelmodel.feature_extractor.train()
    conformal_deepkernelmodel.train()
    conformal_deepkernelmodel.likelihood.train()

    optimizer = torch.optim.Adam([
    {'params': conformal_deepkernelmodel.feature_extractor.parameters(), 'lr': 0.02},
    {'params': conformal_deepkernelmodel.covar_module.parameters(), 'lr': 0.02 },
    {'params': conformal_deepkernelmodel.mean_module.parameters(), 'lr': 0.02},
    {'params': conformal_deepkernelmodel.likelihood.parameters(),  'lr': 0.02} ], weight_decay=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(conformal_likelihood, conformal_deepkernelmodel)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20, 50, 70], gamma=0.1)

    train_loss, val_loss = [], []

    for i in tqdm(range(training_iterations)):
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

    # print('Run Inference on Calibration Subjects')
    for id_ in calibration_ids:
        # print('Subject ID', id_)
        subject_data = datasamples[datasamples['anon_id'] == id_]
        subject_covariates = longitudinal_covariates[longitudinal_covariates['anon_id'] == id_]
        subject_x = subject_data['X']
        subject_y = subject_data['Y']

        subject_dict = {'y': [], 'score': [], 'id': [], 'time': []}

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
            lower, upper = y_preds.confidence_region()

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

    # print('Calculate the Non-Conformity Scores')
    conformity_scores_per_subject = {'id': [], 'conformal_scores': []}
    for subject in calibration_results_df['id'].unique():
        subject_df = calibration_results_df[calibration_results_df['id'] == subject]
        std = np.sqrt(subject_df['variance'])

        conformal_scores = np.abs(subject_df['score'] - subject_df['y'])/std #
        nonnorm = np.abs(subject_df['score'] - subject_df['y'])
        conformity_scores_per_subject['id'].append(subject)
        conformity_scores_per_subject['conformal_scores'].append(np.max(conformal_scores))

    conformity_scores_per_subject_df = pd.DataFrame(data=conformity_scores_per_subject)
    
    ### Calculate the Total Conformity Scores in the Unstratified Calibration Set ####
    # print('Store the Conformity Scores in the unstratified calibration set')
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
    unstrat_qhat = sorted_conformity_scores[k-1]
    # print('Qhat', unstrat_qhat)
    # print('Calibration Set Size',len(calibration_ids))

    unstrat_qhat_dict['qhat'].append(unstrat_qhat)
    unstrat_qhat_dict['calibration_set_size'].append(len(calibration_results_df['id'].unique()))
    unstrat_qhat_dict['fold'].append(fold)

    # print('Stratify Conformity Scores per Covariate')
    # Merge datasets
    merged_data = conformity_scores_per_subject_df.merge(
        longitudinal_covariates, left_on='id', right_on='anon_id', how='inner'
    )

    covariates = ['Diagnosis']
    for c in covariates: 
        # print('Covariate', c)
        # split the merged data into the covariate of interest
        covariate_unique_values = merged_data[c].unique()
        
        for cu in covariate_unique_values:

            if c == 'Diagnosis': 
                if cu not in ['CN', 'MCI', 'AD']:
                    continue

            # print('Covariate Value', cu)
            # print(type(cu))
                
            covariate_df = merged_data[merged_data[c] == cu]
            # print('Covariate DF', covariate_df.shape)

            conformity_scores = np.array(covariate_df['conformal_scores'])
            sorted_conformity_scores = np.sort(conformity_scores)
            # n is the number of the validation subjects 
            n = conformity_scores.shape[0]
            conformal_alpha = alpha


            k = int(np.ceil((n + 1) * (1 - conformal_alpha)))
            # Ensure k does not exceed n
            k = min(k, n)
            # Get the (n - k + 1)-th smallest value since we want the k-th largest value
            qhat = sorted_conformity_scores[k-1]
 
            qhat_dict['qhat'].append(qhat)
            qhat_dict['covariate'].append(cu)            
            qhat_dict['calibration_set_size'].append(len(covariate_df['id'].unique()))
            qhat_dict['fold'].append(fold)

    # print('Conformalized Inference on Test Subjects')
    # print('Run Inference on Test Subjects')

    # keep the test ids that belong to the longitudinal covariates
    test_ids = [x for x in test_ids if x in longitudinal_covariates['anon_id'].unique()]

    for id_ in test_ids: 
        # print('Subject ID', id_)
        winkler_scores = []
        subject_data = datasamples[datasamples['anon_id'] == id_]
        subject_covariates = longitudinal_covariates[longitudinal_covariates['anon_id'] == id_]
        # print('Subject Data', subject_data.shape)
        # print('Subject Covariates', subject_covariates.shape)
        # assert that subject_data has the same shape as subject_covariates
        # assert subject_data.shape[0] == subject_covariates.shape[0]
        
        subject_x = subject_data['X']
        subject_y = subject_data['Y']

        subject_dict = {'y': [], 'score': [], 'id': [], 'time': []}

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

        # Fetch the correct qhat value based on the covariate and perform stratified conformal inference
        # print('Test Subject ID:', id_)
        for c in covariates: 
            covariate_value = subject_covariates[c].values[0]
            # print(f"Processing covariate: {c}, Value: {covariate_value} (Type: {type(covariate_value)})")
            
            # Skip invalid or irrelevant covariate values
            if isinstance(covariate_value, float) or covariate_value in {'UKN', 'Unknown', 'other', 'More than one'}:
                continue

            # Define valid options for each covariate
            valid_values = {
                'Baseline_Diagnosis': {'CN', 'MCI', 'AD'}}

            # Check if covariate value is valid
            if c in valid_values and covariate_value not in valid_values[c]:
                continue

            # Fetch qhat value for the covariate
            try:
                qhat_strat = qhat_dict['qhat'][qhat_dict['covariate'].index(covariate_value)]
            except ValueError:
                # print(f"Qhat value not found for covariate: {c}, Value: {covariate_value}")
                continue

            # print('Qhat:', qhat_strat)
            # print('Covariate Value:', covariate_value)

            # Calculate conformal intervals
            std = variance.sqrt()
            conformal_lower = mean - qhat_strat * std
            conformal_upper = mean + qhat_strat * std

            # Calculate the Winkler Score for this Subject
            winkler_scores = []
            for i in range(len(subject_y)):
                y_true = subject_y[i].cpu().detach().numpy()
                l = conformal_lower[i].cpu().detach().numpy()
                u = conformal_upper[i].cpu().detach().numpy()

                if l <= y_true <= u:
                    winkler_score = u - l  # Width of the interval
                elif y_true < l:
                    winkler_score = (u - l) + (2 / conformal_alpha) * (l - y_true)  # Underprediction penalty
                else:  # y_true > u
                    winkler_score = (u - l) + (2 / conformal_alpha) * (y_true - u)  # Overprediction penalty

                winkler_scores.append(winkler_score)

            # print('Storing the Results!')

            # Store the results in the dictionary
            population_conformalized_results['id'].extend([id_] * subject_x.shape[0])
            population_conformalized_results['kfold'].extend([fold] * subject_x.shape[0])
            population_conformalized_results['score'].extend(mean.cpu().detach().numpy().tolist())
            population_conformalized_results['lower'].extend(conformal_lower.cpu().detach().numpy().tolist())
            population_conformalized_results['upper'].extend(conformal_upper.cpu().detach().numpy().tolist())
            population_conformalized_results['y'].extend(subject_y.cpu().detach().numpy().tolist())
            population_conformalized_results['variance'].extend(variance.cpu().detach().numpy().tolist())
            population_conformalized_results['time'].extend(subject_x[:, -1].cpu().detach().numpy().tolist())

            ae = np.abs(mean.cpu().detach().numpy() - subject_y.cpu().detach().numpy())
            population_conformalized_results['ae'].extend(ae.tolist())
            population_conformalized_results['winkler'].extend(winkler_scores)
            population_conformalized_results['covariate'].extend([c] * subject_x.shape[0])

            # unstratified conformalized inference
            unstrat_conf_lower = mean - unstrat_qhat * std
            unstrat_conf_upper = mean + unstrat_qhat * std

            group_conditional_conformalized_results['id'].extend([id_] * subject_x.shape[0])
            group_conditional_conformalized_results['kfold'].extend([fold] * subject_x.shape[0])
            group_conditional_conformalized_results['score'].extend(mean.cpu().detach().numpy().tolist())
            group_conditional_conformalized_results['lower'].extend(unstrat_conf_lower.cpu().detach().numpy().tolist())
            group_conditional_conformalized_results['upper'].extend(unstrat_conf_upper.cpu().detach().numpy().tolist())
            group_conditional_conformalized_results['y'].extend(subject_y.cpu().detach().numpy().tolist())
            group_conditional_conformalized_results['variance'].extend(variance.cpu().detach().numpy().tolist())
            group_conditional_conformalized_results['time'].extend(subject_x[:, -1].cpu().detach().numpy().tolist())
            group_conditional_conformalized_results['ae'].extend(ae.tolist())
            group_conditional_conformalized_results['winkler'].extend(winkler_scores)

conformalized_predictions_df = pd.DataFrame(data=population_conformalized_results)
conformalized_predictions_df.to_csv('./results/group_conditional_' + str(list_index)  + '_results_calibrationset_' + str(calibrationset) + '_alpha_' + str(alpha) + '.csv')

group_conditional_conformalized_predictions_df = pd.DataFrame(data=group_conditional_conformalized_results)
group_conditional_conformalized_predictions_df.to_csv('./results/population_cp_' + str(list_index)  + '_results_calibrationset_' + str(calibrationset) + '_alpha_' + str(alpha) + '.csv')

# store the qhat values
qhat_df = pd.DataFrame(data=qhat_dict)
qhat_df.to_csv('./results/group_conditional_qhat_'+ str(list_index) +'_calibrationset_' + str(calibrationset) + '_alpha_' + str(alpha) + '.csv')

unstrat_qhat_df = pd.DataFrame(data=unstrat_qhat_dict)
unstrat_qhat_df.to_csv('./results/population_cp_qhat_' + str(list_index)  + '_calibrationset_' + str(calibrationset) + '_alpha_' + str(alpha) + '.csv')

t1 = time.time() - t0 
