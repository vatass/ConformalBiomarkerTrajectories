'''
Deep Quantile Regression for Randomly-timed Biomarker Trajectories
Conformal Deep Quantile Regression for Randomly-timed Biomarker Trajectories
'''
import numpy as np
import torch
import torch.nn as nn
import argparse
import time 
import pandas as pd
import pickle
import json
from functions import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

class DeepQuantileRegressor(nn.Module):
    def __init__(self, input_dim, output_quantiles):
        """
        Initializes the deep quantile regression model.
        Args:
            input_dim: Int, number of input features.
            output_quantiles: Int, number of quantiles to predict (e.g., 3 for 0.1, 0.5, 0.9).
        """
        super(DeepQuantileRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_quantiles)  # One output per quantile
        )

    def forward(self, x):
        return self.model(x)

def quantile_loss(preds, target, quantile):
    """
    Computes the quantile loss for quantile regression.
    Args:
        preds: Tensor, predicted values (shape: batch_size).
        target: Tensor, true target values (shape: batch_size).
        quantile: Float, the quantile to estimate (e.g., 0.1, 0.5, 0.9).
    Returns:
        Tensor: The quantile loss value.
    """
    error = target - preds
    return torch.mean(torch.max((quantile - 1) * error, quantile * error))

def train_deep_quantile_regressor(
    model, train_x, train_y, quantiles, epochs, learning_rate, gpuid=-1
):
    """
    Trains the deep quantile regression model.
    Args:
        model: DeepQuantileRegressor, the quantile regression model.
        train_x: Tensor, input training data.
        train_y: Tensor, target training data.
        quantiles: List of floats, quantiles to estimate (e.g., [0.1, 0.5, 0.9]).
        epochs: Int, number of training epochs.
        learning_rate: Float, learning rate for the optimizer.
        gpuid: Int, GPU device ID (-1 for CPU).
    """
    if gpuid >= 0 and torch.cuda.is_available():
        model = model.cuda(gpuid)
        train_x, train_y = train_x.cuda(gpuid), train_y.cuda(gpuid)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(train_x)  # Shape: (batch_size, num_quantiles)
        loss = sum(
            quantile_loss(preds[:, i], train_y, quantiles[i]) for i in range(len(quantiles))
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

def conformalized_inference_on_set(model, test_x, test_y, test_ids, test_time, kfold, results, z, alpha, study, gpuid=-1):
    """
    Tests the model and updates results into the specified dataframes.
    Args:
        model: DeepQuantileRegressor, the trained quantile regression model.
        test_x: Tensor, input test data.
        test_y: Tensor, true target values.
        test_ids: List or array, subject IDs corresponding to each sample.
        test_time: ndarray, time information for each sample (converted to NumPy array).
        kfold: Int, current fold index.
        calibration_results: Dict, stores sample-level results across folds.
        gpuid: Int, GPU device ID (-1 for CPU).
        z: Float, z-score for the confidence interval (default: 1.645 for 90% CI).
    """
    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_time = torch.tensor(test_time, device=device)  # Ensure test_time is on the same device

    model.eval()
    with torch.no_grad():
        preds = model(test_x)  # Shape: (num_samples, num_quantiles)

    test_ids = np.array(test_ids)

    # Group predictions by subject
    unique_ids = np.unique(test_ids)

    for subject_id in unique_ids:

        # Use NumPy array for masking
        subject_mask = (test_ids == subject_id)
        
        # Convert mask to PyTorch-compatible indexing
        subject_y = test_y[torch.tensor(subject_mask, device=device)]
        subject_preds = preds[torch.tensor(subject_mask, device=device)]
        subject_time = test_time[torch.tensor(subject_mask, device=device)]

        # Extract lower and upper bounds
        lower_bounds = subject_preds[:, 0].cpu().numpy()
        upper_bounds = subject_preds[:, 2].cpu().numpy()
        median_preds = subject_preds[:, 1].cpu().numpy()

        # Compute variance from quantile spread
        if alpha == 0.05: 
            ci = 1.96
        elif alpha == 0.1:
            ci = 1.645
        elif alpha == 0.01:
            ci = 2.576

        print(ci)
        uncertainty = ((upper_bounds - lower_bounds) / (2 * ci)) ** 2
        print('Qhat', z)
        print('Uncertainty', uncertainty)
        print('Standard Deviation', np.sqrt(uncertainty))
        print('Median Predictions', median_preds)
        conformal_lower_bounds =  median_preds - z * np.sqrt(uncertainty)
        conformal_upper_bounds = median_preds + z * np.sqrt(uncertainty)

        print('Conformal Lower Bounds', conformal_lower_bounds)
        print('Conformal Upper Bounds', conformal_upper_bounds)
        print('True Values', subject_y)

        print('Calculate the Subject Coverage')
        # 0 if the true value is not in the interval, 1 otherwise
        coverage = np.zeros(len(subject_y))
        for i in range(len(subject_y)):
            if conformal_lower_bounds[i] <= subject_y[i] <= conformal_upper_bounds[i]:
                coverage[i] = 1
        print('Coverage Rate', np.mean(coverage))


        # Calculate the Winkler Score 
        ### Calculate the Winkler Score for this Subject ### 
        winkler_scores = []
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = conformal_lower_bounds[i]
            u = conformal_upper_bounds[i]

            if l <= y_true <= u:
                winkler_score = u - l  # Width of the interval if y_true is within the interval
            elif y_true < l:
                winkler_score = (u - l) + (2 / alpha) * (l - y_true)  # Penalty for underprediction
            else:  # y_true > u
                winkler_score = (u - l) + (2 / alpha) * (y_true - u)  # Penalty for overprediction

            winkler_scores.append(winkler_score) 

        # Append to population results (per sample)
        for i in range(len(subject_y)):
            results['id'].append(subject_id)
            results['kfold'].append(kfold)
            results['score'].append(median_preds[i])
            results['lower'].append(conformal_lower_bounds[i].item())
            results['upper'].append(conformal_upper_bounds[i].item())
            results['variance'].append(uncertainty[i].item())
            results['y'].append(subject_y[i].item())
            results['time'].append(subject_time[i].item())
            ae = abs(subject_y[i].item() - median_preds[i])
            results['ae'].append(ae)
            results['winkler'].append(winkler_scores[i])
            results['study'].append(study)

    return results


def conformalize_with_train_sigma(model, test_x, test_y, test_ids, test_time, kfold, results, z, alpha, sigma, study, gpuid=-1):
    """
    Tests the model and updates results into the specified dataframes.
    Args:
        model: DeepQuantileRegressor, the trained quantile regression model.
        test_x: Tensor, input test data.
        test_y: Tensor, true target values.
        test_ids: List or array, subject IDs corresponding to each sample.
        test_time: ndarray, time information for each sample (converted to NumPy array).
        kfold: Int, current fold index.
        calibration_results: Dict, stores sample-level results across folds.
        gpuid: Int, GPU device ID (-1 for CPU).
        z: Float, z-score for the confidence interval (default: 1.645 for 90% CI).
    """
    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_time = torch.tensor(test_time, device=device)  # Ensure test_time is on the same device

    model.eval()
    with torch.no_grad():
        preds = model(test_x)  # Shape: (num_samples, num_quantiles)

    test_ids = np.array(test_ids)

    # Group predictions by subject
    unique_ids = np.unique(test_ids)

    for subject_id in unique_ids:

        # Use NumPy array for masking
        subject_mask = (test_ids == subject_id)
        
        # Convert mask to PyTorch-compatible indexing
        subject_y = test_y[torch.tensor(subject_mask, device=device)]
        subject_preds = preds[torch.tensor(subject_mask, device=device)]
        subject_time = test_time[torch.tensor(subject_mask, device=device)]

        median_preds = subject_preds[:, 1].cpu().numpy()
    
        conformal_lower_bounds =  median_preds - z * np.sqrt(sigma)
        conformal_upper_bounds = median_preds + z * np.sqrt(sigma)

        print('Conformal Lower Bounds', conformal_lower_bounds)
        print('Conformal Upper Bounds', conformal_upper_bounds)
        print('True Values', subject_y)

        print('Calculate the Subject Coverage')
        # 0 if the true value is not in the interval, 1 otherwise
        coverage = np.zeros(len(subject_y))
        for i in range(len(subject_y)):
            if conformal_lower_bounds[i] <= subject_y[i] <= conformal_upper_bounds[i]:
                coverage[i] = 1
        print('Coverage Rate', np.mean(coverage))


        # Calculate the Winkler Score 
        ### Calculate the Winkler Score for this Subject ### 
        winkler_scores = []
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = conformal_lower_bounds[i]
            u = conformal_upper_bounds[i]

            if l <= y_true <= u:
                winkler_score = u - l  # Width of the interval if y_true is within the interval
            elif y_true < l:
                winkler_score = (u - l) + (2 / alpha) * (l - y_true)  # Penalty for underprediction
            else:  # y_true > u
                winkler_score = (u - l) + (2 / alpha) * (y_true - u)  # Penalty for overprediction

            winkler_scores.append(winkler_score) 

        # Append to population results (per sample)
        for i in range(len(subject_y)):
            results['id'].append(subject_id)
            results['kfold'].append(kfold)
            results['score'].append(median_preds[i])
            results['lower'].append(conformal_lower_bounds[i].item())
            results['upper'].append(conformal_upper_bounds[i].item())
            results['variance'].append(sigma)
            results['y'].append(subject_y[i].item())
            results['time'].append(subject_time[i].item())
            ae = abs(subject_y[i].item() - median_preds[i])
            results['ae'].append(ae)
            results['winkler'].append(winkler_scores[i])
            results['study'].append(study)

    return results

def inference_on_set(model, test_x, test_y, test_ids, test_time, kfold, results, z, study, gpuid=-1):
    """
    Tests the model and updates results into the specified dataframes.
    Args:
        model: DeepQuantileRegressor, the trained quantile regression model.
        test_x: Tensor, input test data.
        test_y: Tensor, true target values.
        test_ids: List or array, subject IDs corresponding to each sample.
        test_time: ndarray, time information for each sample (converted to NumPy array).
        kfold: Int, current fold index.
        calibration_results: Dict, stores sample-level results across folds.
        gpuid: Int, GPU device ID (-1 for CPU).
        z: Float, z-score for the confidence interval (default: 1.645 for 90% CI).
    """
    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_time = torch.tensor(test_time, device=device)  # Ensure test_time is on the same device

    model.eval()
    with torch.no_grad():
        preds = model(test_x)  # Shape: (num_samples, num_quantiles)

    test_ids = np.array(test_ids)

    # Group predictions by subject
    unique_ids = np.unique(test_ids)

    for subject_id in unique_ids:

        # Use NumPy array for masking
        subject_mask = (test_ids == subject_id)
        
        # Convert mask to PyTorch-compatible indexing
        subject_y = test_y[torch.tensor(subject_mask, device=device)]
        subject_preds = preds[torch.tensor(subject_mask, device=device)]
        subject_time = test_time[torch.tensor(subject_mask, device=device)]

        # Extract lower and upper bounds
        lower_bounds = subject_preds[:, 0]
        upper_bounds = subject_preds[:, 2]
        median_preds = subject_preds[:, 1].cpu().numpy()

        # Compute variance from quantile spread
        variance = ((upper_bounds - lower_bounds) / (2 * z)) ** 2

        # Calculate the Winkler Score 
        ### Calculate the Winkler Score for this Subject ### 
    
        winkler_scores = []
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = lower_bounds[i].cpu().detach().numpy()
            u = upper_bounds[i].cpu().detach().numpy()

            if l <= y_true <= u:
                winkler_score = u - l  # Width of the interval if y_true is within the interval
            elif y_true < l:
                winkler_score = (u - l) + (2 / alpha) * (l - y_true)  # Penalty for underprediction
            else:  # y_true > u
                winkler_score = (u - l) + (2 / alpha) * (y_true - u)  # Penalty for overprediction

            winkler_scores.append(winkler_score) 

        # Append to population results (per sample)
        for i in range(len(subject_y)):
            results['id'].append(subject_id)
            results['kfold'].append(kfold)
            results['score'].append(median_preds[i])
            results['lower'].append(lower_bounds[i].item())
            results['upper'].append(upper_bounds[i].item())
            results['variance'].append(variance[i].item())
            results['y'].append(subject_y[i].item())
            results['time'].append(subject_time[i].item())
            ae = abs(subject_y[i].item() - median_preds[i])
            results['ae'].append(ae)
            results['winkler'].append(winkler_scores[i])
            results['study'].append(study)

    return results

def test_and_update_results(
    model, test_x, test_y, test_ids, test_time, kfold, population_results, gpuid=-1, z=1.645
):
    """
    Tests the model and updates results into the specified dataframes.
    Args:
        model: DeepQuantileRegressor, the trained quantile regression model.
        test_x: Tensor, input test data.
        test_y: Tensor, true target values.
        test_ids: List or array, subject IDs corresponding to each sample.
        test_time: ndarray, time information for each sample (converted to NumPy array).
        kfold: Int, current fold index.
        population_results: Dict, stores sample-level results across folds.
        population_mae_kfold: Dict, stores fold-level metrics across folds.
        population_metrics_per_subject: Dict, stores subject-level metrics across folds.
        gpuid: Int, GPU device ID (-1 for CPU).
        z: Float, z-score for the confidence interval (default: 1.645 for 90% CI).
    """
    device = torch.device(f"cuda:{gpuid}" if gpuid >= 0 and torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_time = torch.tensor(test_time, device=device)  # Ensure test_time is on the same device

    model.eval()
    with torch.no_grad():
        preds = model(test_x)  # Shape: (num_samples, num_quantiles)

    # Per-sample metrics
    ae_list = []
    interval_size_list = []

    test_ids = np.array(test_ids)

    # Group predictions by subject
    unique_ids = np.unique(test_ids)

    for subject_id in unique_ids:

        # Use NumPy array for masking
        subject_mask = (test_ids == subject_id)
        
        # Convert mask to PyTorch-compatible indexing
        subject_y = test_y[torch.tensor(subject_mask, device=device)]
        subject_preds = preds[torch.tensor(subject_mask, device=device)]
        subject_time = test_time[torch.tensor(subject_mask, device=device)]

        # Extract lower and upper bounds
        lower_bounds = subject_preds[:, 0]
        upper_bounds = subject_preds[:, 2]
        median_preds = subject_preds[:, 1].cpu().numpy()

        # Compute variance from quantile spread
        variance = ((upper_bounds - lower_bounds) / (2 * z)) ** 2


        # Calculate the Winkler Score
        winkler_scores = []
        winkler_score = 0
        for i in range(len(subject_y)):
            y_true = subject_y[i].cpu().detach().numpy()
            l = lower_bounds[i].cpu().detach().numpy()
            u = upper_bounds[i].cpu().detach().numpy()

            if l <= y_true <= u:
                winkler_score = u - l
            elif y_true < l:
                winkler_score = (u - l) + (2 / alpha) * (l - y_true)
            else:  # y_true > u
                winkler_score = (u - l) + (2 / alpha) * (y_true - u)

            winkler_scores.append(winkler_score)

        # Append to population results (per sample)
        for i in range(len(subject_y)):
            population_results['id'].append(subject_id)
            population_results['kfold'].append(kfold)
            population_results['score'].append(median_preds[i])
            population_results['lower'].append(lower_bounds[i].item())
            population_results['upper'].append(upper_bounds[i].item())
            population_results['variance'].append(variance[i].item())
            population_results['y'].append(subject_y[i].item())
            population_results['time'].append(subject_time[i].item())
            ae = abs(subject_y[i].item() - median_preds[i])
            population_results['ae'].append(ae)
            population_results['winkler'].append(winkler_scores[i])
            ae_list.append(ae)
            interval_size_list.append((upper_bounds[i] - lower_bounds[i]).item())


parser = argparse.ArgumentParser(description='Deep Quantile Regression for Longitudinal Data')
## Data Parameters 
parser.add_argument("--gpuid", help="GPUs", default=0)
parser.add_argument("--file", help="Identifier for the data", default="./data/data.csv")
parser.add_argument("--biomarker_idx", type=int, default=14)
## Conformal Prediction Parameters 
parser.add_argument("--alpha", help='Significance Level', type=float, default=0.1)
parser.add_argument("--calibrationset", help='Size of the Calibration Set', type=float, default=0.04)

population_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': []  }
conformal_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': [], 'study': [] }

qhat_dict = {'qhat': [], 'calibration_set_size': [], 'fold': [] }

t0= time.time()
args = parser.parse_args()
gpuid = int(args.gpuid)
file = args.file
biomarker_idx = args.biomarker_idx
list_index = biomarker_idx
# Conrformal Prediction Parameters
alpha = float(args.alpha)
calibrationset = args.calibrationset

datasamples = pd.read_csv(file)


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

    ### Check if there is negative time
    time_ = train_x[:, -1].cpu().detach().numpy().tolist()

    print('Train Time', len(time_))
    print('Train Subjectes', len(corresponding_train_ids))

    test_y = test_y[:, list_index]
    train_y = train_y[:, list_index]

    train_y = train_y.squeeze() 
    test_y = test_y.squeeze()

    #### Define the Quantile Regressor ####
    input_dim = train_x.shape[1]  # Number of input features
    # quantiles = [0.1, 0.5, 0.9]  # Desired quantiles

    if alpha == 0.05: 
        quantiles = [0.05, 0.5, 0.95]
        z = 1.96
    elif alpha == 0.1:
        quantiles = [0.1, 0.5, 0.9]
        z = 1.645
    elif alpha == 0.01:
        quantiles = [0.01, 0.5, 0.99]
        z = 2.576

    output_quantiles = len(quantiles)
    gpuid = 0  # Use -1 for CPU

    # Initialize model
    model = DeepQuantileRegressor(input_dim, output_quantiles)

    # Train the model
    print("Starting training...")
    train_deep_quantile_regressor(
        model=model,
        train_x=train_x,
        train_y=train_y,
        quantiles=quantiles,
        epochs=50,
        learning_rate=0.01,
        gpuid=gpuid
    )
    print("Training completed.")

    test_time = np.array(test_x[:, -1].cpu().detach().numpy())  # Convert directly to NumPy array
    test_ids = corresponding_test_ids
    # convert test_ids to str
    test_ids = [str(i) for i in test_ids]

    # Test and update results
    test_and_update_results(
        model=model,
        test_x=test_x,
        test_y=test_y,
        test_ids=test_ids,
        test_time=test_time,
        kfold=fold,
        population_results=population_results,
        gpuid=gpuid, 
        z=z
    )

    ### Split the train data into train/calibration set ###
    # convert to float
    conformal_split_percentage = float(calibrationset)

    # random selection of the calibration subjects! 
    print('Random Selection of Calibration Set')
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

    ### Train the Conformalized Quantile Regressor ###
    # Initialize model
    conformal_model = DeepQuantileRegressor(input_dim, output_quantiles)

    # Train the model
    print("Starting training...")
    train_deep_quantile_regressor(
        model=conformal_model,
        train_x=train_x,
        train_y=train_y,
        quantiles=quantiles,
        epochs=50,
        learning_rate=0.01,
        gpuid=gpuid
    )
    print("Conformal Quantile Regressor: Training completed.")

    ### Calculate the Non-Conformity Scores ###
    print('Run Inference on Calibration Subjects')
    calibration_time = np.array(calibration_x[:, -1].cpu().detach().numpy())  # Convert directly to NumPy array
    calibration_ids = corresponding_calibration_ids
    # convert test_ids to str
    calibration_ids = [str(i) for i in calibration_ids]
    calibration_results = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'time': [], 'ae': [], 'winkler': [], 'study': [] }
    # Test and update results
    inference_on_set(
        model=conformal_model,
        test_x=calibration_x,
        test_y=calibration_y,
        test_ids=calibration_ids,
        test_time=calibration_time,
        kfold=fold,
        results=calibration_results,
        gpuid=gpuid, 
        z=z, 
        study = 'sample_study_calibration'
    )
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

    ### Test the Conformalized Quantile Regressor ###
    print('Run Inference for the Conformalized Quantile Regressor on Test Subjects')
    test_time = np.array(test_x[:, -1].cpu().detach().numpy())  # Convert directly to NumPy array   
    test_ids = corresponding_test_ids
    # convert test_ids to str
    test_ids = [str(i) for i in test_ids]

    # Test and update results
    print('Conformalize with Qhat', qhat, 'for alpha', alpha)
    conformal_results = conformalized_inference_on_set(
        model=conformal_model,
        test_x=test_x,
        test_y=test_y,
        test_ids=test_ids,
        test_time=test_time,
        kfold=fold,
        results=conformal_results,
        gpuid=gpuid, 
        alpha=alpha,
        z=qhat, 
        study='sample_study'
    )


# Convert dictionaries to DataFrames
population_results_df = pd.DataFrame(population_results)
conformal_results_df = pd.DataFrame(conformal_results)

# Save results to CSV
qhat_df = pd.DataFrame(qhat_dict)
qhat_df.to_csv('./results/qhat_quantile_'+str(biomarker_idx)+'.csv', index=False)
population_results_df.to_csv("./results/deep_quantile_regression_"+str(biomarker_idx)+"_results.csv", index=False)
conformal_results_df.to_csv("./results/conformalized_deep_quantile_regression_"+str(biomarker_idx)+"_results.csv", index=False)

def calculate_subject_level_metrics(df):
    df['interval_width'] = df['upper'] - df['lower']
    grouped = df.groupby(['id', 'kfold']).apply(lambda group: pd.Series({
        'id': group['id'].iloc[0],
        'kfold': group['kfold'].iloc[0],
        'coverage': int((group['lower'] <= group['y']).all() & (group['y'] <= group['upper']).all()),
        'interval_width': group['interval_width'].mean()
    })).reset_index(drop=True)
    return grouped

# Calculate mean and confidence intervals for each fold
def calculate_fold_metrics(results, method_name):
    grouped = results.groupby('kfold').agg(
        mean_coverage=('coverage', 'mean'),
        mean_interval_width=('interval_width', 'mean'),
        coverage_ci=('coverage', lambda x: 1.96 * np.std(x) / np.sqrt(len(x))),
        width_ci=('interval_width', lambda x: 1.96 * np.std(x) / np.sqrt(len(x))),
    ).reset_index()
    grouped['method'] = method_name
    return grouped

dqr_results = calculate_subject_level_metrics(population_results_df)
dqr_conformalized_results = calculate_subject_level_metrics(conformal_results_df)

print('DQR Results')
print(dqr_results.head())
print('DQR Conformalized Results')
print(dqr_conformalized_results.head())