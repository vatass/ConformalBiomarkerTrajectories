'''
Visualization Script for NeurIPS Paper
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import json 
import os 
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.cm as cm


os.makedirs('./figures', exist_ok=True)

parser = argparse.ArgumentParser(description='Generate plots for NeurIPS paper')
parser.add_argument('--r', type=int, default=14, help='Biomarker Index to Plot')
parser.add_argument('--cs', type=float, default=0.2, help='Calibration Split Percentage')
parser.add_argument('--alpha', type=float, default=0.1, help='Confidence Level Alpha')
args = parser.parse_args() 

r = args.r
cs = args.cs
alpha = args.alpha


r_name = {14: 'Hippocampal L', 17: 'Ventricular'} 

fontsize_ticks = 18 
fontsize_labels = 18 
fontsize_legend = 18 


##### UTILS ######
# Create the status column
def determine_status(diagnosis_list):
    if diagnosis_list[0] == 'CN' and diagnosis_list[-1] == 'CN':
        return 'Healthy Control'
    elif diagnosis_list[0] == 'CN' and diagnosis_list[-1] == 'MCI':
        return 'MCI Progressor'
    elif diagnosis_list[0] == 'CN' and diagnosis_list[-1] == 'AD':
        return 'AD Progressor from CN'
    elif diagnosis_list[0] == 'MCI' and diagnosis_list[-1] == 'AD':
        return 'AD Progressor from MCI'
    elif diagnosis_list[0] == 'MCI' and diagnosis_list[-1] == 'MCI':
        return 'MCI Stable'
    elif diagnosis_list[0] == 'AD' and diagnosis_list[-1] == 'AD':
        return 'AD'
    else:
        return 'UKN'
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

def calculate_subject_level_metrics(df):
    df['interval_width'] = df['upper'] - df['lower']
    grouped = df.groupby(['id', 'kfold']).apply(lambda group: pd.Series({
        'id': group['id'].iloc[0],
        'kfold': group['kfold'].iloc[0],
        'coverage': int((group['lower'] <= group['y']).all() & (group['y'] <= group['upper']).all()),
        'interval_width': group['interval_width'].mean()
    })).reset_index(drop=True)
    return grouped

def calculate_subject_level_metrics_per_covariate(df):
    df['interval_width'] = df['upper'] - df['lower']
    grouped = df.groupby(['id', 'kfold', 'covariate']).apply(lambda group: pd.Series({
        'id': group['id'].iloc[0],
        'kfold': group['kfold'].iloc[0],
        'covariate': group['covariate'].iloc[0],
        'coverage': int((group['lower'] <= group['y']).all() & (group['y'] <= group['upper']).all()),
        'interval_width': group['interval_width'].mean()
    })).reset_index(drop=True)
    return grouped

# Function to calculate confidence intervals
def calculate_ci(data):
    mean = data.mean()
    std = data.std()
    n = len(data)
    ci = 1.96 * (std / np.sqrt(n))  # 95% confidence interval
    return mean, ci

# SHARED STYLE 
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi": 300,
    "text.usetex": True,
})

print('Figure 1: Qualitative Trajectories.')

before_dkgp = pd.read_csv(f'./results/dkgp_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')
conformalized_dkgp = pd.read_csv(f'./results/conformalized_dkgp_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

before_dkgp['method'] = 'DKGP'
conformalized_dkgp['method'] = 'CP-DKGP'

# deep quantile regression 
before_dqr = pd.read_csv(f'./results/deep_quantile_regression_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')
conformalized_dqr= pd.read_csv(f'./results/conformalized_deep_quantile_regression_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

before_dqr['method'] = 'DQR'
conformalized_dqr['method'] = 'CP-DQR'

# deep regression with monte carlo dropout
before_drmc = pd.read_csv(f'./results/drmc_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')
conformalized_drmc= pd.read_csv(f'./results/conformalized_drmc_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

before_drmc['method'] = 'DRMC'
conformalized_drmc['method'] = 'CP-DRMC'

before_bootstrap = pd.read_csv(f'./results/deep_regression_bootstrap_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')
conformalized_bootstrap = pd.read_csv(f'./results/conformalized_deep_regression_bootstrap_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

before_bootstrap['method'] = 'Bootstrap'
conformalized_bootstrap['method'] = 'CP-Bootstrap'

combined_dkgp = pd.concat([before_dkgp, conformalized_dkgp], ignore_index=True)
combined_dqr = pd.concat([before_dqr, conformalized_dqr], ignore_index=True)
combined_drmc = pd.concat([before_drmc, conformalized_drmc], ignore_index=True)
combined_bootstrap = pd.concat([before_bootstrap, conformalized_bootstrap], ignore_index=True)

combined_all = pd.concat([combined_dkgp, combined_dqr, combined_drmc, combined_bootstrap], ignore_index=True)

longitudinal_covariates = pd.read_csv(f'./data/anonymized_covariates.csv')

subjects_to_plot = combined_all['id'].unique()[:10]

for s in subjects_to_plot:
    subject_data = combined_all[combined_all['id'] == s]


    # extract the bounds before for all the 5 methods  and plot them in the first subplot 
    before_dkgp = subject_data[subject_data['method'] == 'DKGP']

    if before_dkgp.shape[0] < 6: 
        continue

    diagnosis = longitudinal_covariates[longitudinal_covariates['anon_id'] == s]['Diagnosis'].tolist()

    before_dqr = subject_data[subject_data['method'] == 'DQR']
    before_drmc = subject_data[subject_data['method'] == 'DRMC']
    before_bootstrap = subject_data[subject_data['method'] == 'Bootstrap']
    
    after_dkgp = subject_data[subject_data['method'] == 'CP-DKGP']
    after_dqr = subject_data[subject_data['method'] == 'CP-DQR']
    after_drmc = subject_data[subject_data['method'] == 'CP-DRMC']
    after_bootstrap = subject_data[subject_data['method'] == 'CP-Bootstrap']
    
    # create figur
    ground_truth_color = '#264653'  # Deep blue-gray for ground truthe

    # Define colors for each methodBlue – #0072B2
    dkgp_color = '#0072B2'
    dqr_color = '#E69F00'
    drmc_color = '#009E73'
    bootstrap_color = '#D55E00'
    dme_color = '#CC79A7'

    ### Altervative plot for the two methods 
    fig, ax = plt.subplots(figsize=(9, 6))
    # Updated DKGP colors
    dkgp_baseline_color = '#89a6c7'  # lighter/muted blue
    dkgp_conformal_color = '#1f77b4'  # strong distinct blue


    # DKGP Baseline with hatch
    ax.fill_between(
        before_dkgp['time'],
        before_dkgp['lower'],
        before_dkgp['upper'],
        facecolor='none',
        edgecolor=dkgp_baseline_color,
        hatch='////',
        linewidth=1,
        label='DKGP'
    )

    # DKGP Conformal with solid fill
    ax.fill_between(
        after_dkgp['time'],
        after_dkgp['lower'],
        after_dkgp['upper'],
        alpha=0.3,
        color=dkgp_conformal_color,
        label='CP-DKGP'
    )

    # DRMC Colors
    drmc_baseline_color = '#c97b8e'   # lighter muted orange
    drmc_conformal_color = '#7b1e3c'  # vivid reddish-orange

    # DRMC Baseline with hatch
    ax.fill_between(
        before_drmc['time'],
        before_drmc['lower'],
        before_drmc['upper'],
        facecolor='none',
        edgecolor=drmc_baseline_color,
        hatch='\\\\',
        linewidth=2,
        label='DRMC'
    )

    # DRMC Conformal with solid fill
    ax.fill_between(
        after_drmc['time'],
        after_drmc['lower'],
        after_drmc['upper'],
        alpha=0.2,
        color=drmc_conformal_color,
        label='CP-DRMC'
    )

    # Ground truth trajectory
    ax.plot(
        before_dkgp['time'],
        before_dkgp['y'],
        color=ground_truth_color,
        linewidth=2,
        linestyle='-',
        label='True Trajectory'
    )
    ax.scatter(
        before_dkgp['time'],
        before_dkgp['y'],
        color=ground_truth_color,
        s=70,
        zorder=3
    )

    ax.set_title(f'Predictive Bounds Before and After Conformalization', fontsize=22)
    ax.set_xlabel('Time (in months)', fontsize=22)
    ax.set_ylabel('Hippocampal Volume (standardized)', fontsize=22)
    ax.set_ylim(-2, 2)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Custom legend to show fill and hatch
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=18)

    plt.tight_layout()
    plt.savefig(f'./figures/trajectories_{s}_{r}_{alpha}_{cs}.png', format='png', bbox_inches='tight')

print('Figure 2: Comparison before and after the CP')
# Load Bayesian and Conformal datasets
conformalized_bounds = pd.read_csv(f'./results/conformalized_dkgp_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')
bayesian_bounds = pd.read_csv(f'./results/dkgp_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

# deep quantile regression 
dqr = pd.read_csv(f'./results/deep_quantile_regression_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')
conformalized_dqr= pd.read_csv(f'./results/conformalized_deep_quantile_regression_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

# deep regression with monte carlo dropout
drmc = pd.read_csv(f'./results/drmc_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')
conformalized_drmc= pd.read_csv(f'./results/conformalized_drmc_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

bootstrap = pd.read_csv(f'./results/deep_regression_bootstrap_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')
conf_bootstrap = pd.read_csv(f'./results/conformalized_deep_regression_bootstrap_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

# CalCulate metrics
bayesian_results = calculate_subject_level_metrics(bayesian_bounds)
conformalized_results = calculate_subject_level_metrics(conformalized_bounds)

dqr_results = calculate_subject_level_metrics(dqr)
dqr_conformalized_results = calculate_subject_level_metrics(conformalized_dqr)

drmc_results = calculate_subject_level_metrics(drmc)
drmc_conformalized_results = calculate_subject_level_metrics(conformalized_drmc)

bootstrap_results = calculate_subject_level_metrics(bootstrap)
bootstrap_conformalized_results = calculate_subject_level_metrics(conf_bootstrap)

# Calculate fold metrics
bayesian_fold_metrics = calculate_fold_metrics(bayesian_results, 'DKGP')
conformalized_fold_metrics = calculate_fold_metrics(conformalized_results, 'CP-DKGP')

dqr_fold_metrics = calculate_fold_metrics(dqr_results, 'DQR')
dqr_conformalized_fold_metrics = calculate_fold_metrics(dqr_conformalized_results, 'CP-DQR')

drmc_fold_metrics = calculate_fold_metrics(drmc_results, 'DRMC')
drmc_conformalized_fold_metrics = calculate_fold_metrics(drmc_conformalized_results, 'CP-DRMC')

bootstrap_fold_metrics = calculate_fold_metrics(bootstrap_results, 'Bootstrap')
bootstrap_conformalized_fold_metrics = calculate_fold_metrics(bootstrap_conformalized_results, 'CP-Bootstrap')


# Combine results for plotting
combined_metrics = pd.concat([bayesian_fold_metrics, conformalized_fold_metrics, dqr_fold_metrics, dqr_conformalized_fold_metrics, drmc_fold_metrics, drmc_conformalized_fold_metrics, bootstrap_fold_metrics, bootstrap_conformalized_fold_metrics])
combined_metrics.to_csv(f'./figures/combined_metrics_allbaselines_{r}_alpha_{alpha}_cs_{cs}.csv', index=False)
# Load the data
data = pd.read_csv(f"./figures/combined_metrics_allbaselines_{r}_alpha_{alpha}_cs_{cs}.csv")
# Process the data
# Assuming data has columns: 'method', 'mean_coverage', 'coverage_ci', 'mean_interval_width', 'width_ci'
data['is_conformalized'] = data['method'].str.startswith("CP-")
original_methods = data[~data['is_conformalized']]
conformalized_methods = data[data['is_conformalized']]

grouped_data = data.groupby('method').agg({
    'mean_coverage': ['mean'],
    'coverage_ci': ['mean'],
    'mean_interval_width': ['mean'],
    'width_ci': ['mean']
}).reset_index()
grouped_data.columns = ['method', 'coverage_mean', 'coverage_std', 'width_mean', 'width_std']

# Reorder methods
method_order = ['DKGP', 'DQR', 'DRMC', 'Bootstrap', '', 'CP-DKGP', 'CP-DQR', 'CP-DRMC', 'CP-Bootstrap']


# Create a DataFrame for ordered methods
ordered_grouped_data = pd.DataFrame({'method': method_order})

# Merge to align metrics with the correct methods
ordered_grouped_data = ordered_grouped_data.merge(
    grouped_data,
    on='method',
    how='left'  # Keeps methods from method_order even if not in grouped_data
)

# Fill the gap placeholder ('') with NaNs for metrics
ordered_grouped_data.loc[ordered_grouped_data['method'] == '', ['coverage_mean', 'coverage_std', 'width_mean', 'width_std']] = None

# Variables for plotting
bar_positions = list(range(len(ordered_grouped_data)))
method_labels = [label if label != '' else '' for label in ordered_grouped_data['method'].tolist()]

# Define color palette and style mappings
unique_methods = ordered_grouped_data['method'].unique()
unique_base_methods = sorted({method.replace("CP-", "") for method in unique_methods if method != ''})
color_palette = cm.viridis
color_mapping = {base: color_palette(i / len(unique_base_methods)) for i, base in enumerate(unique_base_methods)}
final_color_mapping = {method: color_mapping.get(method.replace("CP-", ""), "white") for method in unique_methods if method != ''}
styles = {'original': None, 'conformalized': '//'}
method_styles = {method: styles['conformalized'] if "CP-" in method else styles['original'] for method in unique_methods if method != ''}

plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "legend.fontsize": 28,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "figure.dpi": 300,
    "text.usetex": True,
})

# Create the plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False, gridspec_kw={'wspace': 0.2})

# Bar Plot for Coverage
bars_coverage = axes[0].bar(
    bar_positions,
    ordered_grouped_data['coverage_mean'],
    yerr=ordered_grouped_data['coverage_std'],
    color=[final_color_mapping.get(method, "white") for method in ordered_grouped_data['method']],
    hatch=[method_styles.get(method, None) for method in ordered_grouped_data['method']],
    capsize=3
)
axes[0].axhline(1-alpha, color='red', linestyle='--', linewidth=2, label='Nominal Coverage ' + str(alpha))
axes[0].set_ylabel("Mean Coverage")
axes[0].tick_params(axis='y')
axes[0].set_xticks(bar_positions)
axes[0].set_xticklabels(method_labels, rotation=90, ha='center')
axes[0].tick_params(axis='both', which='major', labelsize=25)

# Bar Plot for Interval Width
bars_interval = axes[1].bar(
    bar_positions,
    ordered_grouped_data['width_mean'],
    yerr=ordered_grouped_data['width_std'],
    color=[final_color_mapping.get(method, "white") for method in ordered_grouped_data['method']],
    hatch=[method_styles.get(method, None) for method in ordered_grouped_data['method']],
    capsize=3
)
axes[1].set_ylabel("Mean Interval Width")
axes[1].set_xticks(bar_positions)
axes[1].set_xticklabels(method_labels, rotation=90, ha='center')
axes[1].tick_params(axis='both', which='major', labelsize=25)


# **Create Custom Legend for Biomarkers (PLACEHOLDERS)**
biomarker_legend_elements = [
    mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Hippocampal Volume'),
]

# Add legends outside the plot
fig.legend(handles=biomarker_legend_elements, loc="lower right", frameon=False, fontsize=31, bbox_to_anchor=(1.15, 0))

# Define the legend
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black', hatch=None, label='Baseline Predictor'),
    plt.Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black', hatch='//', label='Conformalized Predictor'),
    plt.Line2D([0], [0], color='red', linestyle='--', label='Nominal Coverage ' + str(1-alpha))
]
fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=32, frameon=False)

# Adjust layout for compactness
plt.tight_layout(rect=[0, 0, 1, 0.88])

# Save the plot
plt.savefig(f"./figures/Comparison_Before_and_After_Conformalization_{r}_{alpha}_{cs}.png", dpi=300, bbox_inches="tight")

method_colors = {
    method: final_color_mapping.get(method, "white")
    for method in ordered_grouped_data['method'].unique()
    if method != ''  # Exclude the gap
}


method_markers = {
    "CP-DKGP": "s",  # Square marker
    "CP-DQR": "^",   # Triangle marker
    "CP-DRMC": "D",  # Diamond marker
    "CP-Bootstrap": "v"  # Inverted triangle marker
}


print('Figure 3: Interval width with Time.')
conformalized_bounds = pd.read_csv(f'./results/conformalized_dkgp_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')
# deep quantile regression 
conformalized_dqr= pd.read_csv(f'./results/conformalized_deep_quantile_regression_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

# deep regression with monte carlo dropout
conformalized_drmc= pd.read_csv(f'./results/conformalized_drmc_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

conf_bootstrap = pd.read_csv(f'./results/conformalized_deep_regression_bootstrap_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

# calculate the interval width
conformalized_bounds['interval_width'] = conformalized_bounds['upper'] - conformalized_bounds['lower']
conformalized_dqr['interval_width'] = conformalized_dqr['upper'] - conformalized_dqr['lower']
conformalized_drmc['interval_width'] = conformalized_drmc['upper'] - conformalized_drmc['lower']
conf_bootstrap['interval_width'] = conf_bootstrap['upper'] - conf_bootstrap['lower']

conformalized_bounds['method'] = 'CP-DKGP'
conformalized_dqr['method'] = 'CP-DQR'
conformalized_drmc['method'] = 'CP-DRMC'
conf_bootstrap['method'] = 'CP-Bootstrap'

# merge the datasets
combined_dkgp = pd.concat([conformalized_bounds, conformalized_dqr, conformalized_drmc, conf_bootstrap], ignore_index=True)

# save 
combined_dkgp.to_csv(f'./figures/combined_dataset_{r}_{alpha}_{cs}.csv', index=False)

# load the dataset
data = pd.read_csv(f"./figures/combined_dataset_{r}_{alpha}_{cs}.csv")

# Generate data for demonstration with CP-DQR and CP-DRMC
methods_to_plot = ['CP-DKGP', 'CP-DQR', 'CP-DRMC', 'CP-Bootstrap']

# Filtering data for the selected methods
filtered_data = data[data['method'].isin(methods_to_plot)]

# Add a 'year' column by converting months to years
filtered_data['year'] = (filtered_data['time'] // 12).astype(int)

# Calculate average interval width for each year and method
avg_width_per_year_method = filtered_data.groupby(['year', 'method'])['interval_width'].mean().reset_index()
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "legend.fontsize": 14,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 300,
    "text.usetex": True,
})
# Ensure all methods appear in the shared legend by explicitly gathering handles and labels from all plots
fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)

# Assign methods to each subplot
methods_in_subplots = ['CP-DKGP', 'CP-DQR', 'CP-DRMC', 'CP-Bootstrap']
# List to store handles and labels for the shared legend
handles_labels = []

for i, (method, ax) in enumerate(zip(methods_in_subplots, axes)):
    method_data = avg_width_per_year_method[avg_width_per_year_method['method'] == method]
    line, = ax.plot(
        method_data['year'],
        method_data['interval_width'],
        label=method,
        color=method_colors[method],
        marker=method_markers[method],
        linestyle='-',
        linewidth=3.5,
        markersize=10
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax.grid(False)

    # Add y-axis title only to the middle plot
    if i == 1:
        ax.set_ylabel('Mean Interval Width')

    # Collect handles and labels for the legend
    handles_labels.append((line, method))

# Common x-label
fig.text(0.5, 0.04, 'Time (in years)', ha='center', fontsize=25)

# increase the font size of the x-axis and y-axis tick labels
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=20)

    
# Add shared legend on top, horizontally aligned
handles, labels = zip(*handles_labels)
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=18, title='Hippocampal Volume', title_fontsize=20, frameon=False,bbox_to_anchor=(0.5, 1.02))

# Adjust spacing between plots and make room for the legend
plt.tight_layout(rect=[0, 0.05, 1, 0.92])  # Adjust top space for the legend
plt.savefig(f"./figures/Interval_Width_with_Time_{r}_{alpha}_{cs}.png", dpi=300, bbox_inches="tight")


print('Figure 4: Population vs Group-Conditional CP for Diagnosis Variable')
# Load longitudinal covariates
longitudinal_covariates = pd.read_csv('data/anonymized_covariates.csv')
longitudinal_covariates['Diagnosis'].replace([-1.0, 0.0, 1.0, 2.0], ['UKN', 'CN', 'MCI', 'AD'], inplace=True)
longitudinal_covariates['Baseline_Diagnosis'] = longitudinal_covariates.groupby('anon_id')['Diagnosis'].transform('first')

longitudinal_covariates['Status'] = (
    longitudinal_covariates.groupby('anon_id')['Diagnosis']
    .transform(lambda x: determine_status(x.tolist()))
)

# rename PTID to id
longitudinal_covariates.rename(columns={'anon_id': 'id'}, inplace=True)

# print('Compare Population vs Group-Conditional CP for Diagnosis Variable')
conformalized_bounds = pd.read_csv(f'./results/population_cp_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')
stratified_conformalized_bounds = pd.read_csv(f'./results/group_conditional_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

# load this file now and check the common subjects
conformalized_dkgp = pd.read_csv(f'./results/population_cp_{r}_results_calibrationset_{cs}_alpha_{alpha}.csv')

# keep only the ids from the stratified bounds
ids = stratified_conformalized_bounds['id'].unique()
conformalized_bounds = conformalized_bounds[conformalized_bounds['id'].isin(ids)]

# Calculate metric
conformalized_results = calculate_subject_level_metrics(conformalized_bounds)
stratified_conformalized_results = calculate_subject_level_metrics_per_covariate(stratified_conformalized_bounds)

# Merge results with covariates    
stratified_conformalized_merged = stratified_conformalized_results.merge(longitudinal_covariates, left_on='id', right_on='id', how='inner')
conformalized_merged = conformalized_results.merge(longitudinal_covariates, left_on='id', right_on='id', how='inner')

rename_mappings = {
'Baseline_Diagnosis': 'BaselineDiagnosis'}

# Rename columns in both dataframes
stratified_conformalized_merged.rename(columns=rename_mappings, inplace=True)
conformalized_merged.rename(columns=rename_mappings, inplace=True)

# In the covariate column, replace the APOE4_Alleles with APOE4Alleles
stratified_conformalized_merged['covariate'] = stratified_conformalized_merged['covariate'].replace('Baseline_Diagnosis', 'BaselineDiagnosis')

covariates = stratified_conformalized_merged['covariate'].unique()

### store the two dataframes
conformalized_merged.to_csv(f'./figures/unstratified_dkgp_merged_{r}_{cs}_{alpha}.csv')
stratified_conformalized_merged.to_csv(f'./figures/stratified_dkgp_merged_{r}_{cs}_{alpha}.csv')

unstratified_data = pd.read_csv(f"./figures/unstratified_dkgp_merged_{r}_{cs}_{alpha}.csv")
stratified_data = pd.read_csv(f"./figures/stratified_dkgp_merged_{r}_{cs}_{alpha}.csv")

# Define marker shapes for each covariate
marker_map = {
    'Diagnosis': 'o',        # Circle
}

# Prepare data for unstratified and stratified metrics
unstratified_means, unstratified_cis, stratified_means, stratified_cis = [], [], [], []
unstratified_interval_means, unstratified_interval_cis = [], []
stratified_interval_means, stratified_interval_cis = [], []
final_labels = []

# Define focused covariates and values
focused_covariates = {
    'Diagnosis': ['AD', 'CN', 'MCI']
}

for covariate, values in focused_covariates.items():
    marker = marker_map.get(covariate, 'o')  # Default marker is 'o'

    for value in values:
        # Unstratified coverage and interval width
        unstratified_group = unstratified_data[unstratified_data[covariate] == value]
        mean, ci = calculate_ci(unstratified_group['coverage'])
        unstratified_means.append(mean)
        unstratified_cis.append(ci)
        mean, ci = calculate_ci(unstratified_group['interval_width'])
        unstratified_interval_means.append(mean)
        unstratified_interval_cis.append(ci)
  
        # Stratified coverage and interval width
        stratified_group = stratified_data[stratified_data[covariate] == value]
        mean, ci = calculate_ci(stratified_group['coverage'])
        stratified_means.append(mean)
        stratified_cis.append(ci)
        mean, ci = calculate_ci(stratified_group['interval_width'])
        stratified_interval_means.append(mean)
        stratified_interval_cis.append(ci)


        final_labels.append(value)

    # Add spacing for separation between covariates
    unstratified_means.append(np.nan)
    unstratified_cis.append(np.nan)
    stratified_means.append(np.nan)
    stratified_cis.append(np.nan)
    unstratified_interval_means.append(np.nan)
    unstratified_interval_cis.append(np.nan)
    stratified_interval_means.append(np.nan)
    stratified_interval_cis.append(np.nan)
    final_labels.append("")

# Create x-axis positions accounting for spacing
x_positions = np.arange(len(unstratified_means))

# Load data
unstratified_data = pd.read_csv(f"./figures/unstratified_dkgp_merged_{r}_{cs}_{alpha}.csv")
stratified_data = pd.read_csv(f"./figures/stratified_dkgp_merged_{r}_{cs}_{alpha}.csv")

# Define marker shapes and colors for stratified and unstratified data
marker_map = {'Diagnosis': 'o'}
colors = {'Group-Conditional Conformal Prediction': '#92140c', 'Population Conformal Prediction': '#111D4A'}

# Prepare data for plotting
final_labels = []
x_positions = []
offset = 0

# Define marker shapes and colors for stratified and unstratified data
colors = {'Group-Conditional Conformal Prediction': '#92140c', 'Population Conformal Prediction': '#111D4A'}

marker_map = { 'Diagnosis': 'o' }
# Desired order for x-tick labels, including spaces
desired_order_with_spaces = [
        "CN", "MCI", "AD"
]

# Create a single plot for overlaying data
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi": 300,
})

# Prepare data for plotting
final_labels = []
x_positions = []
offset = 0

# Define marker shapes for each covariate
marker_map = {
    'Diagnosis': 'o',        # Circle
}

# Iterate through covariates and plot their metrics
for covariate, values in focused_covariates.items():
    marker = marker_map.get(covariate, 'o')  # Get marker for the covariate
    for value in values:
        # Filter data for current covariate value
        unstratified_group = unstratified_data[unstratified_data[covariate] == value]
        stratified_group = stratified_data[stratified_data[covariate] == value]

        # Calculate confidence intervals and means
        if not unstratified_group.empty:
            mean_coverage_unstrat, ci_coverage_unstrat = calculate_ci(unstratified_group['coverage'])
            mean_width_unstrat, ci_width_unstrat = calculate_ci(unstratified_group['interval_width'])

            # Plot unstratified data
            axes[0].errorbar(
                offset,
                mean_coverage_unstrat,
                yerr=ci_coverage_unstrat,
                fmt=marker,
                color=colors['Population Conformal Prediction'],
                ecolor='black',
                capsize=5,
                markersize=8,
                alpha=0.8,
                label="Population Conformal Prediction" if offset == 0 else ""
            )

            axes[1].errorbar(
                offset,
                mean_width_unstrat,
                yerr=ci_width_unstrat,
                fmt=marker,
                color=colors['Population Conformal Prediction'],
                ecolor='black',
                capsize=5,
                markersize=8,
                alpha=0.8
            )

        if not stratified_group.empty:
            mean_coverage_strat, ci_coverage_strat = calculate_ci(stratified_group['coverage'])
            mean_width_strat, ci_width_strat = calculate_ci(stratified_group['interval_width'])

            # Plot stratified data
            axes[0].errorbar(
                offset,
                mean_coverage_strat,
                yerr=ci_coverage_strat,
                fmt=marker,
                color=colors['Group-Conditional Conformal Prediction'],
                ecolor='black',
                capsize=5,
                markersize=8,
                alpha=0.8,
                label="Group-Conditional Conformal Prediction" if offset == 0 else ""
            )

            axes[1].errorbar(
                offset,
                mean_width_strat,
                yerr=ci_width_strat,
                fmt=marker,
                color=colors['Group-Conditional Conformal Prediction'],
                ecolor='black',
                capsize=5,
                markersize=8,
                alpha=0.8
            )

        # Append label and update offset
        final_labels.append(value)
        x_positions.append(offset)
        offset += 1

    # Add spacing after each covariate group
    final_labels.append("")  # Add an empty label for spacing
    x_positions.append(offset)
    offset += 1

# Manually adjust positions to match desired order
x_positions = list(range(len(desired_order_with_spaces)))

for ax in axes:
    ax.set_xticks(x_positions)
    ax.set_xticklabels(desired_order_with_spaces, rotation=90, ha='right')

# Set y-axis labels
axes[0].set_ylabel("Mean Coverage")
axes[1].set_ylabel("Mean Interval Width")

# Add titles for subplots
axes[0].set_title("Coverage Comparison")
axes[1].set_title("Interval Width Comparison")

# reduce the font size of the y-axis labels
for ax in axes:
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)

# increase the font size of the x-ticks labels
for ax in axes:
    ax.tick_params(axis='x', labelsize=16)

# increate the font size of the y-ticks labels 
for ax in axes:
    ax.tick_params(axis='y', labelsize=16)

# Add horizontal line for nominal coverage at 0.9
axes[0].axhline(y=1-alpha, color='red', linestyle='--', linewidth=2, label='Nominal Coverage ' + str(1-alpha))

# **Create Custom Legend for Marker Shapes (Covariates)**
shape_legend_elements = [
    mlines.Line2D([], [], color='black', marker=marker_map['Diagnosis'], linestyle='None', markersize=8, label='Diagnosis', markerfacecolor='black'),
]

biomarker_legend_elements = [
    mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Hippocampal Volume', markerfacecolor='black'),
]

# Add legends outside the plot
fig.legend(handles=shape_legend_elements, loc="upper right", frameon=False, fontsize=16, bbox_to_anchor=(1.15, 1))
fig.legend(handles=biomarker_legend_elements, loc="lower right", frameon=False, fontsize=16, bbox_to_anchor=(1.15, 0))

# Update legends to include the nominal coverage line
axes[0].legend(loc="upper center", ncol=2, frameon=False, fontsize=16)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(f"./figures/Population_vs_Group_Conditional_CP_for_Diagnosis_Variable_{r}_{alpha}_{cs}.png", format="png", dpi=300, bbox_inches="tight")



