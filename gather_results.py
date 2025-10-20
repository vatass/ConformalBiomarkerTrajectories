import pandas as pd
import numpy as np
import glob
import os

def load_dkgp_results(task, r, cs, alpha):
    """Load CP-DKGP results"""
    file_path = f'./conformalresults/singletask_{task}_{r}_dkgp_population_adniblsa_conformal_{cs}_alpha_{alpha}.csv'
    df = pd.read_csv(file_path)
    df['method'] = 'DKGP'
    return df

def load_cp_dkgp_results(task, r, cs, alpha):
    """Load CP-DKGP results"""
    file_path = f'./conformalresults/singletask_{task}_{r}_dkgp_conformalized_predictions_adniblsa_conformal_{cs}_alpha_{alpha}.csv'
    df = pd.read_csv(file_path)
    df['method'] = 'CP-DKGP'
    return df

def load_drmc_results(task, r, cs, alpha):
    """Load DRMC results"""
    file_path = f'./conformalresults/deep_regression_mc_results_{task}_{r}_alpha_{alpha}_calibrationset_{cs}.csv'
    df = pd.read_csv(file_path)
    df['method'] = 'DRMC'
    return df

def load_cp_drmc_results(task, r, cs, alpha):
    """Load CP-DRMC results"""
    file_path = f'./conformalresults/deep_regression_mc_conformal_results_{task}_{r}_alpha_{alpha}_calibrationset_{cs}.csv'
    df = pd.read_csv(file_path)
    df['method'] = 'CP-DRMC'
    return df

def load_dqr_results(task, r, cs, alpha):
    """Load DQR results"""
    file_path = f'./conformalresults/deep_quantile_regression_results_{task}_{r}_alpha_{alpha}_calibrationset_{cs}.csv'
    df = pd.read_csv(file_path)
    df['method'] = 'DQR'
    return df

def load_cp_dqr_results(task, r, cs, alpha):
    """Load CP-DQR results"""
    file_path = f'./conformalresults/deep_quantile_regression_conformal_results_{task}_{r}_alpha_{alpha}_calibrationset_{cs}.csv'
    df = pd.read_csv(file_path)
    df['method'] = 'CP-DQR'
    return df

def load_bootstrap_results(task, r, cs, alpha):
    """Load Bootstrap results"""
    file_path = f'./conformalresults/deep_regression_bootstrap_results_{task}_{r}_alpha_{alpha}_calibrationset_{cs}.csv'
    df = pd.read_csv(file_path)
    df['method'] = 'Bootstrap'
    return df

def load_cp_bootstrap_results(task, r, cs, alpha):
    """Load CP-Bootstrap results"""
    file_path = f'./conformalresults/deep_regression_bootstrap_conformal_results_{task}_{r}_alpha_{alpha}_calibrationset_{cs}.csv'
    df = pd.read_csv(file_path)
    df['method'] = 'CP-Bootstrap'
    return df

def load_dme_results(task, r, cs, alpha):
    """Load DM results"""
    file_path = f'./conformalresults/dme_results_{task}_{r}_alpha_{alpha}_calibrationset_{cs}.csv'
    df = pd.read_csv(file_path)
    df['method'] = 'DME'
    return df

def load_cp_dme_results(task, r, cs, alpha):
    """Load CP-DM results"""
    file_path = f'./conformalresults/dme_conformal_results_{task}_{r}_alpha_{alpha}_calibrationset_{cs}.csv'
    df = pd.read_csv(file_path)
    df['method'] = 'CP-DME'
    return df

def load_covariates():
    """Load longitudinal covariates data and determine converter status"""
    covariates = pd.read_csv('longitudinal_covariates_total.csv')
    
    # Determine progression status for each subject
    def get_progression_status(group):
        diagnoses = group['Diagnosis'].tolist()
        if len(diagnoses) < 2:
            return f"{diagnoses[0]}->{diagnoses[0]}"
        
        initial = diagnoses[0]
        final = diagnoses[-1]
        return f"{initial}->{final}"
    
    progression_status = covariates.groupby('PTID').apply(get_progression_status).reset_index()
    progression_status.columns = ['PTID', 'progression_status']
    
    # Determine converter status based on progression pattern
    def determine_converter_status(progression):
        if progression.startswith('MCI->'):
            if progression == 'MCI->AD':
                return 1  # Converter (MCI to AD)
            elif progression == 'MCI->MCI':
                return 0  # Non-converter (stable MCI)
        return -1  # Not applicable (didn't start as MCI)
    
    progression_status['converter'] = progression_status['progression_status'].apply(determine_converter_status)
    
    # Add other covariates (taking the first occurrence for each subject)
    other_covariates = covariates.groupby('PTID').first().reset_index()
    covariate_cols = ['PTID', 'Education_Years', 'Race', 'Sex', 'APOE4_Alleles']
    other_covariates = other_covariates[covariate_cols]
    
    # Merge all information
    final_covariates = pd.merge(progression_status, other_covariates, on='PTID', how='left')
    
    return final_covariates

def create_results_dataframe(dkgp_df, cp_dkgp_df, covariates):
    """Create the final results dataframe in the requested format"""
    # Combine both methods
    results = pd.concat([dkgp_df, cp_dkgp_df], ignore_index=True)
    
    # Merge with covariates
    results = pd.merge(results, covariates, left_on='id', right_on='PTID', how='left')
    
    # Rename columns to match requested format
    results = results.rename(columns={
        'id': 'subject',
        'time': 'months',
        'y': 'biom_true',
        'score': 'biom_pred',
        'lower': 'biom_lower',
        'upper': 'biom_upper',
        'Education_Years': 'education',
        'Sex': 'sex',
        'APOE4_Alleles': 'apoe4'
    })
    
    # Calculate band width if not present
    if 'band_width' not in results.columns:
        results['band_width'] = results['biom_upper'] - results['biom_lower']
    
    # Select and reorder columns
    final_columns = [
        'subject', 'method', 'months', 'biom_true', 'biom_pred',
        'biom_lower', 'biom_upper', 'band_width', 'converter',
        'progression_status', 'education', 'sex', 'apoe4', 'Race'
    ]
    
    # Add any additional covariates if present
    covariate_columns = [col for col in results.columns if col not in final_columns + ['PTID', 'id']]
    final_columns.extend(covariate_columns)
    
    return results[final_columns]

def main():
    # Parameters
    task = 'MUSE'  # or whatever task you're using
    r = '14'      # conformal split percentage
    cs = '0.3'     # calibration set size
    alpha = '0.01'  # alpha value
    
    # Load and process covariates to get converter status
    covariates = load_covariates()
    
    # Print statistics about conversion status
    print("\nConversion Status Statistics:")
    print("----------------------------")
    conversion_counts = covariates['converter'].value_counts()
    print("\nNumber of subjects per conversion status:")
    print(f"Converters (1): {conversion_counts.get(1, 0)}")
    print(f"Non-converters (0): {conversion_counts.get(0, 0)}")
    print(f"Not applicable (-1): {conversion_counts.get(-1, 0)}")
    
    # Print progression patterns
    print("\nProgression patterns:")
    progression_counts = covariates['progression_status'].value_counts()
    for pattern, count in progression_counts.items():
        print(f"{pattern}: {count}")
    
    # DKGP AND CP-DKGP
    dkgp_results = load_dkgp_results(task, r, cs, alpha)  
    cp_dkgp_results = load_cp_dkgp_results(task, r, cs, alpha)

    # DRMC AND CP-DRMC
    drmc_results = load_drmc_results(task, r, cs, alpha)
    cp_drmc_results = load_cp_drmc_results(task, r, cs, alpha)
    
    # DQR AND CP-DQR
    dqr_results = load_dqr_results(task, r, cs, alpha)
    cp_dqr_results = load_cp_dqr_results(task, r, cs, alpha)

    # Bootstrap AND CP-Bootstrap
    bootstrap_results = load_bootstrap_results(task, r, cs, alpha)
    cp_bootstrap_results = load_cp_bootstrap_results(task, r, cs, alpha)
    
    # DME AND CP-DME
    dme_results = load_dme_results(task, r, cs, alpha)
    cp_dme_results = load_cp_dme_results(task, r, cs, alpha)

    # Create final results dataframes
    dkgp_results = create_results_dataframe(dkgp_results, cp_dkgp_results, covariates)
    drmc_results = create_results_dataframe(drmc_results, cp_drmc_results, covariates)
    dqr_results = create_results_dataframe(dqr_results, cp_dqr_results, covariates)
    bootstrap_results = create_results_dataframe(bootstrap_results, cp_bootstrap_results, covariates)
    dme_results = create_results_dataframe(dme_results, cp_dme_results, covariates)

    methods = dkgp_results['method'].unique()
 

    # Print time statistics per conversion status
    print("\nTime Statistics per Conversion Status:")
    print("-------------------------------------")
    for status in [-1, 0, 1]:
        status_name = "Not applicable" if status == -1 else "Non-converters" if status == 0 else "Converters"
        status_data = dkgp_results[dkgp_results['converter'] == status]
        
        print(f"\n{status_name}:")
        print(f"Number of unique subjects: {status_data['subject'].nunique()}")
        print(f"Number of timepoints: {len(status_data)}")
        print(f"Mean months: {status_data['months'].mean():.2f}")
        print(f"Min months: {status_data['months'].min():.2f}")
        print(f"Max months: {status_data['months'].max():.2f}")
        print(f"Median months: {status_data['months'].median():.2f}")

    # Save to CSV
    dkgp_results.to_csv(f'combined_results_{r}_{alpha}_dkgp.csv', index=False)
    drmc_results.to_csv(f'combined_results_{r}_{alpha}_drmc.csv', index=False)
    dqr_results.to_csv(f'combined_results_{r}_{alpha}_dqr.csv', index=False)
    bootstrap_results.to_csv(f'combined_results_{r}_{alpha}_bootstrap.csv', index=False)
    dme_results.to_csv(f'combined_results_{r}_{alpha}_dme.csv', index=False)
    print(f"\nResults saved to combined_results.csv")

if __name__ == "__main__":
    main() 