'''
NeuRIPS 2025
Clinical Application: Utilize the Worst Case Rate of Change as a Biomarker for Conversion MCI-AD

Showcase that if we conformalize the trajectory predictor that gives already a notion of uncertainty I am 
able to capture more progressors.
This is the clinical gain of the conformal approach.
Comparison with RoC, WROC, and WROC-stratified.
'''
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Define file paths
file_paths = {
    "DRMC": "./results/combined_results_14_0.05_drmc.csv",
    "DKGP": "./results/combined_results_14_0.05_dkgp.csv",
    "DQR": "./results/combined_results_14_0.05_dqr.csv",
    "Bootstrap": "./results/combined_results_14_0.05_bootstrap.csv"
}

# Step 1: Compute rate of change and lower-bound-based rate of change (WCRC)
def compute_wcrc(df_method):
    wcrc_list = []
    for subject_id, group in df_method.groupby("subject"):
        group_sorted = group.sort_values("months")
        t0 = group_sorted.iloc[0]["months"]
        y0 = group_sorted.iloc[0]["biom_pred"]
        t_last = group_sorted["months"].max()


        y_t = group_sorted.iloc[-1]["biom_pred"]
        y_min = group_sorted["biom_lower"].iloc[-1]

        # Compute both rate of change and lower-bound-based rate of change
        roc = (y_t - y0) / (t_last - t0)
        wcrc = (y_min - y0) / (t_last - t0)
        converter = int(group_sorted.iloc[0]["converter"])

        wcrc_list.append({
            "subject": subject_id,
            "roc": roc,
            "wcrc": wcrc,
            "converter": converter
        })

    return pd.DataFrame(wcrc_list)

# Step 2: Compute comprehensive metrics for either 'roc' or 'wcrc'
def compute_metrics_from_score(wcrc_df, score_name, method_name):
    # ROC analysis
    fpr, tpr, roc_thresholds = roc_curve(wcrc_df["converter"], -wcrc_df[score_name])
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall analysis
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(wcrc_df["converter"], -wcrc_df[score_name])
    pr_auc = average_precision_score(wcrc_df["converter"], -wcrc_df[score_name])
    
    # Youden's J statistic for optimal threshold
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_thresh = roc_thresholds[best_idx]
    y_pred = (-wcrc_df[score_name] >= best_thresh).astype(int)

    # Metrics at optimal threshold
    precision_opt = precision_score(wcrc_df["converter"], y_pred)
    recall_opt = recall_score(wcrc_df["converter"], y_pred)
    f1_opt = f1_score(wcrc_df["converter"], y_pred)
    
    # Additional threshold-free metrics
    # Balanced accuracy at optimal threshold
    tn = ((wcrc_df["converter"] == 0) & (y_pred == 0)).sum()
    fp = ((wcrc_df["converter"] == 0) & (y_pred == 1)).sum()
    fn = ((wcrc_df["converter"] == 1) & (y_pred == 0)).sum()
    tp = ((wcrc_df["converter"] == 1) & (y_pred == 1)).sum()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "Method": f"{method_name}_{score_name}",
        "Optimal Threshold (Youden's J)": -best_thresh,
        "Precision (Optimal)": precision_opt,
        "Recall/Sensitivity (Optimal)": recall_opt,
        "Specificity (Optimal)": specificity,
        "Balanced Accuracy (Optimal)": balanced_accuracy,
        "F1 Score (Optimal)": f1_opt,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "ROC_Curve_Data": (fpr, tpr, roc_thresholds),
        "PR_Curve_Data": (precision_curve, recall_curve, pr_thresholds)
    }

def calculate_prediction_horizon_statistics():
    """
    Calculate and print statistics of the prediction horizon (last timepoint) 
    across all subjects for all methods.
    """
    print("\n" + "="*100)
    print("📊 PREDICTION HORIZON STATISTICS")
    print("="*100)
    
    all_horizon_data = []
    
    for method, path in file_paths.items():
        print(f"\n🔍 Analyzing prediction horizon for method: {method}")
        print(f"   📁 Loading data from: {path}")
        
        try:
            df = pd.read_csv(path)
            df = df.dropna(subset=["converter"])
            df = df[df["converter"] != -1]
            
            print(f"   📊 Total subjects in dataset: {df['subject'].nunique()}")
            
            # Calculate prediction horizon for each subject
            horizon_stats = []
            for subject_id, group in df.groupby("subject"):
                group_sorted = group.sort_values("months")
                prediction_horizon = group_sorted["months"].max()  # Last timepoint
                
                horizon_stats.append({
                    "method": method,
                    "subject": subject_id,
                    "prediction_horizon": prediction_horizon,
                    "converter": int(group_sorted.iloc[0]["converter"])
                })
            
            horizon_df = pd.DataFrame(horizon_stats)
            all_horizon_data.append(horizon_df)
            
            # Print statistics for this method
            print(f"   📈 Prediction Horizon Statistics for {method}:")
            print(f"      • Mean: {horizon_df['prediction_horizon'].mean():.2f} months")
            print(f"      • Median: {horizon_df['prediction_horizon'].median():.2f} months")
            print(f"      • Std: {horizon_df['prediction_horizon'].std():.2f} months")
            print(f"      • Min: {horizon_df['prediction_horizon'].min():.2f} months")
            print(f"      • Max: {horizon_df['prediction_horizon'].max():.2f} months")
            print(f"      • Range: {horizon_df['prediction_horizon'].max() - horizon_df['prediction_horizon'].min():.2f} months")
            
            # Statistics by conversion status
            converters = horizon_df[horizon_df['converter'] == 1]
            non_converters = horizon_df[horizon_df['converter'] == 0]
            
            print(f"      • Converters (n={len(converters)}):")
            if len(converters) > 0:
                print(f"        - Mean: {converters['prediction_horizon'].mean():.2f} months")
                print(f"        - Median: {converters['prediction_horizon'].median():.2f} months")
                print(f"        - Std: {converters['prediction_horizon'].std():.2f} months")
            
            print(f"      • Non-converters (n={len(non_converters)}):")
            if len(non_converters) > 0:
                print(f"        - Mean: {non_converters['prediction_horizon'].mean():.2f} months")
                print(f"        - Median: {non_converters['prediction_horizon'].median():.2f} months")
                print(f"        - Std: {non_converters['prediction_horizon'].std():.2f} months")
            
        except Exception as e:
            print(f"   ❌ Error processing {method}: {e}")
            continue
    
    # Combine all data and calculate overall statistics
    if all_horizon_data:
        combined_df = pd.concat(all_horizon_data, ignore_index=True)
        
        print(f"\n" + "="*80)
        print("📊 OVERALL PREDICTION HORIZON STATISTICS")
        print("="*80)
        
        print(f"📈 Overall Statistics (All Methods Combined):")
        print(f"   • Total subjects: {len(combined_df)}")
        print(f"   • Mean prediction horizon: {combined_df['prediction_horizon'].mean():.2f} months")
        print(f"   • Median prediction horizon: {combined_df['prediction_horizon'].median():.2f} months")
        print(f"   • Standard deviation: {combined_df['prediction_horizon'].std():.2f} months")
        print(f"   • Minimum: {combined_df['prediction_horizon'].min():.2f} months")
        print(f"   • Maximum: {combined_df['prediction_horizon'].max():.2f} months")
        print(f"   • Range: {combined_df['prediction_horizon'].max() - combined_df['prediction_horizon'].min():.2f} months")
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\n📊 Percentiles:")
        for p in percentiles:
            value = combined_df['prediction_horizon'].quantile(p/100)
            print(f"   • {p}th percentile: {value:.2f} months")
        
        # Statistics by conversion status
        all_converters = combined_df[combined_df['converter'] == 1]
        all_non_converters = combined_df[combined_df['converter'] == 0]
        
        print(f"\n🏥 By Conversion Status:")
        print(f"   • Converters (n={len(all_converters)}):")
        if len(all_converters) > 0:
            print(f"     - Mean: {all_converters['prediction_horizon'].mean():.2f} months")
            print(f"     - Median: {all_converters['prediction_horizon'].median():.2f} months")
            print(f"     - Std: {all_converters['prediction_horizon'].std():.2f} months")
            print(f"     - Range: {all_converters['prediction_horizon'].min():.2f} - {all_converters['prediction_horizon'].max():.2f} months")
        
        print(f"   • Non-converters (n={len(all_non_converters)}):")
        if len(all_non_converters) > 0:
            print(f"     - Mean: {all_non_converters['prediction_horizon'].mean():.2f} months")
            print(f"     - Median: {all_non_converters['prediction_horizon'].median():.2f} months")
            print(f"     - Std: {all_non_converters['prediction_horizon'].std():.2f} months")
            print(f"     - Range: {all_non_converters['prediction_horizon'].min():.2f} - {all_non_converters['prediction_horizon'].max():.2f} months")
        
        # Statistical test for difference between converters and non-converters
        if len(all_converters) > 0 and len(all_non_converters) > 0:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(all_converters['prediction_horizon'], 
                                            all_non_converters['prediction_horizon'])
            print(f"\n🔬 Statistical Test (Converters vs Non-converters):")
            print(f"   • t-statistic: {t_stat:.4f}")
            print(f"   • p-value: {p_value:.4f}")
            print(f"   • Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        
        # Save detailed statistics to CSV
        detailed_stats = combined_df.groupby(['method', 'converter'])['prediction_horizon'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        detailed_stats.to_csv('HighRisk_prediction_horizon_statistics.csv')
        print(f"\n💾 Detailed statistics saved to: HighRisk_prediction_horizon_statistics.csv")
        
        print(f"\n✅ PREDICTION HORIZON STATISTICS COMPLETED!")
        print(f"   The analysis shows the distribution of prediction horizons across all subjects")
        print(f"   and methods, providing insight into the temporal characteristics of the dataset.")
        
    else:
        print(f"\n❌ No valid data found for prediction horizon analysis.")

# Step 3: Aggregate results
all_metrics = []

for method, path in file_paths.items():
    df = pd.read_csv(path)

    df = df.dropna(subset=["converter"])
    df = df[df["converter"] != -1]

    for variant in df["method"].unique():
        df_variant = df[df["method"] == variant]
        wcrc_df = compute_wcrc(df_variant)

        metrics_roc = compute_metrics_from_score(wcrc_df, "roc", variant)
        metrics_wcrc = compute_metrics_from_score(wcrc_df, "wcrc", variant)

        all_metrics.append(metrics_roc)
        all_metrics.append(metrics_wcrc)

# Step 4: Combine results into DataFrame and print


df_all_metrics = pd.DataFrame(all_metrics)

# remove the ROC_Curve_Data and PR_Curve_Data columns
df_all_metrics = df_all_metrics.drop(columns=["ROC_Curve_Data", "PR_Curve_Data"])

print(df_all_metrics[['Method', 'Optimal Threshold (Youden\'s J)', 'Precision (Optimal)', 'Recall/Sensitivity (Optimal)', 'F1 Score (Optimal)']])