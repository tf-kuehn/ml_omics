import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os

def main():
    os.makedirs("../results", exist_ok=True)
    
    print("Loading predictions for all strategies...")
    with open('../results/task3_model_predictions.pkl', 'rb') as f:
        data = pickle.load(f)
        
    all_strategy_preds = data["predictions"]
    Y_test_dict = data["Y_test_dict"]
    obs_test_dict = data["obs_test_dict"]
    
    conditions = ['Control', 'IFNγ', 'Co-culture']
    all_results = []
    
    for strategy_name, strategy_preds in all_strategy_preds.items():
        Y_test = Y_test_dict[strategy_name]
        obs_test = obs_test_dict[strategy_name]
        
        for model_name, y_pred in strategy_preds.items():
            
            for cond in conditions:
                cond_mask = (obs_test['condition'] == cond).values
                if not cond_mask.any():
                    continue
                    
                y_true_cond = Y_test[cond_mask]
                
                # y_pred now contains predictions from n_bootstraps models
                # Shape: (n_bootstraps, n_test_samples, n_genes)
                n_bootstraps = y_pred.shape[0]
                
                boot_mses = []
                boot_pearsons = []
                
                for b in range(n_bootstraps):
                    y_pred_b = y_pred[b]
                    y_pred_cond = y_pred_b[cond_mask]
                    
                    # MSE for this bootstrap model
                    boot_mses.append(mean_squared_error(y_true_cond, y_pred_cond))
                    
                    # Pearson r for this bootstrap model
                    corrs = []
                    for i in range(len(y_true_cond)):
                        if np.std(y_true_cond[i]) > 0 and np.std(y_pred_cond[i]) > 0:
                            r, _ = pearsonr(y_true_cond[i], y_pred_cond[i])
                            corrs.append(r)
                        else:
                            corrs.append(0.0)
                    boot_pearsons.append(np.nanmean(corrs) if corrs else 0.0)

                # Mean metrics
                mse = np.mean(boot_mses)
                mean_corr = np.mean(boot_pearsons)

                # 95% Confidence Intervals
                mse_ci_low, mse_ci_high = np.percentile(boot_mses, [2.5, 97.5])
                pearson_ci_low, pearson_ci_high = np.percentile(boot_pearsons, [2.5, 97.5])

                all_results.append({
                    'Strategy': strategy_name,
                    'Condition': cond,
                    'Model': model_name,
                    'MSE': mse,
                    'MSE_CI_Lower': mse_ci_low,
                    'MSE_CI_Upper': mse_ci_high,
                    'Pearson_r': mean_corr,
                    'Pearson_r_CI_Lower': pearson_ci_low,
                    'Pearson_r_CI_Upper': pearson_ci_high
                })
                
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("../results/task3_metrics_all_strategies.csv", index=False)
    
    # Clean up model names for plotting
    df_results['Model'] = df_results['Model'].str.replace('Baseline_', 'Base: ')
    df_results['Strategy'] = df_results['Strategy'].str.replace('_', ' ').str.title()
    
    sns.set_theme(style='whitegrid', font_scale=1.1)
    
    print("Generating Pearson Correlation Plot...")
    g1 = sns.catplot(
        data=df_results,
        x='Strategy',
        y='Pearson_r',
        hue='Model',
        col='Condition',
        kind='bar',
        palette='Set2',
        height=5,
        aspect=1.2,
    )
    g1.set_axis_labels('Selection Strategy', 'Mean Pearson Correlation (r)')
    g1.set_titles(col_template='Condition: {col_name}')
    g1.set_xticklabels(rotation=45, ha='right')
    g1.fig.subplots_adjust(top=0.8)
    g1.fig.suptitle('Perturbation Prediction: Pearson Correlation by Condition')
    plt.savefig("../results/task3_pearson_strategies_plot.png", bbox_inches='tight')
    plt.show()

    print("Generating Mean Squared Error Plot...")
    g2 = sns.catplot(
        data=df_results,
        x='Strategy',
        y='MSE',
        hue='Model',
        col='Condition',
        kind='bar',
        palette='Set2',
        height=5,
        aspect=1.2,
    )
    g2.set_axis_labels('Selection Strategy', 'Mean Squared Error (MSE)')
    g2.set_titles(col_template='Condition: {col_name}')
    g2.set_xticklabels(rotation=45, ha='right')
    g2.fig.subplots_adjust(top=0.8)
    g2.fig.suptitle('Perturbation Prediction: MSE by Condition')
    plt.savefig("../results/task3_mse_strategies_plot.png", bbox_inches='tight')
    plt.show()
    
    print("\n--- Summary Statistics with 95% CIs (Averaged across conditions) ---")
    summary_df = df_results.groupby(['Strategy', 'Model']).agg({
        'Pearson_r': 'mean',
        'Pearson_r_CI_Lower': 'mean',
        'Pearson_r_CI_Upper': 'mean',
        'MSE': 'mean',
        'MSE_CI_Lower': 'mean',
        'MSE_CI_Upper': 'mean'
    }).round(4)
    
    # Format CI bounds into readable strings for console display
    summary_df['Pearson_r (95% CI)'] = summary_df.apply(
        lambda row: f"{row['Pearson_r']} [{row['Pearson_r_CI_Lower']}, {row['Pearson_r_CI_Upper']}]", axis=1
    )
    summary_df['MSE (95% CI)'] = summary_df.apply(
        lambda row: f"{row['MSE']} [{row['MSE_CI_Lower']}, {row['MSE_CI_Upper']}]", axis=1
    )
    
    display_df = summary_df[['Pearson_r (95% CI)', 'MSE (95% CI)']].unstack('Model')
    print(display_df)

if __name__ == "__main__":
    main()