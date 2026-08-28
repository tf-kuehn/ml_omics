import numpy as np
import scanpy as sc
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

def main():
    os.makedirs("../results", exist_ok=True)
    
    print("Loading features and multi-strategy splits...")
    adata_lfc = sc.read_h5ad('../processed_data/task3_lfc_features.h5ad')
    
    with open('../processed_data/task3_splits.pkl', 'rb') as f:
        all_splits = pickle.load(f)
        
    all_strategy_predictions = {}
    obs_test_dict = {}
    Y_test_dict = {}

    for strategy_name, split_info in all_splits.items():
        print(f"\n=========================================")
        print(f"Training models for Strategy: {strategy_name}")
        print(f"=========================================")
        
        train_genes = split_info["train_genes"]
        test_genes = split_info["test_genes"]
        all_50 = split_info["all_50"]
        
        # Filter AnnData to the current strategy's 50 perturbations
        adata_50 = adata_lfc[adata_lfc.obs['perturbation'].isin(all_50)].copy()
        is_test = adata_50.obs['perturbation'].isin(test_genes).values
        
        # Extract predefined features
        X_raw = adata_50.obsm['X_features']
        X_rand_raw = adata_50.obsm['X_random']
        X_ideal_raw = adata_50.obsm['X_idealized']
        X_mofa_raw = adata_50.obsm['X_mofa_features']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        X_rand_scaled = scaler.fit_transform(X_rand_raw)
        X_ideal_scaled = scaler.fit_transform(X_ideal_raw)
        X_mofa_scaled = scaler.fit_transform(X_mofa_raw)
        
        Y = adata_50.X
        
        # Train / Test Splitting
        X_train, Y_train = X_scaled[~is_test], Y[~is_test]
        X_test, Y_test = X_scaled[is_test], Y[is_test]
        
        X_train_rand, X_test_rand = X_rand_scaled[~is_test], X_rand_scaled[is_test]
        X_train_ideal, X_test_ideal = X_ideal_scaled[~is_test], X_ideal_scaled[is_test]
        X_train_mofa, X_test_mofa = X_mofa_scaled[~is_test], X_mofa_scaled[is_test]
        obs_test_dict[strategy_name] = adata_50.obs[is_test].reset_index(drop=True)
        
        n_samples_train = Y_train.shape[0]
        n_features = Y_train.shape[1]
        
        # Compress targets using PCA to make training tractable
        target_pcs_train = min(50, n_samples_train, n_features)
        pca_y = PCA(n_components=target_pcs_train, random_state=42)
        Y_train_pca = pca_y.fit_transform(Y_train)
        
        models = {
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
            "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance')
        }
        
        strategy_preds = {}
        n_bootstraps = 30  # Number of data subsets for parameter uncertainty
        
        print(f"Training ML models with {n_bootstraps} bootstraps for parameter uncertainty...")
        for model_name, model in models.items():
            boot_preds = []
            for _ in range(n_bootstraps):
                # Resample training data with replacement
                boot_idx = np.random.choice(n_samples_train, size=n_samples_train, replace=True)
                model.fit(X_train[boot_idx], Y_train_pca[boot_idx])
                pred_pca = model.predict(X_test)
                # Inverse transform back to 2000 gene space
                boot_preds.append(pca_y.inverse_transform(pred_pca))
                
            # Store array of shape (n_bootstraps, n_test_samples, n_genes)
            strategy_preds[model_name] = np.array(boot_preds)
            
        print("Computing Baselines...")
        
        # Train MOFA Enhanced Model
        boot_mofa_preds = []
        for _ in range(n_bootstraps):
            boot_idx = np.random.choice(n_samples_train, size=n_samples_train, replace=True)
            rf_mofa = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            rf_mofa.fit(X_train_mofa[boot_idx], Y_train_pca[boot_idx])
            boot_mofa_preds.append(pca_y.inverse_transform(rf_mofa.predict(X_test_mofa)))
        strategy_preds["MOFA_RandomForest"] = np.array(boot_mofa_preds)
        
        # Baseline 1: Train Mean
        boot_mean_preds = []
        for _ in range(n_bootstraps):
            boot_idx = np.random.choice(n_samples_train, size=n_samples_train, replace=True)
            dummy_mean = DummyRegressor(strategy='mean')
            dummy_mean.fit(X_train[boot_idx], Y_train[boot_idx])
            boot_mean_preds.append(dummy_mean.predict(X_test))
        strategy_preds["Baseline_TrainMean"] = np.array(boot_mean_preds)
        
        # Baseline 2: Negative (No Change / Zeros)
        # Using DummyRegressor as requested
        zeros_pred = np.zeros((X_test.shape[0], Y_train.shape[1]))
        boot_zero_preds = []
        for _ in range(n_bootstraps):
            dummy_zero = DummyRegressor(strategy="constant", constant=np.zeros(Y_train.shape[1]))
            # DummyRegressor requires fitting, though it ignores X
            dummy_zero.fit(X_train, Y_train) 
            boot_zero_preds.append(dummy_zero.predict(X_test))
        strategy_preds["Baseline_Negative_Zeros"] = np.array(boot_zero_preds)
        
        # Baseline 3: Random Embeddings 
        # Tests if biological features are better than gaussian noise
        boot_rand_preds = []
        for _ in range(n_bootstraps):
            boot_idx = np.random.choice(n_samples_train, size=n_samples_train, replace=True)
            ridge_rand = Ridge(alpha=1.0)
            ridge_rand.fit(X_train_rand[boot_idx], Y_train_pca[boot_idx])
            boot_rand_preds.append(pca_y.inverse_transform(ridge_rand.predict(X_test_rand)))
        strategy_preds["Baseline_Random"] = np.array(boot_rand_preds)
        
        # Baseline 4: Idealized Embeddings (Data Leakage - Upper Bound)
        boot_ideal_preds = []
        for _ in range(n_bootstraps):
            boot_idx = np.random.choice(n_samples_train, size=n_samples_train, replace=True)
            ridge_ideal = Ridge(alpha=1.0)
            ridge_ideal.fit(X_train_ideal[boot_idx], Y_train_pca[boot_idx])
            boot_ideal_preds.append(pca_y.inverse_transform(ridge_ideal.predict(X_test_ideal)))
        strategy_preds["Baseline_Idealized_PCA"] = np.array(boot_ideal_preds)
        
        # Save strategy results
        all_strategy_predictions[strategy_name] = strategy_preds

    output_data = {
        "predictions": all_strategy_predictions,
        "Y_test_dict": Y_test_dict,
        "obs_test_dict": obs_test_dict
    }
    
    with open('../results/task3_model_predictions.pkl', 'wb') as f:
        pickle.dump(output_data, f)
    print("\nSaved predictions for all strategies to ../results/task3_model_predictions.pkl")

if __name__ == "__main__":
    main()