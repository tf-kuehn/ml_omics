import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp
import os
from sklearn.decomposition import PCA

def main():
    os.makedirs("../processed_data", exist_ok=True)
    
    print("Loading data...")
    adata_pre = sc.read_h5ad('../data/frangieh/adata_preprocessed.h5ad')
    adata_lfc = sc.read_h5ad('../data/frangieh/log_fold_change_adata_hvg.h5ad')
    
    print("Calculating basal expression...")
    ctrl_cells = adata_pre[adata_pre.obs['perturbation'] == 'control']
    df_ctrl = ctrl_cells.to_df()
    df_ctrl['condition'] = ctrl_cells.obs['condition'].values
    mean_ctrl_by_cond = df_ctrl.groupby('condition').mean()
    
    basal_ko_values = []
    for _, row in adata_lfc.obs.iterrows():
        cond = row['condition']
        ko_gene = row['perturbation']
        val = mean_ctrl_by_cond.loc[cond, ko_gene] if ko_gene in mean_ctrl_by_cond.columns else 0.0
        basal_ko_values.append(val)
        
    print("Mapping Hallmark Pathways...")
    hallmark = gp.get_library(name="MSigDB_Hallmark_2020", organism="Human")
    hallmark_upper = {pathway: {str(g).upper() for g in genes} for pathway, genes in hallmark.items()}
    
    eligible_genes = [g for g in adata_lfc.obs["perturbation"].unique() if str(g).lower() != 'control']
    pathway_matrix = pd.DataFrame(0, index=eligible_genes, columns=hallmark_upper.keys(), dtype=int)
    for gene in eligible_genes:
        g_up = str(gene).upper()
        for pathway, p_genes in hallmark_upper.items():
            if g_up in p_genes:
                pathway_matrix.loc[gene, pathway] = 1
                
    pathway_features = []
    for pert in adata_lfc.obs['perturbation']:
        if pert in pathway_matrix.index:
            pathway_features.append(pathway_matrix.loc[pert].values)
        else:
            pathway_features.append(np.zeros(len(pathway_matrix.columns)))
    pathway_features = np.array(pathway_features)

    cond_dummies = pd.get_dummies(adata_lfc.obs['condition'], dtype=float)
    
    # 1. Standard Biological Features
    X_features = np.column_stack([
        np.array(basal_ko_values).reshape(-1, 1), 
        cond_dummies.values,
        pathway_features
    ])
    
    # 2. Random Baseline Features (Condition + Gaussian Noise)
    np.random.seed(42)
    noise_dim = 50
    X_random = np.column_stack([
        cond_dummies.values,
        np.random.normal(0, 1, size=(adata_lfc.n_obs, noise_dim))
    ])
    
    # 3. Idealized Baseline Features (Condition + PCA of Target Y - Data Leakage)
    pca_ideal = PCA(n_components=min(50, adata_lfc.X.shape[0], adata_lfc.X.shape[1]), random_state=42)
    target_pca = pca_ideal.fit_transform(adata_lfc.X)
    X_idealized = np.column_stack([
        cond_dummies.values,
        target_pca
    ])
    
    adata_lfc.obsm['X_features'] = X_features
    adata_lfc.obsm['X_random'] = X_random
    adata_lfc.obsm['X_idealized'] = X_idealized
    
    print(f"Biological Feature matrix shape: {X_features.shape}")
    print(f"Random Feature matrix shape: {X_random.shape}")
    print(f"Idealized Feature matrix shape: {X_idealized.shape}")
    
    adata_lfc.write_h5ad('../processed_data/task3_lfc_features.h5ad')
    print("Saved to ../processed_data/task3_lfc_features.h5ad")

if __name__ == "__main__":
    main()