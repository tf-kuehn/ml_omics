import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp
import os
from sklearn.decomposition import PCA

def main():
    """
    Constructs feature matrices for Task 3. Generates standard biological features alongside rigorous negative (random) and positive (idealized) controls.
    """
    os.makedirs("../processed_data", exist_ok=True)
    
    print("Loading data...")
    # Load the preprocessed single-cell data and the aggregated LFC targets
    adata_pre = sc.read_h5ad('../data/frangieh/adata_preprocessed.h5ad')
    adata_lfc = sc.read_h5ad('../data/frangieh/log_fold_change_adata_hvg.h5ad')
    
    print("Calculating basal expression...")
    # Isolate unperturbed control cells to establish the baseline expression state
    ctrl_cells = adata_pre[adata_pre.obs['perturbation'] == 'control']
    df_ctrl = ctrl_cells.to_df()
    df_ctrl['condition'] = ctrl_cells.obs['condition'].values
    mean_ctrl_by_cond = df_ctrl.groupby('condition').mean()
    
    # Extract the scalar basal expression value of the specific knockout gene for each experimental condition
    basal_ko_values = []
    for _, row in adata_lfc.obs.iterrows():
        cond = row['condition']
        ko_gene = row['perturbation']
        # Fallback to 0.0 if the target KO gene isn't captured in the baseline expression matrix
        val = mean_ctrl_by_cond.loc[cond, ko_gene] if ko_gene in mean_ctrl_by_cond.columns else 0.0
        basal_ko_values.append(val)
        
    print("Mapping Hallmark Pathways...")
    # Fetch MSigDB Hallmark pathways to provide the models with biological context/priors
    hallmark = gp.get_library(name="MSigDB_Hallmark_2020", organism="Human")
    # Standardize gene names to uppercase to ensure robust matching across datasets
    hallmark_upper = {pathway: {str(g).upper() for g in genes} for pathway, genes in hallmark.items()}
    
    # Build a binary perturbation-by-pathway multi-hot matrix
    eligible_genes = [g for g in adata_lfc.obs["perturbation"].unique() if str(g).lower() != 'control']
    pathway_matrix = pd.DataFrame(0, index=eligible_genes, columns=hallmark_upper.keys(), dtype=int)
    for gene in eligible_genes:
        g_up = str(gene).upper()
        for pathway, p_genes in hallmark_upper.items():
            if g_up in p_genes:
                pathway_matrix.loc[gene, pathway] = 1

    # Map the multi-hot pathway vectors back to LFC observations     
    pathway_features = []
    for pert in adata_lfc.obs['perturbation']:
        if pert in pathway_matrix.index:
            pathway_features.append(pathway_matrix.loc[pert].values)
        else:
            # Default to all zeros if the perturbation isn't annotated in the Hallmark database
            pathway_features.append(np.zeros(len(pathway_matrix.columns)))
    pathway_features = np.array(pathway_features)

    # One-hot encode the experimental conditions (e.g., Control, IFNγ, Co-culture)
    cond_dummies = pd.get_dummies(adata_lfc.obs['condition'], dtype=float)
    
    # 1. Standard Biological Features
    # Combines basal target expression, condition flags, and pathway mappings
    X_features = np.column_stack([
        np.array(basal_ko_values).reshape(-1, 1), 
        cond_dummies.values,
        pathway_features
    ])
    
    # 2. Random Baseline Features (Condition + Gaussian Noise)
    # Negative control replacing biological priors with pure 50-dimensional noise to test for actual predictive signal
    np.random.seed(42)
    noise_dim = 50
    X_random = np.column_stack([
        cond_dummies.values,
        np.random.normal(0, 1, size=(adata_lfc.n_obs, noise_dim))
    ])
    
    # 3. Idealized Baseline Features (Condition + PCA of Target Y - Data Leakage)
    # Positive control simulating data leakage to establish the theoretical upper limit of model accuracy
    pca_ideal = PCA(n_components=min(50, adata_lfc.X.shape[0], adata_lfc.X.shape[1]), random_state=42)
    target_pca = pca_ideal.fit_transform(adata_lfc.X)
    X_idealized = np.column_stack([
        cond_dummies.values,
        target_pca
    ])
    
    # Store the engineered feature matrices directly within the AnnData object's multidimensional observation mapping
    adata_lfc.obsm['X_features'] = X_features
    adata_lfc.obsm['X_random'] = X_random
    adata_lfc.obsm['X_idealized'] = X_idealized
    
    print(f"Biological Feature matrix shape: {X_features.shape}")
    print(f"Random Feature matrix shape: {X_random.shape}")
    print(f"Idealized Feature matrix shape: {X_idealized.shape}")
    
    # Save the fully annotated object for downstream modeling scripts
    adata_lfc.write_h5ad('../processed_data/task3_lfc_features.h5ad')
    print("Saved to ../processed_data/task3_lfc_features.h5ad")

if __name__ == "__main__":
    main()