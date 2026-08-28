# Complete, well-commented, runnable code for this single file
import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp
import os
import mudata as md
import muon as mu
from sklearn.decomposition import PCA

def main():
    """
    Executes the feature engineering pipeline.
    Loads LFC data, integrates biological features (pathways, conditions),
    generates baselines (Random, Idealized), and incorporates CITE-seq
    protein data via MOFA to establish the unperturbed baseline state.
    """
    os.makedirs("../processed_data", exist_ok=True)
    
    print("Loading preprocessed and LFC data...")
    try:
        adata_pre = sc.read_h5ad('../data/frangieh/adata_preprocessed.h5ad')
        adata_lfc = sc.read_h5ad('../data/frangieh/log_fold_change_adata_hvg.h5ad')
    except FileNotFoundError as e:
        print(f"Error loading required data files: {e}")
        print("Please ensure you have run the data preprocessing steps first.")
        return

    print("Calculating basal expression...")
    # Filter original data to only include control cells (no perturbation)
    ctrl_cells = adata_pre[adata_pre.obs['perturbation'].str.lower() == 'control'].copy()
    
    basal_means = {}
    # Calculate the mean expression profile for the control cells in each condition
    for cond in adata_lfc.obs['condition'].unique():
        cells_in_cond = ctrl_cells[ctrl_cells.obs['condition'] == cond]
        if cells_in_cond.n_obs > 0:
             # Handle both sparse and dense matrix formats safely
            if hasattr(cells_in_cond.X, 'todense') or pd.api.types.is_sparse(cells_in_cond.X):
                mean_expr = np.asarray(cells_in_cond.X.mean(axis=0)).ravel()
            else:
                mean_expr = cells_in_cond.X.mean(axis=0)
            basal_means[cond] = pd.Series(mean_expr, index=adata_pre.var_names)
        else:
            basal_means[cond] = pd.Series(0.0, index=adata_pre.var_names)
            
    # Extract the baseline expression value specifically for the knocked-out gene
    basal_ko_values = []
    for _, row in adata_lfc.obs.iterrows():
        cond = row['condition']
        ko_gene = row['perturbation']
        
        # If the knocked-out gene exists in our dataset, get its baseline expression
        if ko_gene in basal_means[cond].index:
            val = basal_means[cond][ko_gene]
        else:
            val = 0.0
        basal_ko_values.append(val)
        
    print("Mapping Hallmark Pathways...")
    try:
        # Fetch the MSigDB Hallmark 2020 gene sets
        hallmark = gp.get_library(name="MSigDB_Hallmark_2020", organism="Human")
    except Exception as e:
        print(f"Failed to download or load gseapy library: {e}")
        # Fallback to an empty dictionary if network fails
        hallmark = {}
        
    # Convert all gene symbols in the library to uppercase for robust matching
    hallmark_upper = {pathway: {str(g).upper() for g in genes} for pathway, genes in hallmark.items()}
    
    # Identify unique targeted genes (excluding controls)
    eligible_genes = [g for g in adata_lfc.obs["perturbation"].unique() if str(g).lower() != 'control']
    
    # Create a binary matrix indicating pathway membership for each gene
    pathway_matrix = pd.DataFrame(0, index=eligible_genes, columns=hallmark_upper.keys(), dtype=int)
    for gene in eligible_genes:
        g_up = str(gene).upper()
        for pathway, p_genes in hallmark_upper.items():
            if g_up in p_genes:
                pathway_matrix.loc[gene, pathway] = 1
                
    # Build the final pathway feature set for every row in the dataset
    pathway_features = []
    for pert in adata_lfc.obs['perturbation']:
        if pert in pathway_matrix.index:
            pathway_features.append(pathway_matrix.loc[pert].values)
        else:
            pathway_features.append(np.zeros(len(pathway_matrix.columns)))
    pathway_features = np.array(pathway_features)

    print("Constructing biological and baseline feature matrices...")
    # One-hot encode the experimental conditions
    cond_dummies = pd.get_dummies(adata_lfc.obs['condition'], dtype=float)
    
    # Primary Biological Features: Baseline KO value + Condition + Pathway Vector
    X_features = np.column_stack([
        np.array(basal_ko_values).reshape(-1, 1), 
        cond_dummies.values,
        pathway_features
    ])
    
    np.random.seed(42)
    # The noise dimension is calculated to match the length of the pathway features plus the basal value
    noise_dim = pathway_features.shape[1] + 1 
    
    # Random Control Baseline: Condition labels concatenated with random Gaussian noise
    X_random = np.column_stack([
        cond_dummies.values,
        np.random.normal(0, 1, size=(adata_lfc.n_obs, noise_dim))
    ])
    
    # Idealized Control Baseline (Upper Bound): Condition labels + Target PCA
    # We fit PCA on the entire target matrix. This intentionally causes data leakage
    # to serve as an absolute upper-bound benchmark for linear/tree models.
    # Note: We must restrict n_components to min(samples, features) to prevent crashing on small test sets later.
    pca_ideal = PCA(n_components=min(50, adata_lfc.X.shape[0], adata_lfc.X.shape[1]), random_state=42)
    target_pca = pca_ideal.fit_transform(adata_lfc.X)
    
    X_idealized = np.column_stack([
        cond_dummies.values,
        target_pca
    ])
    
    print("Loading protein data for MOFA integration...")
    try:
        # Load the raw multi-omics object that contains protein counts
        adata_prot = sc.read_h5ad('../data/frangieh/rna.h5ad')
        # Standardize the condition column name to match the RNA processing
        if "perturbation_2" in adata_prot.obs.columns:
            adata_prot.obs.rename(columns={"perturbation_2": "condition"}, inplace=True)
            
        # Identify cells present in both the preprocessed RNA and the loaded Protein data
        common_cells = adata_pre.obs_names.intersection(adata_prot.obs_names)
        print(f"Found {len(common_cells)} common cells between RNA and Protein data.")
        
        if len(common_cells) > 0:
            # OPTIMIZATION 1: Filter to control cells only. 
            # MOFA's job here is solely to embed the *baseline* state before perturbation.
            ctrl_mask = (adata_pre[common_cells].obs['perturbation'].str.lower() == 'control')
            ctrl_cell_names = common_cells[ctrl_mask]
            
            # OPTIMIZATION 2: Subsample to prevent Memory Errors
            if len(ctrl_cell_names) > 10000:
                print("Subsampling to 10,000 control cells to optimize MOFA training...")
                np.random.seed(42)
                ctrl_cell_names = np.random.choice(ctrl_cell_names, size=10000, replace=False)
                
            # OPTIMIZATION 3: Restrict RNA features to only the LFC Highly Variable Genes
            adata_pre_sub = adata_pre[ctrl_cell_names, adata_lfc.var_names].copy()
            adata_prot_sub = adata_prot[ctrl_cell_names].copy()
            
            # Create a MuData object combining RNA and Protein modalities
            mdata = md.MuData({'rna': adata_pre_sub, 'prot': adata_prot_sub})
            
            print("Training MOFA+ model with optimized parameters...")
            # OPTIMIZATION 4: Fast training hyperparameters (float32, 10 factors, 500 iterations max)
            mu.tl.mofa(
                mdata, 
                n_factors=10, 
                outfile='../processed_data/mofa_model.hdf5',
                use_float32=True,
                n_iterations=500
            )
            
            # Transfer the condition label from the RNA modality to the global MuData obs
            # FIX: Access the RNA modality directly using .mod['rna']
            mdata.obs['condition'] = mdata.mod['rna'].obs['condition'].values
            mofa_means = {}
            
            # Aggregate the learned MOFA factors (X_mofa) by condition
            for cond in adata_lfc.obs['condition'].unique():
                cond_cells = mdata[mdata.obs['condition'] == cond]
                if cond_cells.n_obs > 0:
                    mofa_means[cond] = cond_cells.obsm['X_mofa'].mean(axis=0)
                else:
                    mofa_means[cond] = np.zeros(10)
                    
            # Map the aggregated MOFA baseline back to the dataset rows
            mofa_features = []
            for _, row in adata_lfc.obs.iterrows():
                cond = row['condition']
                mofa_features.append(mofa_means[cond])
            mofa_features = np.array(mofa_features)
            
            # Construct the MOFA-enriched feature matrix
            X_mofa_features = np.column_stack([
                np.array(basal_ko_values).reshape(-1, 1), 
                mofa_features, # Multi-omics baseline factors
                pathway_features # Biological pathway annotations
            ])
        else:
            print("Warning: No common cells found. Skipping MOFA integration.")
            X_mofa_features = X_features # Fallback to standard features
            
    except Exception as e:
        print(f"Error during MOFA integration: {e}. Skipping MOFA step.")
        X_mofa_features = X_features # Fallback to standard features
    
    # Attach all feature matrices to the AnnData obsm dictionary
    adata_lfc.obsm['X_features'] = X_features
    adata_lfc.obsm['X_random'] = X_random
    adata_lfc.obsm['X_idealized'] = X_idealized
    adata_lfc.obsm['X_mofa_features'] = X_mofa_features
    
    print(f"\nFeature Matrices Constructed:")
    print(f"- Biological Features (X_features): {X_features.shape}")
    print(f"- Random Baseline (X_random):       {X_random.shape}")
    print(f"- Idealized Baseline (X_idealized): {X_idealized.shape}")
    print(f"- MOFA Enriched (X_mofa_features):  {X_mofa_features.shape}")
    
    # Save the updated AnnData object
    adata_lfc.write_h5ad('../processed_data/task3_lfc_features.h5ad')
    print("\nSuccessfully saved LFC object with engineered features to ../processed_data/task3_lfc_features.h5ad")

if __name__ == "__main__":
    main()