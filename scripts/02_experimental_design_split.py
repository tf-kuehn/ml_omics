import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import gseapy as gp
import pickle
import os

def main():
    os.makedirs("../processed_data", exist_ok=True)
    
    print("Loading engineered features and raw data...")
    adata_lfc = sc.read_h5ad('../processed_data/task3_lfc_features.h5ad')
    
    try:
        adata_pre = sc.read_h5ad('../data/frangieh/adata_preprocessed.h5ad')
        has_pre = True
    except FileNotFoundError:
        print("Warning: adata_preprocessed.h5ad not found. Falling back to LFC counts.")
        has_pre = False

    eligible_genes = [g for g in adata_lfc.obs["perturbation"].unique() if str(g).lower() != 'control']
    all_strategies = {}

    print("--- Strategy A: Essentiality (Top 50 Frequent) ---")
    if has_pre:
        pert_counts = adata_pre.obs[adata_pre.obs['perturbation'] != 'control']['perturbation'].value_counts()
    else:
        pert_counts = pd.Series(eligible_genes).value_counts() 
    
    top50_perts = pert_counts.head(50).index.tolist()
    all_strategies['essentiality'] = top50_perts

    print("--- Strategy B: Random (50 Random) ---")
    np.random.seed(42)
    all_perts = pert_counts.index.tolist()
    random50_perts = list(np.random.choice(all_perts, size=50, replace=False))
    all_strategies['random'] = random50_perts

    print("--- Strategy C: Highest LFC ---")
    df_lfc_all = adata_lfc.to_df()
    df_lfc_all['perturbation'] = adata_lfc.obs['perturbation'].values
    df_lfc_all['mean_abs_lfc'] = df_lfc_all.drop(columns='perturbation').abs().mean(axis=1)
    pert_strength = df_lfc_all.groupby('perturbation')['mean_abs_lfc'].max()
    top_lfc50_perts = pert_strength.sort_values(ascending=False).head(50).index.tolist()
    all_strategies['highest_lfc'] = top_lfc50_perts

    print("--- Strategy D: Leiden Clusters ---")
    conds = ['Control', 'IFNγ', 'Co-culture']
    dfs = []
    for cond in conds:
        mask = adata_lfc.obs['condition'] == cond
        sub_df = adata_lfc[mask].to_df()
        sub_df.index = adata_lfc[mask].obs['perturbation'].values
        sub_df = sub_df.add_suffix(f"_{cond}")
        dfs.append(sub_df)

    df_lfc_all_conds = pd.concat(dfs, axis=1).fillna(0)
    adata_perts_combined = ad.AnnData(
        X=df_lfc_all_conds.values,
        obs=pd.DataFrame({'perturbation': df_lfc_all_conds.index}, index=df_lfc_all_conds.index),
    )
    
    sc.pp.pca(adata_perts_combined, n_comps=30)
    sc.pp.neighbors(adata_perts_combined, n_pcs=8)
    sc.tl.leiden(adata_perts_combined, resolution=0.6, flavor='igraph', directed=False)
    
    np.random.seed(42)
    cluster_series = adata_perts_combined.obs['leiden']
    n_clusters = cluster_series.nunique()
    samples_per_cluster = int(np.ceil(50 / n_clusters))
    
    cluster50_perts = []
    for cluster_id in sorted(cluster_series.unique()):
        cluster_perts = cluster_series[cluster_series == cluster_id].index.tolist()
        n_sample = min(samples_per_cluster, len(cluster_perts))
        chosen = np.random.choice(cluster_perts, size=n_sample, replace=False).tolist()
        cluster50_perts.extend(chosen)
        
    if len(cluster50_perts) > 50:
        cluster50_perts = np.random.choice(cluster50_perts, size=50, replace=False).tolist()
    elif len(cluster50_perts) < 50:
        remaining = [p for p in adata_perts_combined.obs['perturbation'] if p not in cluster50_perts]
        needed = 50 - len(cluster50_perts)
        cluster50_perts.extend(np.random.choice(remaining, size=needed, replace=False).tolist())
    all_strategies['leiden_clusters'] = cluster50_perts

    print("--- Strategy E: Hallmark Pathways ---")
    hallmark = gp.get_library(name="MSigDB_Hallmark_2020", organism="Human")
    hallmark_upper = {pathway: {str(g).upper() for g in genes} for pathway, genes in hallmark.items()}
    
    pathway_matrix = pd.DataFrame(0, index=eligible_genes, columns=hallmark_upper.keys(), dtype=int)
    for gene in eligible_genes:
        g_up = str(gene).upper()
        for pathway, p_genes in hallmark_upper.items():
            if g_up in p_genes:
                pathway_matrix.loc[gene, pathway] = 1

    rng = np.random.default_rng(42)
    annotated_genes = pathway_matrix.index[pathway_matrix.sum(axis=1) > 0].tolist()
    
    pathway50_perts = []
    covered_pathways = set()
    remaining = annotated_genes.copy()
    
    while len(pathway50_perts) < 50 and len(remaining) > 0:
        gains = {}
        for gene in remaining:
            gene_pathways = set(pathway_matrix.columns[pathway_matrix.loc[gene] == 1])
            new_pathways = gene_pathways - covered_pathways
            gains[gene] = len(new_pathways)
            
        max_gain = max(gains.values())
        if max_gain == 0:
            break
            
        best_genes = [g for g, gain in gains.items() if gain == max_gain]
        chosen = rng.choice(best_genes)
        
        pathway50_perts.append(chosen)
        covered_pathways.update(pathway_matrix.columns[pathway_matrix.loc[chosen] == 1])
        remaining.remove(chosen)

    if len(pathway50_perts) < 50:
        extra_candidates = [g for g in eligible_genes if g not in pathway50_perts]
        extra = rng.choice(extra_candidates, size=50 - len(pathway50_perts), replace=False)
        pathway50_perts.extend(extra.tolist())
    all_strategies['hallmark_pathways'] = pathway50_perts

    print("--- Generating Train/Test Splits ---")
    final_splits = {}
    for strategy_name, perts_50 in all_strategies.items():
        np.random.seed(42)
        # Random 40/10 split
        test_genes_10 = np.random.choice(perts_50, size=10, replace=False).tolist()
        train_genes_40 = [p for p in perts_50 if p not in test_genes_10]
        
        final_splits[strategy_name] = {
            "train_genes": train_genes_40,
            "test_genes": test_genes_10,
            "all_50": perts_50
        }
    
    with open('../processed_data/task3_splits.pkl', 'wb') as f:
        pickle.dump(final_splits, f)
    print("Saved all multi-strategy splits to ../processed_data/task3_splits.pkl")

if __name__ == "__main__":
    main()