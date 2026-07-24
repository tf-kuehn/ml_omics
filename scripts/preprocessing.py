import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os

#Load the data
data_dir = "../data/frangieh"
output_dir = "../results/preprocessing"
os.makedirs(output_dir, exist_ok=True)

def save_figure(filename):
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
    plt.close()
    
adata = sc.read_h5ad(f"{data_dir}/rna.h5ad")

adata.obs.rename({"perturbation_2": "condition"}, axis=1, inplace=True)

# create a dataframe to store cell and gene counts before and after filtering
data_shape = pd.DataFrame(index=['pre_filtering', 'post_filtering1', 'post_filtering2', 'post_filtering3', 'post_doublet_removal'], columns=['cells', 'genes'])
data_shape.loc['pre_filtering'] = adata.shape[0], adata.shape[1]

# QC steps via scanpy workflow
# mitochondrial genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

#Violin plots for qc values
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    show=False,
)
save_figure("qc_violin_pre_filtering.png")

sc.pl.scatter(
    adata,
    "total_counts",
    "n_genes_by_counts",
    color="pct_counts_mt",
    show=False,
)
save_figure("qc_scatter_pre_filtering.png")

#Filter cells out that express less then 100 genes and genes that are expressed in less then 3 cells
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)

#Remove cells with high mitochondrial counts
adata = adata[adata.obs.pct_counts_mt < 15].copy()

# Visualize after filtering the cells and the genes
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    show=False,
)
save_figure("qc_violin_post_filtering1.png")

data_shape.loc['post_filtering1'] = adata.shape[0], adata.shape[1]

# Remove cells with high total counts
adata = adata[adata.obs.total_counts < 40000].copy()

sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    show=False,
)
save_figure("qc_violin_post_filtering2.png")

data_shape.loc['post_filtering2'] = adata.shape[0], adata.shape[1]

#Remove the too low genes by counts
sc.pp.filter_cells(adata, min_counts=200)

sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    show=False,
)
save_figure("qc_violin_post_filtering3.png")

data_shape.loc['post_filtering3'] = adata.shape[0], adata.shape[1]

adata.write(f"{data_dir}/adata_post_filtering.h5ad")

# doublet detection
# split per batch into new objects.
import scrublet as scr
import pandas as pd
import scipy.sparse as sp

batches = adata.obs["condition"].cat.categories.tolist()
alldata = {}

for batch in batches:
    tmp = adata[adata.obs["condition"] == batch].copy()
    print(batch, ":", tmp.shape[0], "cells")

    # Use raw counts
    X = tmp.raw.X if tmp.raw is not None else tmp.X

    scrub = scr.Scrublet(X)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        verbose=False,
        n_prin_comps=20
    )

    alldata[batch] = pd.DataFrame(
        {
            "doublet_score": doublet_scores,
            "predicted_doublets": predicted_doublets
        },
        index=tmp.obs.index
    )

    print(predicted_doublets.sum(), "predicted doublets")

# add predictions to the adata object.
scrub_pred = pd.concat(alldata.values())
adata.obs['doublet_scores'] = scrub_pred['doublet_score'] 
adata.obs['predicted_doublets'] = scrub_pred['predicted_doublets'] 

# Visualize doublet scores and predicted doublets
adata.obs['doublet_info'] = adata.obs["predicted_doublets"].astype(str)
sc.pl.violin(
    adata,
    'n_genes_by_counts',
    jitter=0.4,
    groupby='doublet_info',
    rotation=45,
    show=False,
)
save_figure("qc_violin_doublet_scores.png")

# remove predicted doublets
adata = adata[adata.obs['predicted_doublets'] == False].copy()

data_shape.loc["post_doublet_removal"] = adata.shape[0], adata.shape[1]

adata.write(f"{data_dir}/adata_qc_done.h5ad")



# Save the counts in a separate layer
adata.layers["counts"] = adata.X.copy()

# Normalizing to median total counts
sc.pp.normalize_total(adata, target_sum=1e4)
# Logarithmize the data
sc.pp.log1p(adata)

# Highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

sc.pl.highly_variable_genes(adata, show=False)
save_figure("highly_variable_genes.png")

#Dimensionality reduction PCA
sc.tl.pca(adata)

# Save the data
adata.write_h5ad("../data/frangieh/adata_preprocessed.h5ad")
data_shape.to_csv(f"{output_dir}/qc_shapes.csv")