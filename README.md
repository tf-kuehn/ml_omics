This is the codespace of Thaddeus, Deniz and Antonia participating in the Practical Machine
Learning for Single-Cell Multiomics course.

This readme outlines the project we did on the topic "Modeling CRISPR perturbations in melanoma cells
under different experimental conditions". For that the CRISPR perturbation dataset by [Frangieh et al.](https://www.nature.com/articles/s41588-021-00779-1),
2021 was used.

## Tasks
### Task 1: Condition Classification
**Conditions**: 
- Control: Maintained in culture medium
- IFN𝞬: Treated with interferon-𝞬
- Co-culture: Co-cultured with tumor-infiltrating lymphocytes (TILs) for 48h

**Objective**:
- predict treatment condition of cells from their expression profile

**Plan**:
- train 3 classification models (logistic regression, tree ensembles or neural
network)
- evaluate and compare performances
- compute feature importance and discuss driver genes

### Task 2: Clustering
**Objective**:
- Apply different clustering methods to the data, visualize and compare the results

**Plan**:
- reduce dimensions with PCA, VAE or scVI
- cluster the data with leiden, K-means, Hierarchical Clustering, Ensemble Clustering, neural network
- visualize with UMAP and t-SNE
- used all different conditions
- compare the clusters with the findings in the paper (pathway expression)

### Task 3:  Perturbation Prediction
**Objective**:
- predict transcriptome changes for a gene knockout ood

**Plan**:
- select only 50 perturbations based on clustering them
- include a baseline control as comparison
- target variable: log2 fold change per condition and perturbation
- train 4 different model types (combination of feature engineering stratagy and learning algorithm)
- one very simple model (linear)
- evaluate performance and uncertainty
- bias-variance tradeoff
- give Hypotheses for differences between models
- integrate the protein data via MOFA?


## Pipeline
### Preprocessing
- Quality control
- log transform and normalize
- HvG selection
- log2 fold change calculation
### Dimension Reduction
- PCA
- MOFA
### Experimental design & Data splitting
- cross validation with train and test set
### Condition Classification
- Input: train/test split preprocessed gene expression data
- Methods: Logistic regression, Tree ensembles, Neural networks
- Output: predicted treatment condition, evaluation metrics, feature importance (driver genes compare paper)
### Clustering
- Input: dimension reduced data
- Methods:K-means, leiden, Hierarchical
- Output: UMAP, t-SNE, cluster assignments, classification metrics (sensitivity, specificity...), comparison to paper pathways
### Perturbation Prediction
- Input (Features): train/test split of subset of 50 perturbations represented as: [One-Hot Condition] + [One-Hot Perturbation] or [Average MOFA Embedding of the Unperturbed Cells in Condition Y] + [One-Hot Perturbation]
- Do we train the model on all conditions or train one model per condition?
- Data splitting strategy: random, cluster based (perturbation), condition?
- Methods: knn, lasso, ridge regression, neural net
- Output (Target): mean log2 fold change per condition and perturbation/ evaluation metrics (Uncertainty, Baselines)

### Further options:
- hyperparameter tuning