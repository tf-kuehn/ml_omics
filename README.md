# Modeling CRISPR Perturbations in Melanoma Cells

This repository contains the coursework project of Thaddeus, Deniz, and Antonia for the **Practical Machine Learning for Single-Cell Multiomics** course.

The project uses the CRISPR perturbation dataset from Frangieh et al. (2021) to investigate condition classification, cell clustering, and perturbation-response prediction in melanoma cells.

## Dataset

The dataset contains melanoma cells measured under three experimental conditions:

- **Control** — cells maintained in culture medium
- **IFNγ** — cells treated with interferon gamma
- **Co-culture** — cells co-cultured with tumor-infiltrating lymphocytes for 48 hours


## Project Tasks

### 1. Condition Classification

**Objective:** Predict the experimental condition from gene-expression profiles.

The project compares several machine-learning approaches, including:

- Logistic regression `t1_logistic_regression.ipynb`
- Random forest `t1_RF.ipynb`
- Neural Networks / Multilayer perceptron `t1_RF.ipynb`

The results include:

- Classification performance
- Confusion matrices
- Feature importance
- Candidate driver genes for each condition


### 2. Clustering

**Objective:** Identify cellular structure and compare clustering methods.

The following clustering methods are includes:

- K-means clustering `t2_k-means.ipynb`
- Leiden clustering `t2_leiden.ipynb`
- Hierarchical clustering `t2_hierachical.ipynb`

### 3. Perturbation Prediction

**Objective:** Predict transcriptomic changes caused by gene perturbations.

The prediction target is the mean log fold change associated with a perturbation under a given condition.



## Analysis Pipeline

1. Quality control
2. Normalization and log transformation
3. Highly variable gene selection
4. Log fold-change calculation
5. Dimension reduction
6. Model training and validation
7. Evaluation and visualization
8. Biological interpretation

## Repository Structure

```text
.
├── README.md
├── scripts/
│   ├── 00_preprocessing.py
│   ├── eda_rna.ipynb
│   ├── Project_data_exploration.ipynb
│   ├── Project_data_qc.ipynb
│   ├── t1_logistic_regression.ipynb
│   ├── t1_MLP.ipynb
│   ├── t1_RF.ipynb
│   ├── t2_hierachical.ipynb
│   ├── t2_k-means.ipynb
│   ├── t2_leiden.ipynb
│   ├── t3_selection_of_50_pertubations.ipynb
│   ├── t3_approach1.ipynb
│   ├── t3_approach2.ipynb
│   ├── t3_approach3_01_feature_engineering.py
│   ├── t3_approach3_01_feature_engineering_MOFA.py
│   ├── t3_approach3_02_experimental_design_split.py
│   ├── t3_approach3_03_model_training.py
│   ├── t3_approach3_03_model_training_MOFA.py
│   └── t3_approach3_04_evaluation_plotting.py
└── results/
    ├── preprocessing/
    ├── t1/
    └── t3/
```

## Discussion

### Task 1: Condition Classification

### Task 2: Clustering

### Task 3: Perturbation Prediction