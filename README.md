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

**Scripts**
3 Apporaches by 3 people:
1. `t3_approach1.ipynb`
2. `t3_approach2.ipynb`
3. `t3_approach3_01_feature_engineering.py, t3_approach3_02_experimental_design_split.py, t3_approach3_03_model_training.py, t3_approach3_04_evaluation_plotting.py`

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
To determine whether transcriptomic profiles are sufficient to predict the experimental conditions (Control, IFNγ and Co-culture), three distinct machine learning approaches were evaluated: a Random Forest classifier, a logistic regression model and a neural network. 

As an initial approach, a Random Forest classifier, an ensemble of decision trees, was trained to predict the condition labels. Several hyperparameters were varied during model optimization, including the number of trees, maximum tree depth, minimum number of observations required to split a node, minimum number of observations in a leaf and maximum number of features considered at each split. Two Random Forest approaches were evaluated: one model was trained directly on the highly variable gene expression features without PCA, while a second model was trained on a dimensionality-reduced representation using the first 50 principal components. After hyperparameter optimization, the best-performing model achieved approximately 98% accuracy on both the training and test sets, demonstrating strong predictive performance and little evidence of overfitting.

Afterwards, the top 20 genes contributing to condition prediction were examined. Among the top features were GBP1, WARS, CD74, IDO1, and STAT1, genes associated with IFN responses, MHC-I and MHC-II antigen presentation, and JAK/STAT signaling. To further investigate these features, their expression patterns were visualized using a heatmap, which showed clear differences across the experimental conditions. The Control condition generally showed lower expression of many of these genes, whereas the IFNγ and Co-culture conditions displayed distinct expression patterns. The IFNγ condition showed a strong interferon-response pattern, with higher expression of genes such as STAT1, GBP1, GBP2, WARS, B2M, and TAP1. The Co-culture condition also showed elevated expression of several immune-response and antigen-presentation-related genes, including HLA-A, B2M, and HLA-B. PFN1, another important feature, is involved in actin cytoskeleton organization and may reflect differences in cellular state between the conditions. 

As a linear baseline model, a logistic regression classifier was trained to predict the three conditions. First, the dataset was filtered to the top 2,000 highly variable genes (HVGs). The data was then split into 80% for training and 20% for testing using stratified sampling. The model showed very high performance on the test set with an overall accuracy of 99%. Co-cultured cells were predicted with a perfect precision of 1 and recall of 1, the control condition was predicted with a F1-score of 0.97 and the IFNγ condition achieved an F1-score of 0.98. These results show that the three conditions have clearly distinct gene expression patterns that can be separated well by a linear model.

Since the gene expression data was standardized before training, the learned regression coefficients directly indicate the most important genes for each treatment condition. For the IFNγ condition, the top positive coefficients correspond to classic interferon-response genes and inflammatory markers, such as IDO1, T-cell recruiting chemokines like CXCL11, CXCL9 and CXCL10 and the transcription factor IRF1. In the Co-culture condition, the top ranking genes are dominated by essential components of the antigen presentation machinery and immune recognition, most notably HLA-B and HLA-A, alongside factors like PFN1, illustrating the direct interaction with tumor-infiltrating lymphocytes. In contrast, the Control condition is primarily defined by basal structural, extracellular matrix and homeostatic markers, such as FN1, KRT18, CD63 and PMEPA1, which characterize the unperturbed baseline state in the absence of inflammatory stimulation.

To complete the evaluation of transcriptomic predictability across the Control, IFNγ, and Co-culture conditions, a neural network approach was implemented. A custom PyTorch Multi-Layer Perceptron was trained with batch normalization and dropout layers to prevent overfitting on the highly sparse single-cell data. The model demonstrated excellent predictive performance, achieving over 98% accuracy and strong F1-scores across all three condition classes. Permutation feature importance analysis revealed that the neural network's predictions were heavily driven by genes involved in antigen presentation and the IFN-γ response, including STAT1, IRF1, HLA-DRA, and IDO1. Biologically, this closely mirrors the findings of Frangieh et al., where IFN-γ stimulation was shown to strongly upregulate these specific immune and inflammatory programs compared to the baseline control state. Furthermore, the model's high ranking of antigen presentation components highlights the distinct transcriptomic shifts occurring in the Co-culture condition due to direct cellular interactions with T-cells. Ultimately, the neural network's robust performance and biologically sound feature importances confirm that the experimental treatments induce clearly separable, pathway-specific gene expression profiles.


### Task 2: Clustering
The objective of Task 2 was to identify clusters of genetic perturbations that exhibit similar transcriptional phenotypes across different microenvironmental conditions. To evaluate which method best captures the underlying biology, k-means clustering, hierarchical agglomerative clustering and graph-based Leiden clustering were applied to the pseudobulk log2​fold change profiles across the 2,000 most highly variable genes.

First, k-means clustering was evaluated, with the optimal number of clusters (k) determined via elbow and silhouette analyses. In the stimulated conditions (IFNγ and Co-culture), perturbations targeting the IFN-γ signaling pathway (JAK1, STAT1, IFNGR1, IFNGR2) grouped closely together, aligning with the findings of Frangieh et al. (2021). However, outside this dominant axis, several pathway-related genes were dispersed across broader, poorly defined clusters, indicating that centroid-based partitioning only partially resolved the phenotypic landscape.

The hierarchical clustering of pseudobulk log-fold changes similarly grouped key JAK/STAT pathway members into a distinct cluster. Biologically, knocking out these specific genes disables the IFN-γ signaling cascade, causing the cells to fail to respond to treatment and cluster closer to unperturbed control cells. The majority of non-essential or non-impactful perturbations formed a dense, neutral cluster reflecting a lack of significant transcriptomic shifts relative to the control. Overall, this clustering approach successfully captures the underlying biological architecture by segregating genetic perturbations into functionally meaningful pathways. 

Graph-based leiden clustering with a resolution of 0.5 applied to the PCA representations also successfully resolved distinct biological perturbation signatures. Most notably, under immune stimulation (IFNγ and Co-culture), Leiden clustering grouped the core IFNγ pathway genes (IFNGR1, IFNGR2, JAK1, JAK2, and STAT1) into a single cluster. Differential expression testing with Welch's t-test confirmed that this cluster showed a strong downregulation of key interferon response and antigen-presenting genes, such as GBP1, GBP2, IRF1, WARS, B2M and MHC Class II genes. Importantly, this pathway-level segregation occurred exclusively under immune pressure, either in co-culture condition or IFNγ treatment. In the control condition, where the IFNγ pathway is transcriptionally quiescent, these genes did not separate into an isolated group.

Overall, all three clustering strategies successfully captured the dominant biological signal of the JAK/STAT axis under immune pressure, while providing different levels of resolution for the remaining perturbation landscape.

### Task 3: Perturbation Prediction
Task 3 investigates how accurately machine learning models can predict whole-transcriptome log2​ fold changes induced by unseen genetic perturbations. To evaluate model generalizability, five distinct strategies were used to select representative subsets of 50 perturbations: most-frequent, random, highest log2​ fold change, Leiden clustering and pathway coverage. For the pathway-coverage strategy, eligible genes were annotated using MSigDB Hallmark gene sets to construct a binary perturbation-by-pathway matrix, allowing a greedy algorithm to maximize the representation of distinct biological pathways across the selected knockouts. For each selection strategy, the 50 perturbations were divided into 40 training and 10 held-out test perturbations. The transcriptional response of each perturbation under each condition was represented as the mean log2 fold-change of 2,000 HVGs relative to the corresponding condition-matched unperturbed population. To enable supervised modeling, several feature engineering strategies were designed and tested to translate qualitative metadata into numerical representations that provide the model with essential biological context. To establish a naive baseline, a DummyRegressor was used, which simply predicts the mean log2​ fold change for each of the 2,000 genes across all training perturbations, entirely ignoring input features and conditions. In addition to the mean baseline, a naive biological null baseline was used by predicting a constant log2 fold change value of zero across all target genes. To predict the transcriptomic responses from the engineered features, four distinct learning algorithms were employed: a linear Ridge Regression model with L2 regularization to capture direct additive relationships, a Lasso regression model, a non-linear Random Forest Regressor to model potential non-linear interactions, and a K-Nearest Neighbors regressor to leverage local pathway synergies across the target genes. Positive and negative controls were also added to define the lower and upper bounds of feature performance. The negative control replaced biological features with Gaussian noise to test whether the models perform better than pure chance. In contrast, the positive control utilized PCA embeddings derived directly from the target response matrix, deliberately simulating data leakage to determine the theoretical upper limit of predictive accuracy.  Model performance on the held-out test perturbations was evaluated using Mean Squared Error (MSE) and mean Pearson correlation coefficient (r). Uncertainty in these performance estimates was additionally assessed using 95% bootstrap confidence intervals.

As an initial feature engineering strategy, an easy and naive baseline strategy was used, where the unperturbed basal expression level of the target perturbation was combined with a one-hot-encoded condition vector. The resulting four-dimensional input matrix X was standardized using StandardScaler to align the scale of continuous expression levels with binary condition indicators. Across all five perturbation-selection strategies, both Ridge Regression and Random Forest barely outperformed the naive Dummy Regressor baseline. In this low-dimensional feature space, the only perturbation-specific information is its basal expression and this lacked a consistent linear relationship with the multi-gene target response, causing Ridge regression to penalize the perturbation-specific weight towards zero. Therefore the Ridge regression model collapsed into predicting condition-specific mean responses rather than perturbation-specific regulatory effects. With only 40 training knockouts, the Random Forest model could not learn reliable splits on basal expression, causing it to default to the average condition response for unseen test genes. The controls clearly illustrate this limitation: while the models reliably beat random noise and the no-change baseline, a wide gap remains to the theoretical upper limit set by the positive control. In conclusion, reducing a complex perturbation to just four scalar features cannot capture specific regulatory cascades across 2,000 target genes. Predicting unseen genetic knockouts reliably requires richer biological priors, such as gene regulatory networks or molecular embeddings.

To address this limitation and incorporate such biological context, a second, pathway-informed feature engineering strategy was implemented. For this pathway-informed model, the input matrix X contained Hallmark pathway annotations, experimental condition, and unperturbed baseline information, while the target matrix Y contained the 2,000-gene log2FC response. To reduce target dimensionality, PCA was fitted on the training data only and the response was reduced to 50 principal components. Random Forest regression, Ridge regression and DummyRegressor as baseline model were used to predict the mean log2FC responses. Hyperparameters were optimized using RandomizedSearchCV for the Random Forest and GridSearchCV for Ridge regression. Cross-validation was performed using GroupKFold, with the perturbation target used as the grouping variable. This ensured that the three condition-specific profiles belonging to the same perturbation remained together during cross-validation. The predicted responses in 50 PCs space were transformed back into the original 2,000-gene log2FC space and compared with the observed responses. Random Forest showed a moderate-to-strong ability to reproduce the transcriptional response patterns of unseen perturbations across all three conditions and slightly outperformed Ridge Regression and the DummyRegressor. Overall, prediction performance was higher in the IFNγ condition compared with the co-culture condition, particularly for the most-frequent strategy, which had a Pearson correlation of approximately 0.85. Adding condition-specific unperturbed transcriptional profiles as additional input features did not improve predictive performance of each model, suggesting that this information was largely redundant with the experimental condition encoding. The most-frequent perturbation selection strategy showed better predictive performance across all three models than the random selected and highest-log2FC strategies. The 95% bootstrap of each model prediction was narrower for this selection strategy compared to other selection methods. This may be because frequently represented perturbations contained more cells, resulting in more stable estimates of their mean transcriptional responses and reduced noise in the log2FC profiles. The highest-log2FC strategy selected perturbations with stronger and potentially more extreme responses, which may have been more difficult to predict, and resulted in the higher MAE and RMSE. Wider bootstrap confidence intervals were also observed for the Leiden-based, pathway-based, and random selection strategies, indicating greater uncertainty in the estimated predictive performance. This was particularly apparent under the co-culture conditions, possibly due to greater heterogeneity among the held-out perturbation responses, combined with the relatively small test and train set. 

Despite the inclusion of biological pathways, the final evaluation revealed that the advanced models (Random Forest, Ridge, Lasso, KNN) performed only marginally better than the Train Mean baseline on the zero-shot targets. Rather than a modeling failure, this highlights a fundamental limitation of the feature space: reducing a complex perturbation to scalar baseline features and binary pathway labels is insufficient to capture highly specific regulatory cascades across 2,000 target genes. Predicting entirely unseen genetic knockouts reliably likely requires richer mechanistic priors, such as explicit gene regulatory networks or multimodal molecular embeddings. 
