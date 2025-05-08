# 🎼 Clustering for Musical Features

This repository contains a set of Python scripts for clustering musical features, using clasical clustering algorithms: KMeans, Agglomerative, DBScan, Meanshift and Self Organised Maps (SOM); and its evaluation metrics. 

The scripts are prepared to use the obtained features from the preprocessing pipeline of the repository.

---

## 📦 Scripts Overview

### 🎶 `unsupervised_clustering.py`

This script performs unsupervised clustering using multiple algorithms (KMeans, Agglomerative Clustering, DBSCAN, MeanShift, and SOM). It supports both labeled and unlabeled data for supervised and pure unsupervised clustering tasks.

#### Description

Clusters features using common unsupervised learning algorithms. The script accepts a dataset in `.csv` or `.xlsx` format and generates an Excel file with clustering results for each feature set and algorithm.

#### 📋 Arguments

```bash
--input           Path to input file (.csv or .xlsx) [required]
--features        List of column names (array-like) to use as feature sets [required]
--output          Output Excel file to save clustering results [default: clustering_results.xlsx]
--label           Optional column with true labels (used to infer number of clusters and evaluate performance)
--n_clusters      Number of clusters (used only if --label is not provided)
--random_state    Random seed for reproducibility
```
#### Behavior
If --label is provided:
 - The number of clusters is inferred from the number of unique labels.
 - Clustering results include a comparison column with the true labels.
 - SOM grid shape is automatically determined as (1, num_labels).

If --label is not provided:
 - You must specify --n_clusters to define the number of clusters for applicable algorithms.
 - The script performs pure unsupervised clustering.

#### Output
The output Excel file contains one sheet per combination of feature set and clustering algorithm. Each sheet includes:
 - `cluster_labels`: Assigned cluster by the algorithm.
 - `label` (if provided): The ground truth label.

Each sheet is named using the format: <FEATURESET>_<ALGORITHM>

#### Example: 

*Supervised clustering* (infers number of clusters from labels)
```bash
python unsupervised_clustering.py \
  --input features_with_labels.xlsx \
  --features pcp_embedding mert_embedding jukemir_embedding \
  --label mode_encoded \
  --output clustering_results.xlsx
```

*Pure Unsupervised Clustering* (you specify the number of clusters)
```bash
python unsupervised_clustering.py \
  --input features_only.xlsx \
  --features pcp_embedding mert_embedding jukemir_embedding \
  --n_clusters 4 \
  --output clustering_results.xlsx
```


### 🎶 `get_metrics`




