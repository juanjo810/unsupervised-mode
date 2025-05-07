# 🎼 Preprocessing for Musical Features

This repository contains a set of Python scripts for preprocessing musical features, including normalization, dimensionality reduction, and class balancing through SMOTE.

All scripts are designed to work with `.csv` or `.xlsx` files and operate on columns containing array-like data (e.g., `[0.1, 0.3, 0.6]` stored as strings).

---

## 📦 Scripts Overview

### `scale_features.py`

Normalize array-valued columns using **MinMaxScaler**.

#### Arguments:
- `--input`: Input file (`.csv` or `.xlsx`) – **required**
- `--columns`: List of columns to normalize – **required**
- `--output`: Output file (default: `normalized_output.xlsx`)

#### Example:
```bash
python scale_features.py \
  --input raw_features.xlsx \
  --columns pitch_class_histogram pc_hist_bs_dur \
  --output normalized_features.xlsx
```

#### Description:

- The script reads the input file, applies **MinMaxScaler** to the specified columns, and saves the normalized values in the output file.
- The scaling operation transforms the values in the selected columns into a range between 0 and 1 based on the minimum and maximum values of each column.

### 🎶 dim_reduction.py

This script applies dimensionality reduction techniques **UMAP** and/or **Locally Linear Embedding (LLE)** to the specified columns in a DataFrame. It helps reduce the dimensionality of feature embeddings for easier analysis and visualization.

#### 📋 Arguments

- `--input` (**required**): Path to the input file (`.csv` or `.xlsx`).

##### UMAP-specific:
- `--umap`: Apply **UMAP** (Universal Manifold Approximation and Projection) dimensionality reduction.
- `--columns`: List of columns to apply UMAP on.
- `--n_components`: Number of UMAP components to retain (default: 2).
- `--output_umap`: Output file to save the UMAP results (default: `umap_reduction.xlsx`).

##### LLE-specific:
- `--lle`: Apply **Locally Linear Embedding (LLE)** dimensionality reduction.
- `--lle_features`: List of feature columns to apply LLE on.
- `--lle_components`: Number of LLE components to retain (default: 2).
- `--lle_neighbors`: Number of neighbors to consider in LLE (default: 5).
- `--output_lle`: Output file to save the LLE results (default: `lle_reduction.xlsx`).

#### Example:
```bash
python dim_reduction.py \
  --input normalized_features.xlsx \
  --umap \
  --columns pitch_class_histogram \
  --n_components 2 \
  --output_umap umap_result.xlsx \
  --lle \
  --lle_features pitch_class_histogram pc_hist_bs_dur \
  --lle_components 3 \
  --lle_neighbors 40 \
  --output_lle lle_result.xlsx
```

#### Description:

- The script supports applying both **UMAP** and **LLE** independently. If both are specified, the script will compute reductions for each technique separately.
- **UMAP** and **LLE** are powerful methods for reducing high-dimensional data into a lower-dimensional space, suitable for visualization or further processing.

### 🎶 smote_augmentation.py

This script applies **SMOTE (Synthetic Minority Over-sampling Technique)** to each specified feature column independently, based on the given label column. SMOTE is used for balancing imbalanced datasets by generating synthetic samples for the minority class.

#### 📋 Arguments

- `--input` (**required**): Path to the input file (`.csv` or `.xlsx`).
- `--features` (**required**): List of feature columns to apply SMOTE on. These columns must contain array-like data (e.g., `[0.1, 0.3, 0.6]` stored as strings).
- `--label` (**required**): Label column for SMOTE balancing. The column must contain numeric values and indicate the class labels.
- `--output_dir` (**optional**): Directory to save individual SMOTE output files for each feature column. Default is `smote_outputs`.
- `--random_state` (**optional**): Random seed for reproducibility of the SMOTE process. Default is `42`.

#### Example:
```bash
  python smote_augmentation.py \
  --input normalized_features.xlsx \
  --features embedding_pcp embedding_mert \
  --label mode \
  --output_dir smote_files \
  --random_state 123

```


#### Description:

- The script applies **SMOTE** independently to each feature column specified in the `--features` argument.
- Synthetic samples are generated for each feature column to balance the dataset based on the specified label column.
- Each output is saved in a separate file for each feature column, under the directory specified in `--output_dir`.

## 📝 Notes

 - The feature columns should contain arrays stored as strings, for example: [0.1, 0.3, 0.6].
 - The label column should be numeric, as it will be used for balancing via SMOTE.
 - All scripts support .csv and .xlsx files.
 - For reproducibility of the SMOTE procedure, you can set the --random_state argument.
 - Each script overwrites output files if they already exist, so be sure to back up any important data.




