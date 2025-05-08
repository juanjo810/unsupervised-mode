# Unsupervised Mode Detection

This repository accompanies the article:

> **Evaluating Preprocessing Techniques for Unsupervised Mode Detection in Irish Traditional Music**  
> Juan José Navarro-Cáceres, Diego M. Jiménez-Bravo, María Navarro-Cáceres  
> *Applied Sciences*, 2025, 15(6), 3162  
> [DOI: 10.3390/app15063162](https://www.mdpi.com/2076-3417/15/6/3162)

---

## 📁 Repository Structure

```
├── preprocessing/
│   ├── dim_reduction.py
│   ├── scale_features.py
│   └── smote_augmentation.py
│
├── classification/
│   ├── clustering_models.py
│   └── evaluate_clustering.py
```

### `preprocessing/`

This directory contains scripts for preparing musical features prior to clustering:

- **`dim_reduction.py`**: Apply dimensionality reduction using UMAP or LLE to high-dimensional feature columns.
- **`scale_features.py`**: Normalize array-based features using MinMaxScaler.
- **`smote_augmentation.py`**: Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance class distributions in feature embeddings.

### `classification/`

This directory contains scripts for unsupervised classification and evaluation:

- **`clustering_models.py`**: Applies multiple unsupervised clustering algorithms (KMeans, Agglomerative, DBSCAN, MeanShift, SOM) to extracted feature embeddings.
- **`evaluate_clustering.py`**: Computes performance metrics (NMI, ARI, Purity) for each clustering result in an Excel file.

---

## ⚙️ General Usage

All scripts are standalone and executable via the command line with customizable arguments.

To view usage details for any script, run:

```bash
python <script_name>.py --help
```

Each script supports input in `.csv` or `.xlsx` format and outputs transformed or evaluated data accordingly.

---

## 📖 Reference

Please cite the following article when using this repository:

```bibtex
@Article{navarro2025modedetection,
  AUTHOR = {Navarro-Cáceres, Juan José and Jiménez-Bravo, Diego M. and Navarro-Cáceres, María},
  TITLE = {Evaluating Preprocessing Techniques for Unsupervised Mode Detection in Irish Traditional Music},
  JOURNAL = {Applied Sciences},
  VOLUME = {15},
  YEAR = {2025},
  NUMBER = {6},
  ARTICLE-NUMBER = {3162},
  URL = {https://www.mdpi.com/2076-3417/15/6/3162},
  ISSN = {2076-3417},
  ABSTRACT = {Significant computational research has been dedicated to automatic key and mode detection in Western tonal music, particularly within the major and minor modes. However, limited research has focused on identifying alternative diatonic modes in traditional and folk music contexts. This paper addresses this gap by comparing the effectiveness of various preprocessing techniques in unsupervised machine learning for diatonic mode detection. Using a dataset of Irish folk music that incorporates diatonic modes such as Ionian, Dorian, Mixolydian, and Aeolian, we assess how different preprocessing approaches influence clustering accuracy and mode distinction. By examining multiple feature transformations and reductions, this study highlights the impact of preprocessing choices on clustering performance, aiming to optimize the unsupervised classification of diatonic modes in folk music traditions.},
  DOI = {10.3390/app15063162}
}
```
