import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from minisom import MiniSom


def load_dataframe(path):
    """Load a DataFrame from a CSV or Excel file."""
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx'):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx.")


def apply_clustering(X, y, feature_name, writer, som_shape, include_labels):
    """Apply multiple clustering algorithms to the given feature matrix."""
    algorithms = {
        'KMeans': KMeans(n_clusters=som_shape[1], random_state=42),
        'Agglomerative': AgglomerativeClustering(n_clusters=som_shape[1]),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'MeanShift': MeanShift()
    }

    df_result = pd.DataFrame(X, columns=[f'{feature_name}_{i}' for i in range(X.shape[1])])
    if include_labels:
        df_result['mode'] = y

    for algo_name, algo in algorithms.items():
        cluster_labels = algo.fit_predict(X)
        df_result['cluster_labels'] = cluster_labels
        columns = ['cluster_labels', 'mode'] if include_labels else ['cluster_labels']
        df_result[columns].to_excel(writer, sheet_name=f"{feature_name}_{algo_name}", index=False)

    # SOM clustering
    som = MiniSom(som_shape[0], som_shape[1], input_len=X.shape[1], sigma=0.3,
                  learning_rate=0.5, neighborhood_function='gaussian', random_seed=10)
    som.train_batch(X, 500, verbose=False)
    winner_coordinates = np.array([som.winner(x) for x in X]).T
    cluster_labels = np.ravel_multi_index(winner_coordinates, som_shape)
    df_result['cluster_labels'] = cluster_labels
    columns = ['cluster_labels', 'mode'] if include_labels else ['cluster_labels']
    df_result[columns].to_excel(writer, sheet_name=f"{feature_name}_SOM", index=False)


def main(args):
    df = load_dataframe(args.input)

    if args.label:
        labels = df[args.label]
        num_clusters = len(np.unique(labels))
        som_shape = (1, num_clusters)
        include_labels = True
    else:
        if args.n_clusters is None:
            raise ValueError("When --label is not provided, you must specify --n_clusters.")
        num_clusters = args.n_clusters
        som_shape = (1, num_clusters)
        include_labels = False
        labels = None

    with pd.ExcelWriter(args.output) as writer:
        for feature_col in args.features:
            try:
                X = np.vstack(df[feature_col].apply(eval).values)
            except Exception as e:
                raise ValueError(f"Error parsing column '{feature_col}': {e}")

            y = labels if include_labels else None
            apply_clustering(X, y, feature_col, writer, som_shape, include_labels)

    print(f"Clustering results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply unsupervised clustering algorithms to given features.")
    parser.add_argument("--input", required=True, help="Input file (.csv or .xlsx) containing array-like feature columns.")
    parser.add_argument("--features", nargs="+", required=True, help="Feature columns (array-like) to use for clustering.")
    parser.add_argument("--output", default="clustering_results.xlsx", help="Output Excel file with clustering results.")
    parser.add_argument("--label", help="Optional label column (used for number of clusters and evaluation).")
    parser.add_argument("--n_clusters", type=int, help="Number of clusters when no label is provided.")
    args = parser.parse_args()

    main(args)
