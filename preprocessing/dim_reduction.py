import pandas as pd
import numpy as np
import argparse
import umap
from sklearn.manifold import LocallyLinearEmbedding

def apply_umap(df, feature_names, n_components, metric='euclidean'):
    df_reduced = pd.DataFrame()
    for name in feature_names:
        array_data = np.vstack(df[name].apply(eval).values)
        reducer = umap.UMAP(n_components=n_components, metric=metric)
        embedding = reducer.fit_transform(array_data)
        df_reduced[f"{name}_umap"] = embedding.tolist()
    return df_reduced

def apply_lle(df, feature_names, n_components, n_neighbors):
    df_reduced = pd.DataFrame()
    for name in feature_names:
        array_data = np.vstack(df[name].apply(eval).values)
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
        X_reduced = lle.fit_transform(array_data)
        df_reduced[f"{name}_lle"] = X_reduced.tolist()
    return df_reduced

def main_pipeline(args):
    # Read the input file
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    elif args.input.endswith('.xlsx'):
        df = pd.read_excel(args.input)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    # UMAP
    if args.umap:
        df_umap = apply_umap(df, args.umap_features, args.n_components)
        df_umap.to_excel(args.output_umap, index=False)
        print(f"Embeddings UMAP guardados en: {args.output_umap}")

    # LLE
    if args.lle:
        df_lle = apply_lle(
            df,
            args.lle_features,
            args.lle_components,
            args.lle_neighbors
        )
        df_lle.to_excel(args.output_lle, index=False)
        print(f"Embeddings LLE guardados en: {args.output_lle}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply UMAP and/or LLE to a Dataframe.")
    parser.add_argument("--input", required=True, help="Input file (CSV or Excel)")

    # UMAP
    parser.add_argument("--umap", action="store_true", help="Apply UMAP")
    parser.add_argument("--columns", nargs="+", help="UMAP columns to apply")
    parser.add_argument("--n_components", type=int, default=2, help="Number of UMAP components")
    parser.add_argument("--output_umap", default="umap_reduction.xlsx", help="Output file for UMAP")

    # LLE
    parser.add_argument("--lle", action="store_true", help="Apply LLE")
    parser.add_argument("--lle_features", nargs="+", help="LLE columns to apply")
    parser.add_argument("--lle_components", type=int, default=2, help="Number of LLE components")
    parser.add_argument("--lle_neighbors", type=int, default=5, help="Number of neighbors for LLE")
    parser.add_argument("--output_lle", default="lle_reduction.xlsx", help="Output file for LLE")

    args = parser.parse_args()
    main_pipeline(args)
