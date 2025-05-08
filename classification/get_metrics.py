import argparse
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def get_purity(df, true_col, pred_col):
    """Compute cluster purity based on majority true label in each predicted cluster."""
    result = df.groupby([true_col, pred_col]).size() / df.groupby(true_col).size()
    result = result.reset_index(name='Percentage')
    best_clusters = result.loc[result.groupby(true_col)['Percentage'].idxmax()]
    return best_clusters['Percentage'].mean()

def main(args):
    df_sheets = pd.read_excel(args.input, sheet_name=None)
    results = []

    for sheet_name, df in df_sheets.items():
        if args.true_labels in df.columns and args.pred_labels in df.columns:
            nmi = normalized_mutual_info_score(df[args.true_labels], df[args.pred_labels])
            ari = adjusted_rand_score(df[args.true_labels], df[args.pred_labels])
            purity = get_purity(df, args.true_labels, args.pred_labels)
            results.append({
                'Sheet': sheet_name,
                'NMI': nmi,
                'ARI': ari,
                'Purity': purity
            })
        else:
            print(f"Skipping sheet '{sheet_name}': missing columns '{args.true_labels}' or '{args.pred_labels}'")

    results_df = pd.DataFrame(results)
    results_df.to_excel(args.output, index=False)
    print(f"Evaluation results saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clustering results using NMI, ARI and Purity.")
    parser.add_argument("--input", required=True, help="Input Excel file with clustering results (multiple sheets)")
    parser.add_argument("--output", default="clustering_metrics.xlsx", help="Output Excel file for metrics")
    parser.add_argument("--true_labels", default="mode", help="Column name for ground truth labels")
    parser.add_argument("--pred_labels", default="cluster_labels", help="Column name for predicted cluster labels")
    args = parser.parse_args()

    main(args)
