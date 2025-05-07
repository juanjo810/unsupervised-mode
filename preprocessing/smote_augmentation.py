import pandas as pd
import numpy as np
import argparse
from imblearn.over_sampling import SMOTE
import os

def apply_smote_to_column(df, feature_col, label_col, random_state=None):
    X = np.vstack(df[feature_col].apply(eval).values)
    y = df[label_col].values

    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)

    df_resampled = pd.DataFrame()
    df_resampled[feature_col] = [list(x) for x in X_res]
    df_resampled[label_col] = y_res
    return df_resampled

def main(args):
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    elif args.input.endswith('.xlsx'):
        df = pd.read_excel(args.input)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    os.makedirs(args.output_dir, exist_ok=True)

    for feature_col in args.features:
        print(f"Applying SMOTE to: {feature_col}")
        df_resampled = apply_smote_to_column(df, feature_col, args.label, random_state=args.random_state)
        out_path = os.path.join(args.output_dir, f"smote_{feature_col}.xlsx")
        df_resampled.to_excel(out_path, index=False)
        print(f"Saved resampled file: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply SMOTE to each embedding column independently and save results separately.")
    parser.add_argument("--input", required=True, help="Input file (.csv or .xlsx)")
    parser.add_argument("--features", nargs="+", required=True, help="Feature columns to apply SMOTE")
    parser.add_argument("--label", required=True, help="Label column for SMOTE balancing. It has to be numeric.")
    parser.add_argument("--output_dir", default="smote_outputs", help="Directory to save individual SMOTE results")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for SMOTE reproducibility")

    args = parser.parse_args()
    main(args)
