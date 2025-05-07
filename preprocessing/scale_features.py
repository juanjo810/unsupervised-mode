import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler

def normalize_array_column(column):
    array_data = np.vstack(column.apply(eval).values) 
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(array_data)
    return [list(vec) for vec in normalized]

def normalize_features(df, feature_columns):
    df_normalized = df.copy()
    for col in feature_columns:
        print(f"Normalizing column: {col}")
        df_normalized[col] = normalize_array_column(df[col])
    return df_normalized

def main(args):
    # Read the input file
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    elif args.input.endswith('.xlsx'):
        df = pd.read_excel(args.input)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    df_norm = normalize_features(df, args.columns)
    df_norm.to_excel(args.output, index=False)
    print(f"Archivo normalizado guardado en: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize array columns in a DataFrame.")
    parser.add_argument("--input", required=True, help="Input file (CSV or Excel)")
    parser.add_argument("--columns", nargs="+", required=True, help="Columns to normalize")
    parser.add_argument("--output", default="normalized_output.xlsx", help="Output file for normalized data")

    args = parser.parse_args()
    main(args)
