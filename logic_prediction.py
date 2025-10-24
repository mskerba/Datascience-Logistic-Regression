import pandas as pd
import numpy as np
import argparse
import os
import sys

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def my_argmax(arr):
    max_idx = 0
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx


def main():
    parser = argparse.ArgumentParser(description="Predict Hogwarts Houses using trained thetas.")
    parser.add_argument("test_csv", help="Path to test dataset CSV file.")
    parser.add_argument("--thetas", default="thetas.csv", help="Path to the thetas CSV file (default: thetas.csv).")
    parser.add_argument("--out", default="houses.csv", help="Output CSV file (default: houses.csv).")
    args = parser.parse_args()

    # === Check file existence ===
    if not os.path.exists(args.thetas):
        print(f"❌ Error: Thetas file '{args.thetas}' not found.")
        sys.exit(1)
    if not os.path.exists(args.test_csv):
        print(f"❌ Error: Test CSV file '{args.test_csv}' not found.")
        sys.exit(1)

    theta_df = pd.read_csv(args.thetas)
    classes = list(theta_df.columns)
    all_thetas = theta_df.to_numpy()

    X_df = pd.read_csv(args.test_csv)

    X_df = X_df.drop(columns=["Hogwarts House"], errors="ignore")
    X_df = X_df.dropna()
    idx = X_df["Index"] if "Index" in X_df.columns else pd.Series(range(len(X_df)))

    drop_cols = [c for c in ["Index", "First Name", "Last Name", "Birthday", "Astronomy"] if c in X_df.columns]
    X_df = X_df.drop(columns=drop_cols)
    

    if "Best Hand" in X_df.columns:
        unique_hands = X_df["Best Hand"].unique()
        mapping = {value: idx for idx, value in enumerate(unique_hands)}
        X_df["Best Hand"] = X_df["Best Hand"].map(mapping)

    # Normalizing
    num_cols = X_df.select_dtypes(include=[np.number]).columns
    mu = X_df[num_cols].mean()
    sigma = X_df[num_cols].std().replace(0, 1.0)
    X_df[num_cols] = (X_df[num_cols] - mu) / sigma

    X = X_df.to_numpy()
    X = np.hstack([np.ones((len(X), 1)), X])

    print(f"Thetas shape: {all_thetas.shape}, Input shape: {X.shape}")

    # === Prediction ===
    Z = np.dot(X, all_thetas)
    H = sigmoid(Z)
    pred_idx = [my_argmax(row) for row in H]
    pred_class = [classes[i] for i in pred_idx]
    
    # === Save output ===
    out = pd.DataFrame({"Index": idx, "Hogwarts House": pred_class})
    out.to_csv(args.out, index=False)
    print(f"💾 Predictions saved to {args.out}")

if __name__ == "__main__":
    main()
