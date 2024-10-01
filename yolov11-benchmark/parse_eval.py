#!/usr/bin/env python3
import argparse
import pandas as pd
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Dataset name")
ap.add_argument("-r", "--results_dir", required=True, help="Directory containing results.csv")
ap.add_argument("-o", "--output", required=True, help="Output file to write")
args = ap.parse_args()

results_csv = os.path.join(args.results_dir, "results.csv")

# Check if results.csv exists
if not os.path.isfile(results_csv):
    print(f"results.csv not found in {args.results_dir}")
    exit(1)

# Read the results.csv file
df = pd.read_csv(results_csv)

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Get the last row (final epoch)
final_epoch = df.iloc[-1]

# Print available columns for debugging
print("Available columns in results.csv:")
print(df.columns.tolist())

# Extract the mAP@0.5 value
try:
    map50 = final_epoch['metrics/mAP50(B)']
except KeyError:
    print("Column 'metrics/mAP50(B)' not found in results.csv.")
    print("Available columns are:", df.columns.tolist())
    exit(1)

# Format the mAP value to 3 decimal places
map50_formatted = f"{map50:.3f}"

# Append the dataset and mAP value to the output file
with open(args.output, "a") as f:
    f.write(f"{args.dataset} {map50_formatted}\n")

print(f"Dataset: {args.dataset}, mAP@0.5: {map50_formatted}")
