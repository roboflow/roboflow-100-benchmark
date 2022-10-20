"""
This script takes a file text (lost in time) and outputs a correctly formated csv file with category -> dataset
"""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import re

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=Path)
parser.add_argument("-o", "--output", type=Path)

args = parser.parse_args()

records = []

with open(args.input, "r") as f:
    for line in f.readlines():
        splitted = line.strip().split(" ")
        is_category = len(splitted) <= 2
        if is_category:
            current_category = " ".join(splitted).lower()
        else:
            dataset = f"{splitted[0].lower()} {splitted[1].lower()}"
            dataset = re.findall(r"\D+", dataset)[0].strip()
            dataset = dataset.replace(" ", "-")
            print(f"[{current_category}] - {dataset}")
            records.append({"category": current_category, "dataset": dataset})

df = pd.DataFrame.from_records(records)
df.to_csv(args.output, index=True)
