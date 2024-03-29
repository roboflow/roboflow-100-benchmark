import os
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

import pandas as pd
import requests
from tqdm import tqdm

ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]
API_URL = "https://api.roboflow.com/roboflow-100"

df = pd.read_csv("./metadata/categories.csv", index_col=0)
datasets = df.index


def get_labels(dataset):
    res = requests.get(f"{API_URL}/{dataset}", params={"api_key": ROBOFLOW_API_KEY})
    classes = res.json()["project"]["classes"]
    return {
        "name": dataset,
        "classes": classes,
        "category": df.loc[dataset]["category"],
    }


with ThreadPoolExecutor() as executor:
    res = list(tqdm(executor.map(get_labels, datasets)))
    pprint(res)

# let's
