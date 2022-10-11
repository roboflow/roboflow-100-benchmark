"""
This script create the result table in the paper.

1) get all links from `dataset_links.txt`
2) get all the stats from RF100 (name, number of images in train/val/test)
3) merge the stats with the result dataframe (created by `merge_final_eval.py`)
4) print the latex code for the table
"""
from pathlib import Path
from textwrap import indent
import pandas as pd

RF100_BASE_URL = "https://app.roboflow.com/roboflow-100"


def get_rf100_version_and_links(root: Path) -> pd.DataFrame:
    filepath = root / "datasets_links_640.txt"
    records = []
    with filepath.open("r") as f:
        for line in f.readlines():
            link = Path(line.strip())
            name, version = link.parent.stem, link.stem
            records.append(
                {
                    "name": name,
                    "version": version,
                    "link": f"{RF100_BASE_URL}/{name}/{version}",
                }
            )
    return pd.DataFrame.from_records(records).set_index("name")


def get_rf100_split_stats(root: Path) -> pd.DataFrame:
    records = []

    get_num_images = lambda x: len(list((folder / x / "labels").glob("*.txt")))
    for folder in root.iterdir():
        records.append(
            {
                "name": folder.stem,
                "train": get_num_images("train"),
                "val": get_num_images("valid"),
                "test": get_num_images("test"),
            }
        )

    return pd.DataFrame.from_records(records).set_index("name")


def get_rf100_results(rf100_version_and_links: pd.DataFrame, rf100_split_stats: pd.DataFrame, rf100_train_results: pd.DataFrame) -> pd.DataFrame:

    df = rf100_version_and_links.join(rf100_split_stats).join(rf100_train_results)

    return df

rf100_version_and_links = get_rf100_version_and_links(Path("."))
rf100_split_stats = get_rf100_split_stats(Path("./rf100"))
rf100_train_results = pd.read_csv('./results.csv', index_col=0)
results = get_rf100_results(rf100_version_and_links, rf100_split_stats, rf100_train_results)
print(results.to_latex())