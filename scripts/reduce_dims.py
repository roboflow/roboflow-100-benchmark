from typing import Dict, Iterator, Tuple, List
from image.pca import pca
from pathlib import Path
from tqdm import tqdm
from pickle import load, dump
import numpy as np
import torch

from sklearn.manifold import TSNE


def get_data(root: Path) -> dict:
    data = None
    for path in tqdm(list(root.glob("*.pk"))):
        with path.open("rb") as f:
            batch = load(f)
            batch["indxs"] = batch["indxs"].numpy()
        if data is None:
            data = batch
        else:
            data["x"] = np.concatenate([data["x"], batch["x"]])
            data["indxs"] = np.concatenate([data["indxs"], batch["indxs"]])
            data["image_paths"] += batch["image_paths"]
    return data


def encode_with_pca(data: dict, k: int) -> dict:
    print(data["x"].shape)
    x = torch.from_numpy(data["x"]).float()
    x = pca(x, k=k)
    data["x"] = x.cpu().numpy()
    return data


def encode_with(data: dict, method: str, k: int) -> dict:
    x = data['x']
    if method == "pca":
        data = encode_with_pca(data, k)
    elif method == "tsne":
        model = TSNE(n_components=k, random_state=0)
        np.set_printoptions(suppress=True)
        fit_model = model.fit_transform(x)
    else:
        raise ValueError(f"{method} not known.")

    return data


def reduce_dims(input_dir: Path, output_dir: Path, method: str, k: int):
    data = get_data(input_dir)
    data = encode_with(data, method, k)
    print(f"data shape = {data['x'].shape}")
    with (output_dir / f"reduced-{method}-k={k}.pk").open("wb") as f:
        dump(data, f)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-i", type=Path, default=Path("./rf100/"), help="RF100 dir.")
    parser.add_argument(
        "-o", type=Path, default=Path("./temp/rf100/embeddings"), help="output dir."
    )
    parser.add_argument(
        "-k", type=int, default=3, help="number of dimensions, defaults to 3."
    )
    parser.add_argument(
        "-method", type=str, default="pca", help="type of method, for now only PCA"
    )

    args = parser.parse_args()

    input_dir, output_dir, k, method = args.i, args.o, args.k, args.method

    output_dir.mkdir(exist_ok=True, parents=True)

    reduce_dims(input_dir, output_dir, method, k)
