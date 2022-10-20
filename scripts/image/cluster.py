import numpy as np
from pickle import load, dump
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision.utils import make_grid
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import pandas as pd
from PIL import Image
from multiprocessing import Pool

from kmeans import kmeans

NUM_CLUSTERS = 32 * 56


def get_data(root: Path):
    for encode_path in tqdm(list(root.glob("*.pk"))):
        with open(encode_path, "rb") as f:
            batch = load(f)
            batch["indxs"] = batch["indxs"].numpy()
        if data is None:
            data = batch
        else:
            data["x"] = np.concatenate([data["x"], batch["x"]])
            data["indxs"] = np.concatenate([data["indxs"], batch["indxs"]])
            data["image_paths"] += batch["image_paths"]

    return data


def pca(x, k, center=True):
    if center:
        m = x.mean(0, keepdim=True)
        s = x.std(0, unbiased=False, keepdim=True)
        x -= m
        x /= s
    # why pca related to svd? https://www.cs.cmu.edu/~elaw/papers/pca.pdf chap VI
    U, S, V = torch.linalg.svd(x)
    reduced = torch.mm(x, V[:k].T)

    return reduced


transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])


def read_image_and_transform(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    return img


def make_image_grid(closest):
    x = list(
        tqdm(
            map(read_image_and_transform, [data["image_paths"][i] for i in closest]),
            total=len(closest),
        )
    )
    grid = make_grid(x, nrow=56)

    to_pil_image(grid).save("./rf100-grid=32x56.png")


data = get_data(Path("./encoded/"))
x = torch.from_numpy(data["x"])
x = pca(x, k=32)
means, bins = kmeans(x, num_clusters=NUM_CLUSTERS, num_iters=25)

# save to disk
kmeans_outfile_name = f"../temp/torch-kmeans_num-clusters={NUM_CLUSTERS}-pca=32.pk"
with open(kmeans_outfile_name, "wb") as f:
    dump(means, f)

# from disk
with open(kmeans_outfile_name, "rb") as f:
    means = load(f)
    means = means.numpy()

print(f"Loaded means, shape={means.shape}")

means = np.sort(means, axis=0)
x = np.sort(x, axis=0)

closest, distances = vq(means[0], x)
print(closest)
print(closest.shape)
print("Computed distances")

make_image_grid(closest)
