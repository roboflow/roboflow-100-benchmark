# read images from rf100
# encode them with clip

from pathlib import Path
from pickle import dump
from typing import Callable, Optional

import clip
import numpy as np
import torch
import torch as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-L/14", device=device, jit=True)


class ImageDataset(Dataset):
    def __init__(self, root: Path, fmt: str = "jpg", transform: Callable = None):
        super().__init__()
        self.images_path = list(root.glob(f"**/*.{fmt}"))
        self.transform = transform or ToTensor()

    def __getitem__(self, idx: int):
        image = Image.open(self.images_path[idx]).convert("RGB")
        return self.transform(image), idx, str(self.images_path[idx])

    def __len__(self):
        return len(self.images_path)


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--batches",
        type=int,
        default=4,
        help="max amout of batches per datasets, default to 4.",
    )
    parser.add_argument("-i", type=Path, default=Path("./rf100/"), help="RF100 dir.")
    parser.add_argument(
        "-o", type=Path, default=Path("./temp/rf100/embeddings"), help="output dir."
    )

    args = parser.parse_args()

    MAX_BATCHES, input_dir, output_dir = args.batches, args.i, args.o

    output_dir: Path = output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    for dataset_path in tqdm(list(input_dir.glob("*/"))):
        ds = ImageDataset(dataset_path / "train/images", transform=preprocess)
        dl = DataLoader(
            ds, batch_size=128, num_workers=8, pin_memory=True, shuffle=True
        )  # we shuffle and we sample 2 batches per dataset
        i = 0
        for (x, indxs, images_paths) in dl:
            with torch.no_grad():
                x = x.to(device)
                x = model.encode_image(x)
                x = x.cpu().numpy()
                encoded_file_name = f"{dataset_path.stem}_{i}.pk"

                with open(Path(output_dir) / encoded_file_name, "wb") as f:
                    dump(
                        {
                            "x": x,
                            "indxs": indxs,
                            "image_paths": images_paths,
                        },
                        f,
                    )

                print(f"Stored to {encoded_file_name}")

            i += 1
            if i >= MAX_BATCHES:
                break
