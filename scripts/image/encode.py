# read images from rf100
# encode them with clip

from typing import Callable, Optional
import torch
import clip
from PIL import Image
from pathlib import Path
import torch as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from pickle import dump
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device, jit=True)


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
    MAX_BATCHES = 4

    datasets_dir = Path(
        "/home/zuppif/Documents/Work/RoboFlow/roboflow-100-benchmark/rf100/"
    )
    encoded_dir = Path("./encoded")
    encoded_dir.mkdir(exist_ok=True)

    for dataset_path in tqdm(list(datasets_dir.glob("*/"))):
        ds = ImageDataset(dataset_path / "train/images", transform=preprocess)
        dl = DataLoader(
            ds, batch_size=128, num_workers=8, pin_memory=True, shuffle=True
        )  # we shuffle and we sample 2 batches per dataset
        i = 0
        for (x, indxs, images_paths) in dl:
            with torch.no_grad():
                x = x.to("cuda")
                x = model.encode_image(x)
                x = x.cpu().numpy()
                encoded_file_name = f"{dataset_path.stem}_{i}.pk"

                with open(Path(encoded_dir) / encoded_file_name, "wb") as f:
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
            if i >= MAX_BATCHES: break