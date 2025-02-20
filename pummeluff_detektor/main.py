import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from pathlib import Path
from skimage.io import collection, imread_collection
import kagglehub


def download_dataset() -> Path:
    # Download latest version
    path = kagglehub.dataset_download("alinapalacios/cs63-pokemon-dataset")
    # path = kagglehub.dataset_download("alinapalacios/cs63-pokemon-dataset")

    return Path(path)


def load_dataset(path: Path) -> collection.ImageCollection:
    images: collection.ImageCollection = imread_collection(str(path))
    return images


def main():
    dataset_dir = download_dataset()
    images = load_dataset(dataset_dir)
    print(images)
    images[0]


if __name__ == "__main__":
    main()
