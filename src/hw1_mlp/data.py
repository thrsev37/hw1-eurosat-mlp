from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    paths: list[Path]


@dataclass
class DatasetBundle:
    train: SplitData
    val: SplitData
    test: SplitData
    class_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    image_size: int


def discover_classes(data_dir: str | Path) -> list[str]:
    data_path = Path(data_dir)
    classes = sorted([p.name for p in data_path.iterdir() if p.is_dir()])
    if not classes:
        raise FileNotFoundError(f"No class folders found under {data_path}")
    return classes


def build_split_index(
    data_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_per_class: int | None = None,
) -> tuple[dict[str, list[Path]], dict[str, np.ndarray], list[str]]:
    data_path = Path(data_dir)
    class_names = discover_classes(data_path)
    rng = np.random.default_rng(seed)

    split_paths: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    split_labels: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for label, class_name in enumerate(class_names):
        files = sorted((data_path / class_name).glob("*.jpg"))
        if not files:
            raise FileNotFoundError(f"No .jpg files found for class {class_name}")
        order = rng.permutation(len(files))
        if max_per_class is not None:
            order = order[: min(max_per_class, len(order))]
        files = [files[i] for i in order]

        n_total = len(files)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = min(n_train, n_total)
        n_val = min(n_val, n_total - n_train)

        partitions = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }
        for split, paths in partitions.items():
            split_paths[split].extend(paths)
            split_labels[split].extend([label] * len(paths))

    labels = {name: np.asarray(vals, dtype=np.int64) for name, vals in split_labels.items()}
    return split_paths, labels, class_names


def load_images(paths: list[Path], image_size: int) -> np.ndarray:
    n = len(paths)
    X = np.empty((n, image_size * image_size * 3), dtype=np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            arr = np.asarray(
                img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        X[i] = (arr / 255.0).reshape(-1)
    return X


def standardize_split(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32, copy=False)


def load_dataset(
    data_dir: str | Path,
    image_size: int = 32,
    seed: int = 42,
    max_per_class: int | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> DatasetBundle:
    paths, labels, class_names = build_split_index(
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        max_per_class=max_per_class,
    )

    X_train_raw = load_images(paths["train"], image_size)
    mean = X_train_raw.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_train_raw.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0

    train = SplitData(standardize_split(X_train_raw, mean, std), labels["train"], paths["train"])
    val = SplitData(
        standardize_split(load_images(paths["val"], image_size), mean, std),
        labels["val"],
        paths["val"],
    )
    test = SplitData(
        standardize_split(load_images(paths["test"], image_size), mean, std),
        labels["test"],
        paths["test"],
    )

    return DatasetBundle(
        train=train,
        val=val,
        test=test,
        class_names=class_names,
        mean=mean,
        std=std,
        image_size=image_size,
    )


def transform_image(path: str | Path, image_size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = load_images([Path(path)], image_size)
    return standardize_split(X, mean, std)
