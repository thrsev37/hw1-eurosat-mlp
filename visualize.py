from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)
(ROOT / ".cache").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hw1_mlp.data import load_dataset
from hw1_mlp.model import ThreeLayerMLP
from hw1_mlp.train_utils import confusion_matrix


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def plot_training_curves(history_path: Path, out_dir: Path) -> None:
    history = read_history(history_path)
    epochs = [r["epoch"] for r in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [r["train_loss"] for r in history], marker="o", label="Train loss")
    plt.plot(epochs, [r["val_loss"] for r in history], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [r["val_acc"] for r in history], marker="o", color="#2563eb", label="Validation accuracy")
    plt.plot(epochs, [r["train_acc"] for r in history], marker="o", color="#475569", alpha=0.65, label="Train accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy over training")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curves.png", dpi=180)
    plt.close()


def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    plt.figure(figsize=(8.2, 7.2))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right", fontsize=8)
    plt.yticks(ticks, class_names, fontsize=8)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Test confusion matrix")
    threshold = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "#0f172a"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_first_layer_weights(model: ThreeLayerMLP, image_size: int, out_path: Path, n_units: int = 24) -> None:
    n_units = min(n_units, model.fc1.W.shape[1])
    cols = 6
    rows = int(np.ceil(n_units / cols))
    plt.figure(figsize=(cols * 1.55, rows * 1.55))
    for i in range(n_units):
        weight_img = model.fc1.W[:, i].reshape(image_size, image_size, 3)
        lo, hi = np.percentile(weight_img, [2, 98])
        weight_img = np.clip((weight_img - lo) / (hi - lo + 1e-8), 0, 1)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(weight_img)
        ax.set_title(f"h{i}", fontsize=8)
        ax.axis("off")
    plt.suptitle("First-layer hidden-unit weight maps", y=0.995, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_error_examples(
    data_dir: Path,
    model: ThreeLayerMLP,
    class_names: list[str],
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    out_path: Path,
    seed: int,
    max_per_class: int | None,
) -> list[dict[str, str]]:
    data = load_dataset(data_dir=data_dir, image_size=image_size, seed=seed, max_per_class=max_per_class)
    data.test.X[...] = ((data.test.X * data.std + data.mean) - mean) / std
    preds = model.predict(data.test.X)
    wrong = np.flatnonzero(preds != data.test.y)
    if len(wrong) == 0:
        return []
    selected = wrong[: min(12, len(wrong))]
    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    plt.figure(figsize=(cols * 2.5, rows * 2.65))
    examples = []
    for panel, idx in enumerate(selected, start=1):
        path = data.test.paths[int(idx)]
        true_label = class_names[int(data.test.y[idx])]
        pred_label = class_names[int(preds[idx])]
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"))
        ax = plt.subplot(rows, cols, panel)
        ax.imshow(arr)
        ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=8)
        ax.axis("off")
        examples.append({"path": str(path), "true": true_label, "pred": pred_label})
    plt.suptitle("Misclassified test examples", y=0.995, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures for the EuroSAT MLP report.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "best_model.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = args.artifact_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model, meta = ThreeLayerMLP.load(args.model)
    class_names = meta["class_names"]
    image_size = int(meta["image_size"])

    history_path = args.artifact_dir / "history.csv"
    if history_path.exists():
        plot_training_curves(history_path, fig_dir)

    data = load_dataset(data_dir=args.data_dir, image_size=image_size, seed=args.seed, max_per_class=args.max_per_class)
    data.test.X[...] = ((data.test.X * data.std + data.mean) - meta["mean"]) / meta["std"]
    preds = model.predict(data.test.X)
    cm = confusion_matrix(data.test.y, preds, len(class_names))
    plot_confusion(cm, class_names, fig_dir / "confusion_matrix.png")
    plot_first_layer_weights(model, image_size, fig_dir / "first_layer_weights.png")
    examples = plot_error_examples(
        args.data_dir,
        model,
        class_names,
        image_size,
        meta["mean"],
        meta["std"],
        fig_dir / "error_examples.png",
        args.seed,
        args.max_per_class,
    )
    with (args.artifact_dir / "error_examples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true", "pred"])
        writer.writeheader()
        writer.writerows(examples)
    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
