from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hw1_mlp.data import load_dataset
from hw1_mlp.model import ThreeLayerMLP
from hw1_mlp.train_utils import accuracy, confusion_matrix, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved best model on the test split.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "best_model.npz")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, meta = ThreeLayerMLP.load(args.model)
    data = load_dataset(
        data_dir=args.data_dir,
        image_size=int(meta["image_size"]),
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    # Use the exact normalization stored with the model.
    data.test.X[...] = ((data.test.X * data.std + data.mean) - meta["mean"]) / meta["std"]
    preds = model.predict(data.test.X)
    acc = accuracy(data.test.y, preds)
    cm = confusion_matrix(data.test.y, preds, len(data.class_names))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_accuracy": float(acc),
        "n_test": int(len(data.test.y)),
        "class_names": data.class_names,
        "model": str(args.model),
    }
    write_json(args.out_dir / "test_metrics.json", metrics)

    with (args.out_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *data.class_names])
        for name, row in zip(data.class_names, cm):
            writer.writerow([name, *row.tolist()])

    print(f"Test accuracy: {acc:.4f} ({(preds == data.test.y).sum()}/{len(data.test.y)})")
    print("Confusion matrix rows=true labels, columns=predicted labels:")
    print(cm)


if __name__ == "__main__":
    main()
