from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hw1_mlp.data import load_dataset
from hw1_mlp.train_utils import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy three-layer MLP on EuroSAT RGB.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to EuroSAT_RGB folder.")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts", help="Output artifact directory.")
    parser.add_argument("--image-size", type=int, default=32, help="Image resize size before flattening.")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.92)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap for quick debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    print(
        f"Loaded EuroSAT: train={len(data.train.y)} val={len(data.val.y)} test={len(data.test.y)} "
        f"classes={len(data.class_names)} input_dim={data.train.X.shape[1]}",
        flush=True,
    )
    train_model(
        data=data,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
    )


if __name__ == "__main__":
    main()
