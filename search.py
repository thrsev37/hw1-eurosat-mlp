from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hw1_mlp.data import load_dataset
from hw1_mlp.train_utils import train_model, write_json


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small hyperparameter grid search.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "search")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden-dims", type=str, default="64,128")
    parser.add_argument("--activations", type=str, default="relu,tanh")
    parser.add_argument("--lrs", type=str, default="0.03,0.05")
    parser.add_argument("--weight-decays", type=str, default="0.0001,0.0005")
    parser.add_argument("--lr-decay", type=float, default=0.92)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=600, help="Use a capped subset for faster search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    print(
        f"Search dataset: train={len(data.train.y)} val={len(data.val.y)} "
        f"test={len(data.test.y)} max_per_class={args.max_per_class}",
        flush=True,
    )

    combos = list(
        itertools.product(
            parse_int_list(args.hidden_dims),
            parse_str_list(args.activations),
            parse_float_list(args.lrs),
            parse_float_list(args.weight_decays),
        )
    )
    rows = []
    best = None
    for run_idx, (hidden_dim, activation, lr, weight_decay) in enumerate(combos, start=1):
        run_dir = args.artifact_dir / f"run_{run_idx:02d}_h{hidden_dim}_{activation}_lr{lr}_wd{weight_decay}"
        print(
            f"\n[{run_idx}/{len(combos)}] hidden={hidden_dim} activation={activation} "
            f"lr={lr} weight_decay={weight_decay}",
            flush=True,
        )
        _, _, summary = train_model(
            data=data,
            hidden_dim=hidden_dim,
            activation=activation,
            lr=lr,
            lr_decay=args.lr_decay,
            weight_decay=weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed + run_idx,
            artifact_dir=run_dir,
            save_best=False,
            verbose=True,
        )
        row = {
            "run": run_idx,
            "hidden_dim": hidden_dim,
            "activation": activation,
            "lr": lr,
            "lr_decay": args.lr_decay,
            "weight_decay": weight_decay,
            "best_val_acc": summary["final_val_acc"],
            "final_train_acc": summary["final_train_acc"],
            "artifact_dir": str(run_dir),
        }
        rows.append(row)
        if best is None or row["best_val_acc"] > best["best_val_acc"]:
            best = row

    csv_path = args.artifact_dir / "search_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.artifact_dir / "search_results.json", {"results": rows, "best": best})
    print(f"\nBest search result: {best}", flush=True)
    print(f"Saved search table to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
