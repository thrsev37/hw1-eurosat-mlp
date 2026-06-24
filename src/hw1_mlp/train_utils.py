from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import DatasetBundle, SplitData
from .model import ThreeLayerMLP
from .optim import ExponentialDecay, SGD


def minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(len(X))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield X[idx], y[idx]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def evaluate_loss_accuracy(
    model: ThreeLayerMLP,
    split: SplitData,
    weight_decay: float = 0.0,
    batch_size: int = 1024,
) -> tuple[float, float]:
    losses = []
    weights = []
    preds = []
    for start in range(0, len(split.X), batch_size):
        X_batch = split.X[start : start + batch_size]
        y_batch = split.y[start : start + batch_size]
        logits = model.forward(X_batch)
        loss = model.loss_fn.forward(logits, y_batch)
        if weight_decay:
            loss += 0.5 * weight_decay * (
                np.sum(model.fc1.W * model.fc1.W) + np.sum(model.fc2.W * model.fc2.W)
            )
        losses.append(loss)
        weights.append(len(X_batch))
        preds.append(np.argmax(logits, axis=1))
    y_pred = np.concatenate(preds)
    return float(np.average(losses, weights=weights)), accuracy(split.y, y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def write_history(path: str | Path, history: list[dict[str, float]]) -> None:
    fieldnames = ["epoch", "lr", "train_loss", "val_loss", "train_acc", "val_acc"]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_json(path: str | Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def train_model(
    data: DatasetBundle,
    hidden_dim: int,
    activation: str,
    lr: float,
    lr_decay: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    artifact_dir: str | Path,
    save_best: bool = True,
    verbose: bool = True,
) -> tuple[ThreeLayerMLP, list[dict[str, float]], dict[str, float]]:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    model = ThreeLayerMLP(
        input_dim=data.train.X.shape[1],
        hidden_dim=hidden_dim,
        output_dim=len(data.class_names),
        activation=activation,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    schedule = ExponentialDecay(lr, lr_decay)
    optimizer = SGD(lr)

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = 0
    best_model_path = artifact_path / "best_model.npz"

    for epoch in range(1, epochs + 1):
        optimizer.lr = schedule.lr_at_epoch(epoch - 1)
        running_losses = []
        for X_batch, y_batch in minibatches(data.train.X, data.train.y, batch_size, rng):
            loss = model.loss_and_backward(X_batch, y_batch, weight_decay=weight_decay)
            optimizer.step(model.parameters_and_grads())
            running_losses.append(loss)

        train_loss, train_acc = evaluate_loss_accuracy(model, data.train, weight_decay=weight_decay)
        val_loss, val_acc = evaluate_loss_accuracy(model, data.val, weight_decay=weight_decay)
        row = {
            "epoch": epoch,
            "lr": optimizer.lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)

        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save(
                best_model_path,
                class_names=data.class_names,
                mean=data.mean,
                std=data.std,
                image_size=data.image_size,
                extra={
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "hidden_dim": hidden_dim,
                    "activation": activation,
                    "lr": lr,
                    "lr_decay": lr_decay,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "seed": seed,
                },
            )

        if verbose:
            print(
                f"epoch {epoch:02d}/{epochs} "
                f"lr={optimizer.lr:.5f} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}",
                flush=True,
            )

    write_history(artifact_path / "history.csv", history)
    summary = {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "best_model_path": str(best_model_path),
        "final_train_acc": float(history[-1]["train_acc"]),
        "final_val_acc": float(history[-1]["val_acc"]),
        "hidden_dim": int(hidden_dim),
        "activation": activation,
        "lr": float(lr),
        "lr_decay": float(lr_decay),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "image_size": int(data.image_size),
        "n_train": int(len(data.train.y)),
        "n_val": int(len(data.val.y)),
        "n_test": int(len(data.test.y)),
    }
    write_json(artifact_path / "training_summary.json", summary)
    return model, history, summary
