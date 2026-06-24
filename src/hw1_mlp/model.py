from __future__ import annotations

from pathlib import Path

import numpy as np

from .layers import CrossEntropyLoss, Linear, ReLU, Sigmoid, Tanh


ACTIVATIONS = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
}


class ThreeLayerMLP:
    """Input-hidden-output MLP with manual forward/backward propagation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "relu",
        seed: int = 42,
    ):
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from {sorted(ACTIVATIONS)}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.activation_name = activation
        rng = np.random.default_rng(seed)
        first_scale = np.sqrt(2.0 / input_dim) if activation == "relu" else np.sqrt(1.0 / input_dim)
        second_scale = np.sqrt(2.0 / hidden_dim) if activation == "relu" else np.sqrt(1.0 / hidden_dim)

        self.fc1 = Linear(input_dim, hidden_dim, rng, first_scale)
        self.act = ACTIVATIONS[activation]()
        self.fc2 = Linear(hidden_dim, output_dim, rng, second_scale)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, X: np.ndarray) -> np.ndarray:
        z1 = self.fc1.forward(X)
        h1 = self.act.forward(z1)
        return self.fc2.forward(h1)

    def loss_and_backward(self, X: np.ndarray, y: np.ndarray, weight_decay: float = 0.0) -> float:
        logits = self.forward(X)
        data_loss = self.loss_fn.forward(logits, y)
        reg_loss = 0.5 * weight_decay * (np.sum(self.fc1.W * self.fc1.W) + np.sum(self.fc2.W * self.fc2.W))
        grad = self.loss_fn.backward()
        grad = self.fc2.backward(grad)
        grad = self.act.backward(grad)
        self.fc1.backward(grad)
        if weight_decay:
            self.fc1.dW += weight_decay * self.fc1.W
            self.fc2.dW += weight_decay * self.fc2.W
        return float(data_loss + reg_loss)

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        preds = []
        for start in range(0, len(X), batch_size):
            logits = self.forward(X[start : start + batch_size])
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds).astype(np.int64)

    def parameters_and_grads(self):
        return [
            (self.fc1.W, self.fc1.dW),
            (self.fc1.b, self.fc1.db),
            (self.fc2.W, self.fc2.dW),
            (self.fc2.b, self.fc2.db),
        ]

    def save(
        self,
        path: str | Path,
        class_names: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        image_size: int,
        extra: dict[str, str | int | float] | None = None,
    ) -> None:
        extra = extra or {}
        np.savez_compressed(
            path,
            W1=self.fc1.W,
            b1=self.fc1.b,
            W2=self.fc2.W,
            b2=self.fc2.b,
            input_dim=np.array(self.input_dim),
            hidden_dim=np.array(self.hidden_dim),
            output_dim=np.array(self.output_dim),
            activation=np.array(self.activation_name),
            class_names=np.asarray(class_names),
            mean=mean,
            std=std,
            image_size=np.array(image_size),
            extra_keys=np.asarray(list(extra.keys())),
            extra_values=np.asarray([str(v) for v in extra.values()]),
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["ThreeLayerMLP", dict[str, object]]:
        ckpt = np.load(path)
        model = cls(
            input_dim=int(ckpt["input_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            output_dim=int(ckpt["output_dim"]),
            activation=str(ckpt["activation"]),
        )
        model.fc1.W[...] = ckpt["W1"]
        model.fc1.b[...] = ckpt["b1"]
        model.fc2.W[...] = ckpt["W2"]
        model.fc2.b[...] = ckpt["b2"]
        extra = dict(zip(ckpt["extra_keys"].tolist(), ckpt["extra_values"].tolist()))
        meta = {
            "class_names": ckpt["class_names"].tolist(),
            "mean": ckpt["mean"],
            "std": ckpt["std"],
            "image_size": int(ckpt["image_size"]),
            "extra": extra,
        }
        return model, meta
