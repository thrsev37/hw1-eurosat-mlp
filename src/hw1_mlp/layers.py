from __future__ import annotations

import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, scale: float):
        self.W = (rng.standard_normal((in_features, out_features)) * scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.x_cache: np.ndarray | None = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_cache = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.x_cache is None:
            raise RuntimeError("Linear.backward called before forward")
        x = self.x_cache
        self.dW[...] = x.T @ grad_out
        self.db[...] = grad_out.sum(axis=0)
        return grad_out @ self.W.T


class ReLU:
    def __init__(self):
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("ReLU.backward called before forward")
        return grad_out * self.mask


class Sigmoid:
    def __init__(self):
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_clip = np.clip(x, -40.0, 40.0)
        self.out = 1.0 / (1.0 + np.exp(-x_clip))
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("Sigmoid.backward called before forward")
        return grad_out * self.out * (1.0 - self.out)


class Tanh:
    def __init__(self):
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("Tanh.backward called before forward")
        return grad_out * (1.0 - self.out * self.out)


class CrossEntropyLoss:
    def __init__(self):
        self.probs: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(logits_shifted)
        self.probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        self.y = y
        n = logits.shape[0]
        correct_log_probs = -np.log(self.probs[np.arange(n), y] + 1e-12)
        return float(correct_log_probs.mean())

    def backward(self) -> np.ndarray:
        if self.probs is None or self.y is None:
            raise RuntimeError("CrossEntropyLoss.backward called before forward")
        grad = self.probs.copy()
        n = grad.shape[0]
        grad[np.arange(n), self.y] -= 1.0
        grad /= n
        return grad.astype(np.float32, copy=False)
