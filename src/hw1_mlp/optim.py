from __future__ import annotations


class SGD:
    def __init__(self, lr: float):
        self.lr = float(lr)

    def step(self, params_and_grads) -> None:
        for param, grad in params_and_grads:
            param -= self.lr * grad


class ExponentialDecay:
    def __init__(self, initial_lr: float, decay: float):
        self.initial_lr = float(initial_lr)
        self.decay = float(decay)

    def lr_at_epoch(self, epoch_index: int) -> float:
        return self.initial_lr * (self.decay ** epoch_index)
