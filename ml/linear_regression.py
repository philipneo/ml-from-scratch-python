from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearRegressionGD:
    learning_rate: float = 0.01
    epochs: int = 1_000
    weights: np.ndarray | None = field(default=None, init=False)
    bias: float = field(default=0.0, init=False)
    loss_history: list[float] = field(default_factory=list, init=False)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        samples, features = x.shape
        self.weights = np.zeros(features)
        self.bias = 0.0
        self.loss_history.clear()

        for _ in range(self.epochs):
            predictions = x @ self.weights + self.bias
            errors = predictions - y

            gradient_w = (2 / samples) * (x.T @ errors)
            gradient_b = float((2 / samples) * np.sum(errors))

            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b
            self.loss_history.append(float(np.mean(errors**2)))

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model must be fit before prediction")
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x @ self.weights + self.bias
