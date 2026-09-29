"""Validation-selected linear ridge probes on frozen encoder representations."""

from dataclasses import dataclass

import numpy as np

from .data import Examples

REGULARIZATIONS: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1.0)
SCALE_FLOOR: float = 1e-6


@dataclass(frozen=True)
class Probe:
    """Train-only feature normalization and coefficients, with an unpenalized intercept."""

    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    regularization: float
    validation_accuracy: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(((x - self.center) / self.scale) @ self.weights + self.bias, axis=-1)


def fit(*, train: Examples, validation: Examples) -> Probe:
    """Choose ridge strength using validation labels, never test labels."""
    center: np.ndarray = np.asarray(np.mean(train.x, axis=0, dtype=np.float64))
    scale: np.ndarray = np.maximum(np.std(train.x, axis=0, dtype=np.float64), SCALE_FLOOR)
    x: np.ndarray = (train.x - center) / scale
    bias: np.ndarray = np.asarray(np.mean(train.y, axis=0, dtype=np.float64))
    covariance: np.ndarray = x.T @ x / len(x)
    target: np.ndarray = x.T @ (train.y - bias) / len(x)
    candidates: list[Probe] = []
    regularization: float
    for regularization in REGULARIZATIONS:
        weights: np.ndarray = np.linalg.solve(
            covariance + regularization * np.eye(x.shape[1]), target
        )
        predicted: np.ndarray = np.argmax(
            ((validation.x - center) / scale) @ weights + bias, axis=-1
        )
        accuracy: float = float(np.mean(predicted == np.argmax(validation.y, axis=-1)))
        candidates.append(
            Probe(
                center=center,
                scale=scale,
                weights=weights,
                bias=bias,
                regularization=regularization,
                validation_accuracy=accuracy,
            )
        )
    return candidates[int(np.argmax([probe.validation_accuracy for probe in candidates]))]
