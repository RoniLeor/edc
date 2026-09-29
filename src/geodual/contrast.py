"""Supervised contrastive objective and deterministic paired image augmentations."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .data import FloatArray

TEMPERATURE: float = 0.1
NORM_EPSILON: float = 1e-8
IMAGE_SIDE: int = 28
MAX_SHIFT: int = 2


@dataclass(frozen=True)
class SupCon:
    """Mean log probability over same-class positives, excluding each anchor itself."""

    def __call__(self, prediction: jax.Array, y: jax.Array) -> jax.Array:
        embeddings: jax.Array = prediction / jnp.maximum(
            jnp.sqrt(jnp.maximum(jnp.sum(prediction**2, axis=-1, keepdims=True), NORM_EPSILON**2)),
            NORM_EPSILON,
        )
        diagonal: jax.Array = jnp.eye(prediction.shape[0], dtype=bool)
        logits: jax.Array = embeddings @ embeddings.T / TEMPERATURE
        logits = jnp.where(diagonal, -1e9, logits)
        labels: jax.Array = jnp.argmax(y, axis=-1)
        positives: jax.Array = (labels[:, None] == labels[None, :]) & ~diagonal
        counts: jax.Array = jnp.sum(positives, axis=-1)
        log_probability: jax.Array = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(jnp.where(positives, log_probability, 0.0), axis=-1) / jnp.maximum(
            counts, 1
        )

    def total(self, prediction: jax.Array, y: jax.Array) -> jax.Array:
        """Sum anchors so every embedding receives both anchor and partner gradients."""
        return jnp.sum(self(prediction=prediction, y=y))

    def gradient(self, *, prediction: jax.Array, y: jax.Array) -> jax.Array:
        return jax.grad(self.total)(prediction, y)


@dataclass(frozen=True)
class Views:
    """Two independent integer translations with background padding; no digit flips."""

    background: float

    def __call__(self, *, x: FloatArray, seed: int) -> FloatArray:
        rng: np.random.Generator = np.random.default_rng(seed)
        images: FloatArray = np.tile(x.reshape(-1, IMAGE_SIDE, IMAGE_SIDE), (2, 1, 1))
        padded: FloatArray = np.pad(
            images,
            pad_width=((0, 0), (MAX_SHIFT, MAX_SHIFT), (MAX_SHIFT, MAX_SHIFT)),
            constant_values=self.background,
        )
        shifts: np.ndarray = rng.integers(0, 2 * MAX_SHIFT + 1, size=(len(images), 2))
        rows: np.ndarray = shifts[:, 0, None, None] + np.arange(IMAGE_SIDE)[None, :, None]
        columns: np.ndarray = shifts[:, 1, None, None] + np.arange(IMAGE_SIDE)[None, None, :]
        return padded[np.arange(len(images))[:, None, None], rows, columns].reshape(len(images), -1)
