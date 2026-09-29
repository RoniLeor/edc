"""Bounded, age-gated training-example retrieval for multiplier initialization."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Records(NamedTuple):
    """Fixed-shape FIFO bank, kept in the compiled training state."""

    keys: jax.Array
    errors: jax.Array
    labels: jax.Array
    ids: jax.Array
    duals: jax.Array
    stamps: jax.Array
    cursor: jax.Array


@dataclass(frozen=True)
class Memory:
    """Nearest neighbors conditioned on label and output-loss signal."""

    capacity: int = 512
    neighbors: int = 4
    strength: float = 0.5
    max_age: int = 8
    minimum_similarity: float = 0.5
    temperature: float = 0.1
    mismatch: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.neighbors <= self.capacity or not 0 <= self.strength <= 1:
            raise ValueError("Invalid capacity, neighbors, or strength")
        if self.max_age < 1 or not -1 <= self.minimum_similarity < 1 or self.temperature <= 0:
            raise ValueError("Invalid age, similarity, or temperature")

    def empty(self, *, layers: int, width: int, outputs: int) -> Records:
        return Records(
            keys=jnp.zeros((self.capacity, width)),
            errors=jnp.zeros((self.capacity, outputs)),
            labels=jnp.full((self.capacity,), -1),
            ids=jnp.full((self.capacity,), -1),
            duals=jnp.zeros((self.capacity, layers, width)),
            stamps=jnp.full((self.capacity,), -self.max_age - 1),
            cursor=jnp.asarray(0),
        )

    def __call__(
        self,
        *,
        bank: Records,
        keys: jax.Array,
        errors: jax.Array,
        labels: jax.Array,
        ids: jax.Array,
        step: jax.Array,
    ) -> jax.Array:
        key_norm: jax.Array = keys / jnp.maximum(
            jnp.linalg.norm(keys, axis=-1, keepdims=True), 1e-8
        )
        error_norm: jax.Array = errors / jnp.maximum(
            jnp.linalg.norm(errors, axis=-1, keepdims=True), 1e-8
        )
        similarity: jax.Array = key_norm @ bank.keys.T
        objective: jax.Array = error_norm @ bank.errors.T
        age: jax.Array = step - bank.stamps
        label_match: jax.Array = labels[:, None] == bank.labels[None, :]
        compatible: jax.Array = ~label_match if self.mismatch else label_match
        valid: jax.Array = (
            compatible
            & (bank.labels[None, :] >= 0)
            & (ids[:, None] != bank.ids[None, :])
            & (age[None, :] <= self.max_age)
            & (age[None, :] >= 0)
        )
        scores: jax.Array = jnp.where(valid, similarity + objective, -1e9)
        best: jax.Array
        indices: jax.Array
        best, indices = jax.lax.top_k(scores, self.neighbors)
        available: jax.Array = best > -1e8
        weights: jax.Array = jax.nn.softmax(best / self.temperature, axis=-1) * available
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-8)
        confidence: jax.Array = jnp.clip(
            (jnp.take_along_axis(similarity, indices, axis=1) - self.minimum_similarity)
            / (1 - self.minimum_similarity),
            0,
            1,
        ) * jnp.exp(-age[indices] / self.max_age)
        initial: jax.Array = self.strength * jnp.einsum(
            "bk,bklw->lbw", weights * confidence, bank.duals[indices]
        )
        return jax.lax.stop_gradient(initial)

    def insert(
        self,
        *,
        bank: Records,
        keys: jax.Array,
        errors: jax.Array,
        labels: jax.Array,
        ids: jax.Array,
        duals: jax.Array,
        step: jax.Array,
    ) -> Records:
        if keys.shape[0] > self.capacity:
            raise ValueError("Bank capacity must cover a full batch")
        indices: jax.Array = (bank.cursor + jnp.arange(keys.shape[0])) % self.capacity
        return Records(
            keys=bank.keys.at[indices].set(
                keys / jnp.maximum(jnp.linalg.norm(keys, axis=-1, keepdims=True), 1e-8)
            ),
            errors=bank.errors.at[indices].set(
                errors / jnp.maximum(jnp.linalg.norm(errors, axis=-1, keepdims=True), 1e-8)
            ),
            labels=bank.labels.at[indices].set(labels),
            ids=bank.ids.at[indices].set(ids),
            duals=bank.duals.at[indices].set(jnp.transpose(duals, axes=(1, 0, 2))),
            stamps=bank.stamps.at[indices].set(step),
            cursor=(bank.cursor + keys.shape[0]) % self.capacity,
        )
