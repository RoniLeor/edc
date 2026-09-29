"""Reference-parameterized residual MLP with batched local constraints."""

from dataclasses import dataclass
from math import sqrt
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .contrast import SupCon


class Weights(NamedTuple):
    """Three parameter groups; hidden weights stack all residual blocks."""

    first: jax.Array
    hidden: jax.Array
    readout: jax.Array


@dataclass(frozen=True)
class Network:
    """ReLU residual MLP following Sakana AI's gamma0=1 parameterization."""

    width: int = 32
    depth: int = 32
    inputs: int = 784
    outputs: int = 10
    objective: str = "mse"

    def __post_init__(self) -> None:
        if self.objective not in {"mse", "ce", "supcon"}:
            raise ValueError("Objective must be mse, ce, or supcon")
        if self.depth < 2 or min(self.width, self.inputs, self.outputs) < 1:
            raise ValueError("Positive dimensions and depth >= 2 required")

    def initialize(self, *, seed: int) -> Weights:
        keys: jax.Array = jax.random.split(jax.random.PRNGKey(seed), self.depth)
        hidden: list[jax.Array] = []
        index: int
        for index in range(1, self.depth - 1):
            hidden.append(jax.random.normal(keys[index], shape=(self.width, self.width)))
        return Weights(
            first=jax.random.normal(keys[0], shape=(self.width, self.inputs)),
            hidden=jnp.stack(hidden) if hidden else jnp.empty((0, self.width, self.width)),
            readout=jax.random.normal(keys[-1], shape=(self.outputs, self.width)),
        )

    def block(self, state: jax.Array, weight: jax.Array) -> tuple[jax.Array, jax.Array]:
        value: jax.Array = state + jax.nn.relu(state) @ weight.T / sqrt(self.width * self.depth)
        return value, value

    def __call__(self, *, weights: Weights, x: jax.Array) -> jax.Array:
        first: jax.Array = x @ weights.first.T / sqrt(self.inputs)
        rest: jax.Array
        _, rest = jax.lax.scan(self.block, first, weights.hidden)
        return jnp.concatenate((first[None], rest), axis=0)

    def output(self, *, weights: Weights, states: jax.Array) -> jax.Array:
        return jax.nn.relu(states[-1]) @ weights.readout.T / self.width

    def supervised(self, prediction: jax.Array, y: jax.Array) -> jax.Array:
        """Per-example supervised objective, distinct from layer-constraint penalties."""
        if self.objective == "supcon":
            return SupCon()(prediction=prediction, y=y)
        if self.objective == "ce":
            return -jnp.sum(y * jax.nn.log_softmax(prediction, axis=-1), axis=-1)
        return 0.5 * jnp.sum(jnp.square(prediction - y), axis=-1)

    def error(self, *, prediction: jax.Array, y: jax.Array) -> jax.Array:
        """Per-example output gradient used as the objective-conditioned retrieval key."""
        if self.objective == "supcon":
            return SupCon().gradient(prediction=prediction, y=y)
        if self.objective == "ce":
            return jax.nn.softmax(prediction, axis=-1) - y
        return prediction - y

    def loss(self, weights: Weights, x: jax.Array, y: jax.Array) -> jax.Array:
        prediction: jax.Array = self.output(weights=weights, states=self(weights=weights, x=x))
        return jnp.mean(self.supervised(prediction=prediction, y=y))

    def residuals(self, *, weights: Weights, x: jax.Array, states: jax.Array) -> jax.Array:
        first: jax.Array = states[0] - x @ weights.first.T / sqrt(self.inputs)
        rest: jax.Array = (
            states[1:]
            - states[:-1]
            - jnp.einsum("lbi,loi->lbo", jax.nn.relu(states[:-1]), weights.hidden)
            / sqrt(self.width * self.depth)
        )
        return jnp.concatenate((first[None], rest), axis=0)

    def energy(
        self,
        weights: Weights,
        x: jax.Array,
        y: jax.Array,
        states: jax.Array,
        duals: jax.Array,
    ) -> jax.Array:
        residual: jax.Array = self.residuals(weights=weights, x=x, states=states)
        prediction: jax.Array = self.output(weights=weights, states=states)
        return (
            jnp.sum(self.supervised(prediction=prediction, y=y))
            + jnp.sum(duals * residual)
            + 0.5 * jnp.sum(residual * residual)
        ) / x.shape[0]
