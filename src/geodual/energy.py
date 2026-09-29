"""Small conditional energy model over a finite set of multiplier candidates."""

from dataclasses import dataclass
from math import sqrt
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import optax

CANDIDATES: int = 3
HIDDEN_WIDTH: int = 16
SCORER_SEED_OFFSET: int = 104729
SCORER_RATE: float = 0.001
TARGET_SCALE_FLOOR: float = 0.1
MERIT_FLOOR: float = 1e-12


class Parameters(NamedTuple):
    """Two-layer scalar energy model, shared across candidate types."""

    first: jax.Array
    bias: jax.Array
    readout: jax.Array


class RankingState(NamedTuple):
    """Scorer optimizer and full-run accounting, separate from the classifier optimizer."""

    parameters: Parameters
    optimizer: optax.OptState
    updates: jax.Array
    batches: jax.Array
    choices: jax.Array
    loss_sum: jax.Array


@dataclass(frozen=True)
class Ranker:
    """Normalized conditional energy probabilities; no continuous-density sampler."""

    inputs: int

    def initialize(self, *, seed: int) -> RankingState:
        parameters: Parameters = Parameters(
            first=jax.random.normal(
                jax.random.PRNGKey(seed + SCORER_SEED_OFFSET), shape=(self.inputs, HIDDEN_WIDTH)
            )
            / sqrt(self.inputs),
            bias=jnp.zeros(HIDDEN_WIDTH),
            readout=jnp.zeros(HIDDEN_WIDTH),
        )
        return RankingState(
            parameters=parameters,
            optimizer=optax.adam(SCORER_RATE).init(parameters),
            updates=jnp.asarray(0),
            batches=jnp.asarray(0),
            choices=jnp.zeros(CANDIDATES, dtype=jnp.int32),
            loss_sum=jnp.asarray(0.0),
        )

    def __call__(self, *, parameters: Parameters, features: jax.Array) -> jax.Array:
        return jnp.tanh(features @ parameters.first + parameters.bias) @ parameters.readout

    def probabilities(self, *, parameters: Parameters, features: jax.Array) -> jax.Array:
        return jax.nn.softmax(-self(parameters=parameters, features=features), axis=-1)

    def targets(self, merit: jax.Array) -> jax.Array:
        """Soft rankings of long-probe residual merit, not ground-truth multiplier labels."""
        logged: jax.Array = jnp.log(jnp.maximum(merit, MERIT_FLOOR))
        centered: jax.Array = logged - jnp.mean(logged, axis=-1, keepdims=True)
        scale: jax.Array = jnp.maximum(jnp.std(logged, axis=-1, keepdims=True), TARGET_SCALE_FLOOR)
        return jax.lax.stop_gradient(jax.nn.softmax(-centered / scale, axis=-1))

    def loss(self, parameters: Parameters, features: jax.Array, targets: jax.Array) -> jax.Array:
        energy: jax.Array = self(parameters=parameters, features=features)
        return -jnp.mean(jnp.sum(targets * jax.nn.log_softmax(-energy, axis=-1), axis=-1))

    def update(self, *, state: RankingState, features: jax.Array, merit: jax.Array) -> RankingState:
        loss: jax.Array
        gradient: Parameters
        loss, gradient = jax.value_and_grad(self.loss)(
            state.parameters, jax.lax.stop_gradient(features), self.targets(merit)
        )
        updates: optax.Updates
        optimizer: optax.OptState
        updates, optimizer = optax.adam(SCORER_RATE).update(
            gradient, state.optimizer, state.parameters
        )
        return state._replace(
            parameters=cast(Parameters, optax.apply_updates(state.parameters, updates)),
            optimizer=optimizer,
            updates=state.updates + 1,
            loss_sum=state.loss_sum + loss,
        )
