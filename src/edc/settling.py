"""Synchronous local primal/dual dynamics with reference weight-credit timing."""

from dataclasses import dataclass
from functools import partial
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .model import Network, Weights

MAX_ELASTIC: float = 0.5
MIN_ELASTIC_STEPS: int = 4


class Equilibrium(NamedTuple):
    """Final activities, final duals and the duals used for weight credit."""

    states: jax.Array
    duals: jax.Array
    credit: jax.Array


@dataclass(frozen=True)
class Settler:
    """PC-ALM; alpha=0 recovers ordinary PC from zero initialization."""

    network: Network
    steps: int
    rate: float = 0.23588
    alpha: float = 1.0
    elastic: float = 0.0

    def __post_init__(self) -> None:
        if self.steps < 1 or self.rate <= 0 or self.alpha < 0:
            raise ValueError("Positive steps/rate and nonnegative alpha required")
        if not isfinite(self.elastic) or not 0 <= self.elastic <= MAX_ELASTIC:
            raise ValueError("Elastic strength must be finite and between zero and 0.5")
        if self.elastic > 0 and (self.steps < MIN_ELASTIC_STEPS or self.alpha == 0):
            raise ValueError("Elastic coupling requires at least four steps and dual ascent")

    def step(
        self,
        index: int,
        carry: Equilibrium,
        *,
        weights: Weights,
        x: jax.Array,
        y: jax.Array,
    ) -> Equilibrium:
        gradient: jax.Array = jax.grad(self.network.energy, argnums=3)(
            weights,
            x,
            y,
            carry.states,
            carry.duals,
        )
        states: jax.Array = carry.states - self.rate * x.shape[0] * gradient
        residual: jax.Array = self.network.residuals(weights=weights, x=x, states=states)
        duals: jax.Array = carry.duals + self.alpha * residual
        if self.elastic > 0:
            fade_steps: int = self.steps // 2 - 1
            strength: jax.Array = self.elastic * jnp.maximum(1 - index / fade_steps, 0)
            # Both partners read the old duals; the center layer couples to itself.
            duals = duals - strength * (carry.duals - carry.duals[::-1])
        return Equilibrium(states=states, duals=duals, credit=carry.duals)

    def __call__(
        self,
        *,
        weights: Weights,
        x: jax.Array,
        y: jax.Array,
        initial: jax.Array,
    ) -> Equilibrium:
        carry: Equilibrium = Equilibrium(
            states=self.network(weights=weights, x=x),
            duals=initial,
            credit=initial,
        )
        return jax.lax.fori_loop(
            0,
            self.steps,
            partial(self.step, weights=weights, x=x, y=y),
            carry,
        )

    def gradient(
        self,
        *,
        weights: Weights,
        x: jax.Array,
        y: jax.Array,
        result: Equilibrium,
    ) -> Weights:
        return jax.grad(self.network.energy)(
            weights,
            x,
            y,
            jax.lax.stop_gradient(result.states),
            jax.lax.stop_gradient(result.credit),
        )

    def diagnostics(
        self,
        *,
        weights: Weights,
        x: jax.Array,
        y: jax.Array,
        result: Equilibrium,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        actual: Weights = self.gradient(weights=weights, x=x, y=y, result=result)
        target: Weights = jax.grad(self.network.loss)(weights, x, y)
        cosines: list[jax.Array] = []
        left: jax.Array
        right: jax.Array
        for left, right in zip(actual, target, strict=True):
            axes: tuple[int, ...] = (-2, -1)
            cosine: jax.Array = jnp.sum(left * right, axis=axes) / jnp.maximum(
                jnp.sqrt(jnp.sum(left * left, axis=axes) * jnp.sum(right * right, axis=axes)),
                1e-30,
            )
            cosines.append(jnp.atleast_1d(cosine))
        residual: jax.Array = self.network.residuals(weights=weights, x=x, states=result.states)
        stationarity: jax.Array = (
            jax.grad(self.network.energy, argnums=3)(weights, x, y, result.states, result.duals)
            * x.shape[0]
        )
        return (
            jnp.concatenate(cosines),
            jnp.sqrt(jnp.mean(residual**2)),
            jnp.sqrt(jnp.mean(stationarity**2)),
        )
