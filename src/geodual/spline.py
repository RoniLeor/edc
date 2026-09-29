"""Local thin-plate smoothing spline weights with conservative geometric fallbacks."""

from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

DIMENSIONS: int = 4
PROJECTION_SEED: int = 314159
RADIUS_FLOOR: float = 1e-6
KERNEL_FLOOR: float = 1e-12
MINIMUM_RANK_RATIO: float = 1e-6
MAXIMUM_WEIGHT_NORM: float = 4.0


class Estimate(NamedTuple):
    """Interpolation coefficients and a per-query indication of spline acceptance."""

    weights: jax.Array
    accepted: jax.Array


@dataclass(frozen=True)
class Spline:
    """Regularized radial basis spline with affine tail; no learned parameters."""

    smoothing: float = 0.1

    def __post_init__(self) -> None:
        if not isfinite(self.smoothing) or self.smoothing <= 0:
            raise ValueError("Positive finite spline smoothing required")

    def __call__(
        self,
        *,
        query: jax.Array,
        points: jax.Array,
        fallback: jax.Array,
        available: jax.Array,
    ) -> Estimate:
        rng: np.random.Generator = np.random.default_rng(PROJECTION_SEED)
        projection: np.ndarray = np.linalg.qr(rng.normal(size=(query.shape[-1], DIMENSIONS)))[
            0
        ].astype(np.float32)
        offsets: jax.Array = (points - query[:, None, :]) @ jnp.asarray(projection)
        radius: jax.Array = jnp.sqrt(jnp.mean(jnp.sum(offsets**2, axis=-1), axis=-1))
        normalized: jax.Array = offsets / jnp.maximum(radius[:, None, None], RADIUS_FLOOR)
        return self.weights(points=normalized, fallback=fallback, available=available)

    def weights(self, *, points: jax.Array, fallback: jax.Array, available: jax.Array) -> Estimate:
        """Evaluate at the origin by solving the transposed symmetric spline system."""
        count: int = points.shape[1]
        terms: int = points.shape[2] + 1
        polynomial: jax.Array = jnp.concatenate((jnp.ones((*points.shape[:2], 1)), points), axis=-1)
        spectrum: jax.Array = jnp.linalg.eigvalsh(jnp.swapaxes(polynomial, -1, -2) @ polynomial)
        eligible: jax.Array = jnp.all(available, axis=-1) & (
            spectrum[:, 0] > MINIMUM_RANK_RATIO * spectrum[:, -1]
        )
        squared: jax.Array = jnp.sum((points[:, :, None, :] - points[:, None, :, :]) ** 2, axis=-1)
        kernel: jax.Array = 0.5 * squared * jnp.log(jnp.maximum(squared, KERNEL_FLOOR))
        upper: jax.Array = jnp.concatenate(
            (kernel + self.smoothing * jnp.eye(count), polynomial), axis=-1
        )
        lower: jax.Array = jnp.concatenate(
            (jnp.swapaxes(polynomial, -1, -2), jnp.zeros((len(points), terms, terms))),
            axis=-1,
        )
        system: jax.Array = jnp.concatenate((upper, lower), axis=-2)
        query_squared: jax.Array = jnp.sum(points**2, axis=-1)
        query_kernel: jax.Array = (
            0.5 * query_squared * jnp.log(jnp.maximum(query_squared, KERNEL_FLOOR))
        )
        rhs: jax.Array = jnp.concatenate(
            (query_kernel, jnp.ones((len(points), 1)), jnp.zeros((len(points), terms - 1))),
            axis=-1,
        )
        safe_system: jax.Array = jnp.where(eligible[:, None, None], system, jnp.eye(count + terms))
        solution: jax.Array = jnp.linalg.solve(safe_system, rhs[..., None])[..., 0]
        coefficients: jax.Array = solution[:, :count]
        accepted: jax.Array = (
            eligible
            & jnp.all(jnp.isfinite(coefficients), axis=-1)
            & (jnp.sum(jnp.abs(coefficients), axis=-1) <= MAXIMUM_WEIGHT_NORM)
        )
        return Estimate(
            weights=jnp.where(accepted[:, None], coefficients, fallback), accepted=accepted
        )
