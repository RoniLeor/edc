"""Matched-weight diagnostics using the full parameter-gradient cosine to BP."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from .calibration import BANK_BATCHES, QUERY_BATCHES, Calibration
from .data import Examples
from .model import Weights
from .settling import Equilibrium
from .training import Config, State, Trainer


class Measurement(TypedDict):
    """Mean over disjoint query batches at one fixed reference checkpoint."""

    method: str
    steps: int
    global_cosine: float
    layer_cosines: list[float]


def cosine(*, actual: Weights, target: Weights) -> jax.Array:
    """Cosine over all flattened parameters, not an average of layer cosines."""
    dot: jax.Array = jnp.sum(
        jnp.stack([jnp.sum(a * b) for a, b in zip(actual, target, strict=True)])
    )
    actual_norm: jax.Array = jnp.sqrt(sum(jnp.sum(a**2) for a in actual))
    target_norm: jax.Array = jnp.sqrt(sum(jnp.sum(b**2) for b in target))
    return dot / jnp.maximum(actual_norm * target_norm, 1e-30)


@dataclass(frozen=True)
class Alignment:
    """Build one ordinary-GDI bank and hold it and all weights fixed across methods."""

    config: Config
    budgets: tuple[int, ...] = (16, 32, 64)

    def __call__(self, *, weights: Weights, examples: Examples) -> list[Measurement]:
        required: int = (BANK_BATCHES + QUERY_BATCHES) * self.config.batch
        if len(examples.x) < required:
            raise ValueError("Insufficient disjoint alignment examples")
        config: Config = replace(self.config, method="gdi", elastic=0, selector="none")
        state: State = Calibration(config=config).fill(weights=weights, examples=examples)
        rows: list[Measurement] = []
        method: str
        elastic: float
        steps: int
        for method, elastic in (("alm", 0.0), ("gdi", 0.0), ("elastic", 0.05)):
            for steps in self.budgets:
                trainer: Trainer = Trainer(
                    config=replace(
                        config,
                        method="alm" if method == "alm" else "gdi",
                        steps=steps,
                        elastic=elastic,
                    )
                )
                settle: Callable[..., Equilibrium] = jax.jit(trainer.settler)
                initial: Callable[..., jax.Array] = jax.jit(trainer.initial)
                gradient: Callable[..., Weights] = jax.jit(trainer.settler.gradient)
                target_gradient: Callable[..., Weights] = jax.jit(jax.grad(trainer.network.loss))
                diagnostics: Callable[..., tuple[jax.Array, jax.Array, jax.Array]] = jax.jit(
                    trainer.settler.diagnostics
                )
                total: float = 0.0
                layers: np.ndarray = np.zeros(config.depth, dtype=np.float64)
                offset: int
                for offset in range(BANK_BATCHES * config.batch, required, config.batch):
                    x: jax.Array = jnp.asarray(examples.x[offset : offset + config.batch])
                    y: jax.Array = jnp.asarray(examples.y[offset : offset + config.batch])
                    ids: jax.Array = jnp.asarray(examples.ids[offset : offset + config.batch])
                    result: Equilibrium = settle(
                        weights=weights,
                        x=x,
                        y=y,
                        initial=initial(state=state, x=x, y=y, ids=ids),
                    )
                    actual: Weights = gradient(weights=weights, x=x, y=y, result=result)
                    target: Weights = target_gradient(weights, x=x, y=y)
                    total += float(cosine(actual=actual, target=target))
                    layer_values: jax.Array = diagnostics(weights=weights, x=x, y=y, result=result)[
                        0
                    ]
                    layers += np.asarray(layer_values)
                rows.append(
                    Measurement(
                        method=method,
                        steps=steps,
                        global_cosine=total / QUERY_BATCHES,
                        layer_cosines=(layers / QUERY_BATCHES).tolist(),
                    )
                )
        return rows
