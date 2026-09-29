"""Training-only fixed-weight multiplier and BP-gradient diagnostics."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import sqrt

import jax
import jax.numpy as jnp
import numpy as np

from .data import Examples
from .model import Network, Weights
from .settling import Equilibrium
from .training import Config, State, Trainer

BANK_BATCHES: int = 16
QUERY_BATCHES: int = 4
DENOMINATOR_FLOOR: float = 1e-30


@dataclass(frozen=True)
class Adjoints:
    """Exact feasible-state multipliers for diagnostics only, never GDI training targets."""

    network: Network

    def step(
        self, credit: jax.Array, inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        states: jax.Array
        weight: jax.Array
        states, weight = inputs
        previous: jax.Array = credit + (states > 0) * (credit @ weight) / sqrt(
            self.network.width * self.network.depth
        )
        return previous, previous

    def __call__(self, *, weights: Weights, x: jax.Array, y: jax.Array) -> jax.Array:
        states: jax.Array = self.network(weights=weights, x=x)
        prediction: jax.Array = self.network.output(weights=weights, states=states)
        tail: jax.Array = (
            -(self.network.error(prediction=prediction, y=y) @ weights.readout)
            * (states[-1] > 0)
            / self.network.width
        )
        rest: jax.Array
        _, rest = jax.lax.scan(self.step, tail, (states[:-1], weights.hidden), reverse=True)
        return jnp.concatenate((rest, tail[None]), axis=0)


@dataclass(frozen=True)
class Calibration:
    """Compare initializers on the same causal memory bank and disjoint training queries."""

    config: Config

    def fill(self, *, weights: Weights, examples: Examples) -> State:
        trainer: Trainer = Trainer(config=replace(self.config, method="gdi", neighbors=4))
        state: State = trainer.initialize()._replace(weights=weights)
        settle: Callable[..., Equilibrium] = jax.jit(trainer.settler)
        initial: Callable[..., jax.Array] = jax.jit(trainer.initial)
        offset: int
        for offset in range(0, BANK_BATCHES * self.config.batch, self.config.batch):
            x: jax.Array = jnp.asarray(examples.x[offset : offset + self.config.batch])
            y: jax.Array = jnp.asarray(examples.y[offset : offset + self.config.batch])
            ids: jax.Array = jnp.asarray(examples.ids[offset : offset + self.config.batch])
            result: Equilibrium = settle(
                weights=weights, x=x, y=y, initial=initial(state=state, x=x, y=y, ids=ids)
            )
            states: jax.Array = trainer.network(weights=weights, x=x)
            state = state._replace(
                bank=trainer.memory.insert(
                    bank=state.bank,
                    keys=states[-1],
                    errors=trainer.network.error(
                        prediction=trainer.network.output(weights=weights, states=states), y=y
                    ),
                    labels=jnp.argmax(y, axis=-1),
                    ids=ids,
                    duals=result.duals,
                    step=state.step,
                ),
                step=state.step + 1,
            )
        return state

    def __call__(self, *, weights: Weights, examples: Examples) -> dict[str, object]:
        required: int = (BANK_BATCHES + QUERY_BATCHES) * self.config.batch
        if len(examples.x) < required:
            raise ValueError("Insufficient disjoint calibration examples")
        state: State = self.fill(weights=weights, examples=examples)
        control: Trainer = Trainer(config=replace(self.config, neighbors=16))
        oracle: Callable[..., jax.Array] = jax.jit(Adjoints(network=control.network))
        rows: dict[str, object] = {}
        name: str
        neighbors: int
        for name, neighbors in (
            ("zero", 4),
            ("gdi4", 4),
            ("gdi16", 16),
        ):
            trainer: Trainer = Trainer(
                config=replace(
                    self.config,
                    method="alm" if name == "zero" else "gdi",
                    neighbors=neighbors,
                )
            )
            total: np.ndarray = np.zeros(5, dtype=np.float64)
            offset: int
            for offset in range(BANK_BATCHES * self.config.batch, required, self.config.batch):
                x: jax.Array = jnp.asarray(examples.x[offset : offset + self.config.batch])
                y: jax.Array = jnp.asarray(examples.y[offset : offset + self.config.batch])
                ids: jax.Array = jnp.asarray(examples.ids[offset : offset + self.config.batch])
                target: jax.Array = oracle(weights=weights, x=x, y=y)
                initial: jax.Array = trainer.initial(state=state, x=x, y=y, ids=ids)
                reference: jax.Array = control.initial(state=state, x=x, y=y, ids=ids)
                result: Equilibrium = jax.jit(trainer.settler)(
                    weights=weights, x=x, y=y, initial=initial
                )
                cosine: jax.Array
                residual: jax.Array
                stationarity: jax.Array
                cosine, residual, stationarity = jax.jit(trainer.settler.diagnostics)(
                    weights=weights, x=x, y=y, result=result
                )
                total += np.asarray(
                    [
                        jnp.sum((initial - target) ** 2),
                        jnp.sum(target**2),
                        jnp.mean(cosine),
                        residual,
                        jnp.mean(jnp.any(jnp.abs(initial - reference) > 1e-10, axis=(0, 2))),
                    ]
                )
            rows[name] = {
                "initial_relative_rmse": float(sqrt(total[0] / max(total[1], DENOMINATOR_FLOOR))),
                "mean_layer_gradient_cosine_after_settling": float(total[2] / QUERY_BATCHES),
                "constraint_rms_after_settling": float(total[3] / QUERY_BATCHES),
                "fraction_initializers_different_from_gdi16": float(total[4] / QUERY_BATCHES),
            }
        return rows
