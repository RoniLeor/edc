"""Fixed-weight retrieval-bank construction for gradient diagnostics."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .data import Examples
from .model import Weights
from .settling import Equilibrium
from .training import Config, State, Trainer

BANK_BATCHES: int = 16
QUERY_BATCHES: int = 4


@dataclass(frozen=True)
class Calibration:
    """Populate the shared causal memory bank without updating reference weights."""

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
