"""Training-only residual probes and learned finite-candidate energy selection."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .energy import CANDIDATES, MERIT_FLOOR, Ranker, RankingState
from .memory import Memory, Records
from .model import Network, Weights
from .settling import Equilibrium, Settler

SHORT_STEPS: int = 2
TEACHER_STEPS: int = 64
TEACHER_INTERVAL: int = 32
WARMUP_UPDATES: int = 4
LAYER_FEATURES: int = 5
GLOBAL_FEATURES: int = 2
LOG_SCALE: float = 10.0
AVERAGE_CANDIDATE: int = 1


class Probe(NamedTuple):
    """Per-example diagnostic features and nonnegative KKT residual merit."""

    features: jax.Array
    merit: jax.Array


class Choice(NamedTuple):
    """Selected initial multipliers and the updated scorer accounting."""

    initial: jax.Array
    state: RankingState


@dataclass(frozen=True)
class Selector:
    """Probe current weights, select using the old scorer, then optionally teach it."""

    network: Network
    memory: Memory
    mode: str
    rate: float = 0.23588

    def __post_init__(self) -> None:
        if self.mode not in {"residual", "learned"} or self.network.objective != "mse":
            raise ValueError("Residual or learned selection with squared-error loss required")

    @property
    def ranker(self) -> Ranker:
        return Ranker(
            inputs=LAYER_FEATURES * (self.network.depth - 1) + GLOBAL_FEATURES + CANDIDATES
        )

    def probe(
        self,
        initial: jax.Array,
        *,
        weights: Weights,
        x: jax.Array,
        y: jax.Array,
        steps: int,
    ) -> Probe:
        result: Equilibrium = Settler(network=self.network, steps=steps, rate=self.rate)(
            weights=weights, x=x, y=y, initial=initial
        )
        residual: jax.Array = self.network.residuals(weights=weights, x=x, states=result.states)
        stationarity: jax.Array = jax.grad(self.network.energy, argnums=3)(
            weights, x, y, result.states, result.duals
        ) * len(x)
        residual_power: jax.Array = jnp.mean(residual**2, axis=-1)
        stationary_power: jax.Array = jnp.mean(stationarity**2, axis=-1)
        layer_features: jax.Array = jnp.stack(
            (
                jnp.log1p(jnp.mean(result.states**2, axis=-1)),
                jnp.log(jnp.mean(initial**2, axis=-1) + MERIT_FLOOR) / LOG_SCALE,
                jnp.log(residual_power + MERIT_FLOOR) / LOG_SCALE,
                jnp.log(stationary_power + MERIT_FLOOR) / LOG_SCALE,
                jnp.mean(result.states > 0, axis=-1),
            ),
            axis=-1,
        )
        prediction: jax.Array = self.network.output(weights=weights, states=result.states)
        global_features: jax.Array = jnp.stack(
            (
                jnp.log1p(jnp.linalg.norm(self.network.error(prediction=prediction, y=y), axis=-1)),
                jnp.log1p(self.network.supervised(prediction=prediction, y=y)),
            ),
            axis=-1,
        )
        features: jax.Array = jnp.concatenate(
            (jnp.swapaxes(layer_features, 0, 1).reshape(len(x), -1), global_features), axis=-1
        )
        return Probe(
            features=jax.lax.stop_gradient(features),
            merit=jnp.mean(residual_power + stationary_power, axis=0),
        )

    def assess(
        self,
        candidates: jax.Array,
        *,
        weights: Weights,
        x: jax.Array,
        y: jax.Array,
        steps: int,
    ) -> Probe:
        probes: Probe = jax.vmap(partial(self.probe, weights=weights, x=x, y=y, steps=steps))(
            candidates
        )
        features: jax.Array = jnp.swapaxes(probes.features, 0, 1)
        identities: jax.Array = jnp.broadcast_to(
            jnp.eye(CANDIDATES), (len(x), CANDIDATES, CANDIDATES)
        )
        return Probe(
            features=jnp.concatenate((features, identities), axis=-1),
            merit=jnp.swapaxes(probes.merit, 0, 1),
        )

    def teach(
        self,
        state: RankingState,
        *,
        features: jax.Array,
        candidates: jax.Array,
        weights: Weights,
        x: jax.Array,
        y: jax.Array,
    ) -> RankingState:
        target: Probe = self.assess(candidates, weights=weights, x=x, y=y, steps=TEACHER_STEPS)
        return self.ranker.update(state=state, features=features, merit=target.merit)

    def __call__(
        self,
        *,
        weights: Weights,
        bank: Records,
        x: jax.Array,
        y: jax.Array,
        ids: jax.Array,
        step: jax.Array,
        movement: jax.Array,
        state: RankingState,
        train: bool,
    ) -> Choice:
        states: jax.Array = self.network(weights=weights, x=x)
        candidates: jax.Array = self.memory.candidates(
            bank=bank,
            keys=states[-1],
            errors=self.network.error(
                prediction=self.network.output(weights=weights, states=states), y=y
            ),
            labels=jnp.argmax(y, axis=-1),
            ids=ids,
            step=step,
            movement=movement,
        )
        probes: Probe = self.assess(candidates, weights=weights, x=x, y=y, steps=SHORT_STEPS)
        indices: jax.Array
        if self.mode == "learned":
            probabilities: jax.Array = self.ranker.probabilities(
                parameters=state.parameters, features=probes.features
            )
            indices = jnp.where(
                state.updates >= WARMUP_UPDATES,
                jnp.argmax(probabilities, axis=-1),
                AVERAGE_CANDIDATE,
            )
        else:
            indices = jnp.argmin(probes.merit, axis=-1)
        initial: jax.Array = jnp.einsum(
            "bc,clbw->lbw", jax.nn.one_hot(indices, CANDIDATES), candidates
        )
        if train:
            state = state._replace(
                choices=state.choices + jnp.bincount(indices, length=CANDIDATES),
                batches=state.batches + 1,
            )
            if self.mode == "learned":
                state = jax.lax.cond(
                    step % TEACHER_INTERVAL == 0,
                    partial(
                        self.teach,
                        features=probes.features,
                        candidates=candidates,
                        weights=weights,
                        x=x,
                        y=y,
                    ),
                    jax.lax.stop_gradient,
                    state,
                )
        return Choice(initial=jax.lax.stop_gradient(initial), state=state)

    def report(self, state: RankingState) -> dict[str, object]:
        updates: int = int(state.updates)
        batches: int = int(state.batches)
        return {
            "mode": self.mode,
            "candidate_names": ["zero", "average", "strongest_neighbor"],
            "selection_counts": np.asarray(state.choices).tolist(),
            "training_batches": batches,
            "teacher_updates": updates,
            "mean_teacher_cross_entropy": float(state.loss_sum) / max(updates, 1),
            "scorer_parameters": sum(v.size for v in state.parameters)
            if self.mode == "learned"
            else 0,
            "extra_state_gradient_evaluations": (
                batches * CANDIDATES * (SHORT_STEPS + 1)
                + updates * CANDIDATES * (TEACHER_STEPS + 1)
            ),
            "short_steps": SHORT_STEPS,
            "teacher_steps": TEACHER_STEPS,
            "teacher_interval": TEACHER_INTERVAL,
        }
