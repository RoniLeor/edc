"""Matched Adam training; test labels never enter settling or retrieval."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import Dataset, Examples, IntArray
from .energy import RankingState
from .memory import Memory, Records
from .model import Network, Weights
from .selection import Choice, Selector
from .settling import MAX_ELASTIC, MIN_ELASTIC_STEPS, Equilibrium, Settler

MOVEMENT_EPSILON: float = 1e-12


@dataclass(frozen=True)
class Config:
    """Predeclared experimental choices; tuning must use validation only."""

    dataset: str = "fashion"
    method: str = "gdi"
    seed: int = 0
    width: int = 32
    depth: int = 32
    steps: int = 64
    epochs: int = 3
    batch: int = 64
    rate: float = 0.23588
    learning_rate: float = 0.001
    capacity: int = 512
    strength: float = 0.5
    inputs: int = 784
    outputs: int = 10
    memory_age: int = 8
    objective: str = "mse"
    drift_scale: float = 0.0
    neighbors: int = 4
    initializer: str = "average"
    spline_smoothing: float = 0.1
    selector: str = "none"
    elastic: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.elastic) or not 0 <= self.elastic <= MAX_ELASTIC:
            raise ValueError("Elastic strength must be finite and between zero and 0.5")
        if self.elastic > 0 and (
            self.method != "gdi" or self.selector != "none" or self.steps < MIN_ELASTIC_STEPS
        ):
            raise ValueError("Elastic coupling requires GDI, no selector, and at least four steps")
        if self.selector not in {"none", "residual", "learned"}:
            raise ValueError("Unknown energy selector")
        if self.selector != "none" and (
            self.method != "gdi" or self.objective != "mse" or self.initializer != "average"
        ):
            raise ValueError("Energy selection requires squared-error GDI with average retrieval")
        if self.initializer not in {"average", "spline"}:
            raise ValueError("Unknown multiplier initializer")
        if not 1 <= self.neighbors <= self.capacity:
            raise ValueError("Neighbors must fit memory capacity")
        if not np.isfinite(self.spline_smoothing) or self.spline_smoothing <= 0:
            raise ValueError("Positive finite spline smoothing required")
        if self.initializer == "spline":
            if self.method != "gdi" or self.neighbors < 6:
                raise ValueError("Spline requires GDI with at least six neighbors")
        if not np.isfinite(self.drift_scale) or self.drift_scale < 0:
            raise ValueError("Drift scale must be finite and nonnegative")
        if self.drift_scale > 0 and self.method != "gdi":
            raise ValueError("Optimizer drift gating is only defined for GDI")
        if self.objective not in {"mse", "ce", "supcon"}:
            raise ValueError("Objective must be mse, ce, or supcon")
        if self.method not in {"bp", "pc", "alm", "gdi", "mismatch"}:
            raise ValueError("Unknown training method")
        if min(self.epochs, self.batch, self.steps) < 1 or self.learning_rate <= 0:
            raise ValueError("Positive training settings required")
        if min(self.inputs, self.outputs, self.memory_age) < 1:
            raise ValueError("Positive input, output, and memory age required")
        if self.capacity < max(self.batch, 4):
            raise ValueError("Capacity must cover batch and four neighbors")


class State(NamedTuple):
    """Complete optimizer and retrieval state."""

    weights: Weights
    optimizer: optax.OptState
    bank: Records
    step: jax.Array
    movement: jax.Array
    selector: RankingState | None = None


@dataclass(frozen=True)
class Trainer:
    """One interchangeable training strategy over the shared architecture."""

    config: Config

    @property
    def network(self) -> Network:
        return Network(
            width=self.config.width,
            depth=self.config.depth,
            inputs=self.config.inputs,
            outputs=self.config.outputs,
            objective=self.config.objective,
        )

    @property
    def memory(self) -> Memory:
        return Memory(
            capacity=self.config.capacity,
            strength=self.config.strength,
            max_age=self.config.memory_age,
            mismatch=self.config.method == "mismatch",
            drift_scale=self.config.drift_scale,
            neighbors=self.config.neighbors,
            initializer=self.config.initializer,
            spline_smoothing=self.config.spline_smoothing,
        )

    @property
    def selection(self) -> Selector:
        return Selector(
            network=self.network,
            memory=self.memory,
            mode=self.config.selector,
            rate=self.config.rate,
        )

    @property
    def settler(self) -> Settler:
        return Settler(
            network=self.network,
            steps=self.config.steps,
            rate=self.config.rate,
            alpha=0.0 if self.config.method == "pc" else 1.0,
            elastic=self.config.elastic,
        )

    def initialize(self) -> State:
        weights: Weights = self.network.initialize(seed=self.config.seed)
        return State(
            weights=weights,
            optimizer=optax.adam(self.config.learning_rate).init(weights),
            bank=self.memory.empty(
                layers=self.config.depth - 1,
                width=self.config.width,
                outputs=self.config.outputs,
            ),
            step=jnp.asarray(0),
            movement=jnp.asarray(0.0),
            selector=(
                self.selection.ranker.initialize(seed=self.config.seed)
                if self.config.selector != "none"
                else None
            ),
        )

    def initial(self, *, state: State, x: jax.Array, y: jax.Array, ids: jax.Array) -> jax.Array:
        if self.config.selector != "none":
            if state.selector is None:
                raise ValueError("Missing selector state")
            return self.selection(
                weights=state.weights,
                bank=state.bank,
                x=x,
                y=y,
                ids=ids,
                step=state.step,
                movement=state.movement,
                state=state.selector,
                train=False,
            ).initial
        states: jax.Array = self.network(weights=state.weights, x=x)
        if self.config.method not in {"gdi", "mismatch"} or self.config.strength == 0:
            return jnp.zeros_like(states)
        errors: jax.Array = self.network.error(
            prediction=self.network.output(weights=state.weights, states=states),
            y=y,
        )
        return self.memory(
            bank=state.bank,
            keys=states[-1],
            errors=errors,
            labels=jnp.argmax(y, axis=-1),
            ids=ids,
            step=state.step,
            movement=state.movement,
        )

    @partial(jax.jit, static_argnums=0)
    def __call__(self, state: State, x: jax.Array, y: jax.Array, ids: jax.Array) -> State:
        gradient: Weights
        bank: Records = state.bank
        selector: RankingState | None = state.selector
        if self.config.method == "bp":
            gradient = jax.grad(self.network.loss)(state.weights, x, y)
        else:
            initial: jax.Array
            if self.config.selector != "none":
                if selector is None:
                    raise ValueError("Missing selector state")
                choice: Choice = self.selection(
                    weights=state.weights,
                    bank=state.bank,
                    x=x,
                    y=y,
                    ids=ids,
                    step=state.step,
                    movement=state.movement,
                    state=selector,
                    train=True,
                )
                initial = choice.initial
                selector = choice.state
            else:
                initial = self.initial(state=state, x=x, y=y, ids=ids)
            result: Equilibrium = self.settler(weights=state.weights, x=x, y=y, initial=initial)
            gradient = self.settler.gradient(weights=state.weights, x=x, y=y, result=result)
            if self.config.method in {"gdi", "mismatch"} and self.config.strength > 0:
                states: jax.Array = self.network(weights=state.weights, x=x)
                errors: jax.Array = self.network.error(
                    prediction=self.network.output(weights=state.weights, states=states),
                    y=y,
                )
                bank = self.memory.insert(
                    bank=bank,
                    keys=states[-1],
                    errors=errors,
                    labels=jnp.argmax(y, axis=-1),
                    ids=ids,
                    duals=result.duals,
                    step=state.step,
                    movement=state.movement,
                )
        updates: optax.Updates
        optimizer: optax.OptState
        updates, optimizer = optax.adam(self.config.learning_rate).update(
            gradient,
            state.optimizer,
            state.weights,
        )
        movement: jax.Array = state.movement
        if self.config.drift_scale > 0 and self.config.strength > 0:
            movement += self.distance(weights=state.weights, updates=updates)
        return State(
            weights=cast(Weights, optax.apply_updates(state.weights, updates)),
            optimizer=optimizer,
            bank=bank,
            step=state.step + 1,
            movement=movement,
            selector=selector,
        )

    def distance(self, *, weights: Weights, updates: optax.Updates) -> jax.Array:
        """Relative size of the applied Adam update; a path-length proxy, not function drift."""
        return optax.tree.norm(updates) / jnp.maximum(optax.tree.norm(weights), MOVEMENT_EPSILON)

    @partial(jax.jit, static_argnums=0)
    def metrics(self, weights: Weights, x: jax.Array, y: jax.Array) -> jax.Array:
        if self.config.objective == "supcon":
            raise ValueError("SupCon requires frozen probe evaluation")
        output: jax.Array = self.network.output(
            weights=weights,
            states=self.network(weights=weights, x=x),
        )
        return jnp.stack(
            (
                jnp.sum(jnp.argmax(output, axis=-1) == jnp.argmax(y, axis=-1)),
                jnp.sum(self.network.supervised(prediction=output, y=y)),
            ),
        )

    def evaluate(self, *, weights: Weights, examples: Examples) -> dict[str, float]:
        total: np.ndarray = np.zeros(2, dtype=np.float64)
        start: int
        for start in range(0, len(examples.x), 1000):
            total += np.asarray(
                self.metrics(
                    weights=weights,
                    x=jnp.asarray(examples.x[start : start + 1000]),
                    y=jnp.asarray(examples.y[start : start + 1000]),
                ),
            )
        total /= len(examples.x)
        if not np.isfinite(total).all():
            raise FloatingPointError("Non-finite evaluation: unstable training run")
        return {"accuracy": float(total[0]), self.config.objective: float(total[1])}

    def run(self, *, data: Dataset, output: Path) -> dict[str, object]:
        state: State = self.initialize()
        best: State = state
        best_accuracy: float = -1
        best_epoch: int = 0
        history: list[dict[str, object]] = []
        training_seconds: float = 0
        first_step_seconds: float = 0
        started: float = perf_counter()
        epoch: int
        for epoch in range(self.config.epochs):
            order: IntArray = np.random.default_rng(self.config.seed + epoch).permutation(
                len(data.train.x),
            )
            batch_start: float = perf_counter()
            offset: int
            for offset in range(0, len(order), self.config.batch):
                indices: IntArray = order[offset : offset + self.config.batch]
                state = self(
                    state=state,
                    x=jnp.asarray(data.train.x[indices]),
                    y=jnp.asarray(data.train.y[indices]),
                    ids=jnp.asarray(data.train.ids[indices]),
                )
                if epoch == 0 and offset == 0:
                    jax.block_until_ready(state)
                    first_step_seconds = perf_counter() - batch_start
            jax.block_until_ready(state)
            training_seconds += perf_counter() - batch_start
            if state.selector is not None and not all(
                np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(state.selector)
            ):
                raise FloatingPointError("Non-finite selector state")
            validation: dict[str, float] = self.evaluate(
                weights=state.weights,
                examples=data.validation,
            )
            row: dict[str, object] = {
                "epoch": epoch + 1,
                "validation": validation,
                "training_seconds": training_seconds,
            }
            history.append(row)
            print(
                f"{self.config.dataset} {self.config.method} T={self.config.steps} "
                f"seed={self.config.seed} epoch={epoch + 1} val={validation['accuracy']:.4f} "
                f"train_s={training_seconds:.1f}",
                flush=True,
            )
            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]
                best = state
                best_epoch = epoch + 1
        train_metrics: dict[str, float] = self.evaluate(weights=best.weights, examples=data.train)
        test_metrics: dict[str, float] = self.evaluate(weights=best.weights, examples=data.test)
        output.mkdir(parents=True, exist_ok=True)
        np.savez(
            output / "weights.npz",
            first=np.asarray(best.weights.first),
            hidden=np.asarray(best.weights.hidden),
            readout=np.asarray(best.weights.readout),
        )
        if best.selector is not None and self.config.selector == "learned":
            np.savez(
                output / "selector.npz",
                first=np.asarray(best.selector.parameters.first),
                bias=np.asarray(best.selector.parameters.bias),
                readout=np.asarray(best.selector.parameters.readout),
            )
        diagnostics: dict[str, object] = self.diagnose(state=best, examples=data.validation)
        return {
            "config": asdict(self.config),
            "history": history,
            "best_epoch": best_epoch,
            "validation_accuracy": best_accuracy,
            "train": train_metrics,
            "test": test_metrics,
            "gap": train_metrics["accuracy"] - test_metrics["accuracy"],
            "training_seconds_including_compile": training_seconds,
            "first_step_seconds_including_compile": first_step_seconds,
            "total_seconds": perf_counter() - started,
            "bank_bytes": sum(v.nbytes for v in best.bank),
            "optimizer_movement": float(best.movement),
            "selector": self.selection.report(state.selector)
            if state.selector is not None
            else None,
            "selected_checkpoint_teacher_updates": (
                int(best.selector.updates) if best.selector is not None else 0
            ),
            "train_examples": len(data.train.x),
            "validation_examples": len(data.validation.x),
            "test_examples": len(data.test.x),
            "diagnostics": diagnostics,
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
        }

    def diagnose(self, *, state: State, examples: Examples) -> dict[str, object]:
        if self.config.method == "bp":
            return {"layer_cosines": [1.0] * self.config.depth}
        x: jax.Array = jnp.asarray(examples.x[: self.config.batch])
        y: jax.Array = jnp.asarray(examples.y[: self.config.batch])
        ids: jax.Array = jnp.asarray(examples.ids[: self.config.batch])
        lookup: Callable[..., jax.Array] = jax.jit(self.initial)
        initial: jax.Array = lookup(state=state, x=x, y=y, ids=ids)
        jax.block_until_ready(initial)
        started: float = perf_counter()
        initial = lookup(state=state, x=x, y=y, ids=ids)
        jax.block_until_ready(initial)
        lookup_seconds: float = perf_counter() - started
        result: Equilibrium = jax.jit(self.settler)(
            weights=state.weights,
            x=x,
            y=y,
            initial=initial,
        )
        cosine: jax.Array
        residual: jax.Array
        stationarity: jax.Array
        cosine, residual, stationarity = self.settler.diagnostics(
            weights=state.weights,
            x=x,
            y=y,
            result=result,
        )
        return {
            "layer_cosines": np.asarray(cosine).tolist(),
            "residual_rms": float(residual),
            "stationarity_rms": float(stationarity),
            "initial_dual_rms": float(jnp.sqrt(jnp.mean(initial**2))),
            "isolated_lookup_seconds": lookup_seconds,
            "lookup_benchmark": "dynamic_inputs",
        }
