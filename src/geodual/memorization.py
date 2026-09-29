"""Training-only memorization diagnostic for squared error versus cross-entropy."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .data import Examples, FloatArray, IntArray
from .model import Weights
from .training import Config, State, Trainer

LOG_INTERVAL: int = 100
SUBSET_SEED: int = 91


@dataclass(frozen=True)
class Memorization:
    """Change only BP's objective while fixing data, initialization, Adam, and batches."""

    config: Config
    objective: str
    updates: int = 1000

    def __post_init__(self) -> None:
        if self.config.method != "bp" or self.objective not in {"mse", "ce"}:
            raise ValueError("Memorization requires BP with mse or ce")
        if self.updates < 1:
            raise ValueError("Positive update count required")

    def loss(self, weights: Weights, x: jax.Array, y: jax.Array) -> jax.Array:
        trainer: Trainer = Trainer(config=replace(self.config, objective=self.objective))
        return trainer.network.loss(weights=weights, x=x, y=y)

    @partial(jax.jit, static_argnums=0)
    def step(self, state: State, x: jax.Array, y: jax.Array, ids: jax.Array) -> State:
        return Trainer(config=replace(self.config, objective=self.objective))(
            state=state, x=x, y=y, ids=ids
        )

    def __call__(self, *, examples: Examples) -> dict[str, object]:
        if len(examples.x) < self.config.batch:
            raise ValueError("Subset must contain at least one full batch")
        trainer: Trainer = Trainer(config=self.config)
        state: State = trainer.initialize()
        history: list[dict[str, object]] = []
        batches: int = (len(examples.x) + self.config.batch - 1) // self.config.batch
        order: IntArray = np.arange(len(examples.x), dtype=np.int64)
        step: int
        for step in range(self.updates + 1):
            if step % LOG_INTERVAL == 0 or step == self.updates:
                metrics: dict[str, float] = trainer.evaluate(
                    weights=state.weights, examples=examples
                )
                objective: float = float(
                    self.loss(
                        weights=state.weights, x=jnp.asarray(examples.x), y=jnp.asarray(examples.y)
                    )
                )
                if not np.isfinite(objective):
                    raise FloatingPointError("Non-finite memorization objective")
                history.append({"update": step, "accuracy": metrics["accuracy"], "loss": objective})
                print(
                    f"{self.objective} lr={self.config.learning_rate} update={step} "
                    f"train_accuracy={metrics['accuracy']:.4f} loss={objective:.6f}",
                    flush=True,
                )
            if step == self.updates:
                break
            if step % batches == 0:
                order = np.random.default_rng(self.config.seed + step // batches).permutation(
                    len(examples.x)
                )
            offset: int = (step % batches) * self.config.batch
            indices: IntArray = order[offset : offset + self.config.batch]
            state = self.step(
                state=state,
                x=jnp.asarray(examples.x[indices]),
                y=jnp.asarray(examples.y[indices]),
                ids=jnp.asarray(examples.ids[indices]),
            )
        return {
            "kind": "training_only_memorization_not_generalization",
            "config": asdict(self.config),
            "objective": self.objective,
            "updates": self.updates,
            "examples": len(examples.x),
            "represented_classes": len(np.unique(np.argmax(examples.y, axis=1))),
            "training_ids": examples.ids.tolist(),
            "history": history,
        }


def main(argv: Sequence[str] | None = None) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/imagenet/train.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--updates", type=int, default=1000)
    args: argparse.Namespace = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; preserve previous diagnostics")
    with np.load(args.data, allow_pickle=False) as cache:
        if not 64 <= args.examples <= len(cache["ids"]):
            parser.error("Require 64 <= examples <= cached training count")
        selected: IntArray = np.random.default_rng(SUBSET_SEED).choice(
            len(cache["ids"]), size=args.examples, replace=False
        )
        x: FloatArray = cache["pixels"][selected].astype(np.float32) / 127.5 - 1
        examples: Examples = Examples(
            x=x,
            y=np.eye(1000, dtype=np.float32)[cache["labels"][selected]],
            ids=cache["ids"][selected].astype(np.int64),
        )
    args.output.mkdir(parents=True)
    objective: str
    rate: float
    for objective in ("mse", "ce"):
        for rate in (0.001, 0.0001):
            config: Config = Config(
                dataset="imagenet-memorization",
                method="bp",
                width=128,
                depth=32,
                inputs=64 * 64 * 3,
                outputs=1000,
                learning_rate=rate,
                capacity=64,
            )
            result: dict[str, object] = Memorization(
                config=config, objective=objective, updates=args.updates
            )(examples=examples)
            (args.output / f"{objective}-{rate}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
