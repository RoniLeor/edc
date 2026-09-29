"""Synthetic full-update timing; this measures compute, never dataset accuracy."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .training import Config, State, Trainer


@dataclass(frozen=True)
class Benchmark:
    """Time dynamic optimizer updates after compilation and bank population."""

    config: Config
    examples: int = 50000
    batches: int = 20
    rounds: int = 3

    def __post_init__(self) -> None:
        if min(self.examples, self.batches, self.rounds) < 1:
            raise ValueError("Positive benchmark counts required")

    def __call__(self) -> dict[str, object]:
        trainer: Trainer = Trainer(config=self.config)
        state: State = trainer.initialize()
        rng: np.random.Generator = np.random.default_rng(self.config.seed)
        x: jax.Array = jnp.asarray(
            rng.normal(size=(self.config.batch, self.config.inputs)).astype(np.float32)
        )
        cycle: int = ceil(self.config.outputs / self.config.batch)
        targets: tuple[jax.Array, ...] = tuple(
            jax.nn.one_hot(
                (jnp.arange(self.config.batch) + index * self.config.batch) % self.config.outputs,
                num_classes=self.config.outputs,
            )
            for index in range(cycle)
        )
        warmup: int = max(3, ceil(self.config.capacity / self.config.batch))
        count: int = warmup + self.rounds * self.batches
        identities: tuple[jax.Array, ...] = tuple(
            jnp.arange(self.config.batch) + index * self.config.batch for index in range(count)
        )
        jax.block_until_ready((x, targets, identities, state))
        index: int
        started: float = perf_counter()
        for index in range(warmup):
            state = trainer(state=state, x=x, y=targets[index % cycle], ids=identities[index])
        jax.block_until_ready(state)
        startup: float = perf_counter() - started
        samples: list[float] = []
        repetition: int
        for repetition in range(self.rounds):
            started = perf_counter()
            for index in range(
                warmup + repetition * self.batches, warmup + (repetition + 1) * self.batches
            ):
                state = trainer(state=state, x=x, y=targets[index % cycle], ids=identities[index])
            jax.block_until_ready(state)
            samples.append((perf_counter() - started) / self.batches)
        weight: jax.Array
        for weight in state.weights:
            values: NDArray[np.float32] = np.asarray(weight)
            if not np.isfinite(values).all():
                raise FloatingPointError("Synthetic profiling updates became non-finite")
        seconds: float = median(samples)
        epoch: float = ceil(self.examples / self.config.batch) * seconds
        return {
            "kind": "synthetic_compute_only_not_imagenet_training",
            "config": asdict(self.config),
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "examples": self.examples,
            "warmup_batches": warmup,
            "startup_and_warmup_seconds": startup,
            "timed_batches_per_round": self.batches,
            "round_seconds_per_batch": samples,
            "median_seconds_per_batch": seconds,
            "estimated_epoch_compute_seconds": epoch,
            "estimated_run_compute_seconds": epoch * self.config.epochs,
            "bank_bytes": sum(value.nbytes for value in state.bank),
            "excludes": [
                "download",
                "image decoding",
                "augmentation",
                "evaluation",
                "initial compilation",
                "final short-batch compilation",
            ],
        }


def main(argv: Sequence[str] | None = None) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bp", "alm", "gdi"), required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--examples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args: argparse.Namespace = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists")
    config: Config = Config(
        dataset="imagenet1k-profile",
        method=args.method,
        steps=args.steps,
        width=128,
        depth=32,
        inputs=64 * 64 * 3,
        outputs=1000,
        capacity=4000,
        memory_age=64,
        epochs=args.epochs,
    )
    result: dict[str, object] = Benchmark(
        config=config, examples=args.examples, batches=args.batches, rounds=args.rounds
    )()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
