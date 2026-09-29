"""Matched supervised-contrastive experiments with frozen linear-probe evaluation."""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from .contrast import TEMPERATURE, Views
from .data import SOURCES, Dataset, Examples, FloatArray, IntArray, load
from .model import Weights
from .probe import Probe, fit
from .training import Config, State, Trainer

FEATURE_BATCH: int = 1000
PROJECTION_DIMENSION: int = 16
VIEWS_PER_IMAGE: int = 2


@dataclass(frozen=True)
class Experiment:
    """Final-epoch encoder comparison; probe regularization selected on tuning data."""

    config: Config

    def features(self, *, weights: Weights, examples: Examples) -> Examples:
        trainer: Trainer = Trainer(config=self.config)
        encode: Callable[..., jax.Array] = jax.jit(trainer.network)
        chunks: list[FloatArray] = []
        start: int
        for start in range(0, len(examples.x), FEATURE_BATCH):
            states: jax.Array = encode(
                weights=weights, x=jnp.asarray(examples.x[start : start + FEATURE_BATCH])
            )
            chunks.append(np.asarray(jax.nn.relu(states[-1])))
        values: FloatArray = np.concatenate(chunks)
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite encoder features")
        return Examples(x=values, y=examples.y, ids=examples.ids)

    def evaluate(self, *, weights: Weights, data: Dataset, output: Path) -> dict[str, float]:
        train: Examples = self.features(weights=weights, examples=data.train)
        validation: Examples = self.features(weights=weights, examples=data.validation)
        probe: Probe = fit(train=train, validation=validation)
        test: Examples = self.features(weights=weights, examples=data.test)
        np.savez(
            output,
            center=probe.center,
            scale=probe.scale,
            weights=probe.weights,
            bias=probe.bias,
            regularization=probe.regularization,
        )
        return {
            "train_accuracy": float(np.mean(probe(train.x) == np.argmax(train.y, axis=-1))),
            "validation_accuracy": probe.validation_accuracy,
            "test_accuracy": float(np.mean(probe(test.x) == np.argmax(test.y, axis=-1))),
            "regularization": probe.regularization,
            "feature_std_mean": float(np.mean(np.std(train.x, axis=0))),
        }

    def __call__(self, *, data: Dataset, output: Path) -> dict[str, object]:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output must be empty")
        if self.config.objective != "supcon" or self.config.batch % VIEWS_PER_IMAGE:
            raise ValueError("SupCon objective and even view batch required")
        output.mkdir(parents=True, exist_ok=True)
        trainer: Trainer = Trainer(config=self.config)
        state: State = trainer.initialize()
        initial: Weights = state.weights
        mean: float = SOURCES[self.config.dataset][2]
        std: float = SOURCES[self.config.dataset][3]
        augment: Views = Views(background=-mean / std)
        batch_images: int = self.config.batch // VIEWS_PER_IMAGE
        history: list[dict[str, float | int]] = []
        training_seconds: float = 0.0
        epoch: int
        loss: Callable[..., jax.Array] = jax.jit(trainer.network.loss)
        for epoch in range(self.config.epochs):
            order: IntArray = np.random.default_rng(self.config.seed + epoch).permutation(
                len(data.train.x)
            )
            started: float = perf_counter()
            loss_sum: float = 0.0
            count: int = 0
            offset: int
            for offset in range(0, len(order), batch_images):
                indices: IntArray = order[offset : offset + batch_images]
                view_seed: int = self.config.seed * 10000000 + epoch * len(order) + offset
                x: jax.Array = jnp.asarray(augment(x=data.train.x[indices], seed=view_seed))
                y: jax.Array = jnp.tile(jnp.asarray(data.train.y[indices]), (VIEWS_PER_IMAGE, 1))
                ids: jax.Array = jnp.tile(jnp.asarray(data.train.ids[indices]), VIEWS_PER_IMAGE)
                state = trainer(state=state, x=x, y=y, ids=ids)
                batch_loss: float = float(loss(weights=state.weights, x=x, y=y))
                if not np.isfinite(batch_loss):
                    raise FloatingPointError("Non-finite contrastive loss")
                loss_sum += batch_loss * len(indices)
                count += len(indices)
            jax.block_until_ready(state)
            training_seconds += perf_counter() - started
            row: dict[str, float | int] = {
                "epoch": epoch + 1,
                "contrastive_loss": loss_sum / count,
                "training_seconds": training_seconds,
                "weight_updates": int(state.step),
            }
            history.append(row)
            (output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
            print(
                f"{self.config.dataset} {self.config.method} T={self.config.steps} {row}",
                flush=True,
            )
        baseline: dict[str, float] = self.evaluate(
            weights=initial, data=data, output=output / "baseline-probe.npz"
        )
        result: dict[str, float] = self.evaluate(
            weights=state.weights, data=data, output=output / "probe.npz"
        )
        np.savez(
            output / "weights.npz",
            first=np.asarray(state.weights.first),
            hidden=np.asarray(state.weights.hidden),
            readout=np.asarray(state.weights.readout),
        )
        summary: dict[str, object] = {
            "config": asdict(self.config),
            "temperature": TEMPERATURE,
            "history": history,
            "baseline_probe": baseline,
            "probe": result,
            "training_seconds": training_seconds,
            "train_examples": len(data.train.x),
            "validation_examples": len(data.validation.x),
            "test_examples": len(data.test.x),
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "fashion"), required=True)
    parser.add_argument("--method", choices=("bp", "alm", "gdi"), required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args: argparse.Namespace = parser.parse_args(argv)
    config: Config = Config(
        dataset=args.dataset,
        method=args.method,
        steps=args.steps,
        epochs=args.epochs,
        seed=args.seed,
        objective="supcon",
        outputs=PROJECTION_DIMENSION,
        batch=128,
        capacity=1024,
    )
    Experiment(config=config)(data=load(name=args.dataset, root=args.data), output=args.output)


if __name__ == "__main__":
    main()
