"""Reproducible command-line experiment entry point."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .data import Dataset, load
from .imagenet import ImageNet
from .training import Config, Trainer


def main(argv: Sequence[str] | None = None) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "fashion", "imagenet"), default="fashion")
    parser.add_argument("--method", choices=("bp", "pc", "alm", "gdi", "mismatch"), default="gdi")
    parser.add_argument("--objective", choices=("mse", "ce"), default="mse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--memory-age", type=int)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rate", type=float, default=0.23588)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--drift-scale", type=float, default=0.0)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--selector", choices=("none", "residual", "learned"), default="none")
    parser.add_argument("--elastic", type=float, default=0.0)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args: argparse.Namespace = parser.parse_args(argv)
    output: Path = args.output
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty; preserve previous experiments")
    imagenet: bool = args.dataset == "imagenet"
    args.width = args.width if args.width is not None else (128 if imagenet else 32)
    args.capacity = args.capacity if args.capacity is not None else (4000 if imagenet else 512)
    args.memory_age = args.memory_age if args.memory_age is not None else (64 if imagenet else 8)
    config: Config = Config(
        **{key: value for key, value in vars(args).items() if key not in {"data", "output"}},
        inputs=64 * 64 * 3 if imagenet else 784,
        outputs=1000 if imagenet else 10,
    )
    data: Dataset = (
        ImageNet(root=args.data)() if imagenet else load(name=config.dataset, root=args.data)
    )
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = Trainer(config=config).run(data=data, output=output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
