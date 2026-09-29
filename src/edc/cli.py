"""Reproducible command-line experiment entry point."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .data import Dataset, load
from .training import Config, Trainer


def main(argv: Sequence[str] | None = None) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "fashion"), default="fashion")
    parser.add_argument("--method", choices=("bp", "pc", "alm", "gdi", "mismatch"), default="gdi")
    parser.add_argument("--objective", choices=("mse", "ce"), default="mse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--memory-age", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rate", type=float, default=0.23588)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--elastic", type=float, default=0.0)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args: argparse.Namespace = parser.parse_args(argv)
    output: Path = args.output
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty; preserve previous experiments")
    config: Config = Config(
        **{key: value for key, value in vars(args).items() if key not in {"data", "output"}},
    )
    data: Dataset = load(name=config.dataset, root=args.data)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = Trainer(config=config).run(data=data, output=output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
