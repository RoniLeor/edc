"""Recompute fixed-checkpoint gradient alignment; run from the repository root."""

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from geodual.alignment import Alignment, Measurement
from geodual.data import Dataset, load
from geodual.model import Weights
from geodual.training import Config

parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path("data"))
parser.add_argument("--output", type=Path, default=Path("results/gradients.json"))
args: argparse.Namespace = parser.parse_args()
manifest: dict = json.loads(Path("benchmarks/manifest.json").read_text())
rows: list[dict] = []
dataset: str
for dataset in ("mnist", "fashion"):
    data: Dataset = load(name=dataset, root=args.data)
    reference: dict
    for reference in manifest["references"]:
        if reference["dataset"] != dataset:
            continue
        filepath: Path = Path(reference["weights"])
        assert hashlib.sha256(filepath.read_bytes()).hexdigest() == reference["sha256"]
        with np.load(filepath) as arrays:
            weights: Weights = Weights(
                first=jnp.asarray(arrays["first"]),
                hidden=jnp.asarray(arrays["hidden"]),
                readout=jnp.asarray(arrays["readout"]),
            )
        measurements: list[Measurement] = Alignment(
            config=Config(dataset=dataset, seed=reference["seed"], steps=16)
        )(weights=weights, examples=data.train)
        rows.append(
            {
                "dataset": dataset,
                "seed": reference["seed"],
                "reference": reference,
                "measurements": measurements,
            }
        )
        print(f"{dataset} seed={reference['seed']} complete", flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        {
            "protocol": (
                "Fixed validation-selected BP10 checkpoints; ordinary GDI16 bank from first "
                "16 training batches; next four disjoint batches queried without modifying "
                "weights or bank; mean batch cosine over all flattened weight gradients versus "
                "exact BP at the same weights; per-layer cosines stored separately."
            ),
            "bank_batches": 16,
            "query_batches": 4,
            "batch_size": 64,
            "rows": rows,
        },
        indent=2,
    )
    + "\n"
)
