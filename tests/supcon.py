"""Small isolated end-to-end contrastive experiment and CLI contract."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from geodual.data import Dataset, Examples
from geodual.model import Weights
from geodual.supcon import Experiment, main
from geodual.training import Config, Trainer


def test_experiment_writes_probes_without_mutating_examples(tmp_path: Path) -> None:
    rng: np.random.Generator = np.random.default_rng(1)
    examples: Examples = Examples(
        x=rng.normal(size=(6, 784)).astype(np.float32),
        y=np.eye(2, dtype=np.float32)[[0, 1, 0, 1, 0, 1]],
        ids=np.arange(6),
    )
    validation: Examples = Examples(
        x=rng.normal(size=(4, 784)).astype(np.float32),
        y=np.eye(2, dtype=np.float32)[[0, 1, 0, 1]],
        ids=np.arange(6, 10),
    )
    test: Examples = Examples(
        x=rng.normal(size=(4, 784)).astype(np.float32), y=validation.y.copy(), ids=-1 - np.arange(4)
    )
    data: Dataset = Dataset(train=examples, validation=validation, test=test)
    config: Config = Config(
        dataset="mnist",
        objective="supcon",
        method="gdi",
        epochs=1,
        width=4,
        depth=3,
        outputs=3,
        batch=4,
        capacity=8,
        steps=2,
    )
    experiment: Experiment = Experiment(config=config)
    original: np.ndarray = examples.x.copy()
    summary: dict[str, object] = experiment(data=data, output=tmp_path / "run")
    assert summary["train_examples"] == 6
    assert (
        json.loads((tmp_path / "run/summary.json").read_text())["history"][0]["weight_updates"] == 3
    )
    assert (tmp_path / "run/probe.npz").exists()
    assert (tmp_path / "run/baseline-probe.npz").exists()
    np.testing.assert_array_equal(actual=examples.x, desired=original)
    with pytest.raises(ValueError, match="empty"):
        experiment(data=data, output=tmp_path / "run")
    with pytest.raises(ValueError, match="SupCon"):
        Experiment(config=replace(config, objective="ce"))(data=data, output=tmp_path / "invalid")
    weights: Weights = Trainer(config=config).initialize().weights
    with pytest.raises(FloatingPointError, match="features"):
        experiment.features(
            weights=weights._replace(first=jnp.full_like(weights.first, jnp.nan)), examples=examples
        )


def test_supcon_cli_uses_paired_batch_and_labels(tmp_path: Path) -> None:
    experiment: MagicMock
    with patch("geodual.supcon.load"), patch("geodual.supcon.Experiment") as experiment:
        main(
            ["--dataset", "fashion", "--method", "alm", "--steps", "64", "--output", str(tmp_path)]
        )
        config: Config = experiment.call_args.kwargs["config"]
        assert config.objective == "supcon"
        assert config.batch == 128
        assert config.outputs == 16
        assert config.steps == 64
        experiment.return_value.assert_called_once()
