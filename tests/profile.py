"""Dimension generalization and isolated profiler contract checks."""

import json
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import pytest

from geodual.profile import Benchmark, main
from geodual.training import Config, State, Trainer


def test_configurable_dimensions_and_age() -> None:
    trainer: Trainer = Trainer(
        config=Config(
            method="gdi",
            inputs=12,
            outputs=17,
            depth=3,
            width=4,
            capacity=4,
            batch=2,
            memory_age=32,
            steps=2,
        )
    )
    state: State = trainer.initialize()
    assert state.weights.first.shape == (4, 12)
    assert state.weights.readout.shape == (17, 4)
    assert state.bank.errors.shape == (4, 17)
    assert trainer.memory.max_age == 32
    updated: State = trainer(state=state, x=jnp.ones((2, 12)), y=jnp.eye(17)[:2], ids=jnp.arange(2))
    assert int(updated.step) == 1
    assert Config().inputs == 784 and Config().outputs == 10 and Config().memory_age == 8
    with pytest.raises(ValueError):
        Config(inputs=0)


@pytest.mark.parametrize("method", ["bp", "alm", "gdi"])
def test_benchmark_executes_updates_and_projects_counts(method: str) -> None:
    benchmark: Benchmark = Benchmark(
        config=Config(
            method=method,
            inputs=4,
            outputs=3,
            depth=2,
            width=2,
            capacity=4,
            batch=2,
            steps=2,
            epochs=3,
        ),
        examples=5,
        batches=2,
        rounds=2,
    )
    with patch("geodual.profile.perf_counter", side_effect=[0.0, 1.0, 2.0, 6.0, 7.0, 11.0]):
        result: dict[str, object] = benchmark()
    assert result["median_seconds_per_batch"] == 2
    assert result["estimated_epoch_compute_seconds"] == 6
    assert result["estimated_run_compute_seconds"] == 18
    assert result["kind"] == "synthetic_compute_only_not_imagenet_training"
    assert "Benchmark" in repr(benchmark)
    with pytest.raises(ValueError):
        Benchmark(config=Config(), batches=0)


def test_profile_cli_preserves_previous_results(tmp_path: Path) -> None:
    output: Path = tmp_path / "timing.json"
    with patch("geodual.profile.Benchmark.__call__", return_value={"kind": "test"}):
        main(["--method", "gdi", "--output", str(output)])
        assert json.loads(output.read_text()) == {"kind": "test"}
        with pytest.raises(SystemExit):
            main(["--method", "gdi", "--output", str(output)])
