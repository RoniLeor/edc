"""Isolated checks for the training-only loss diagnostic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geodual.data import Examples
from geodual.memorization import Memorization, main
from geodual.model import Weights
from geodual.training import Config, State, Trainer


def test_mse_step_matches_production_bp_and_ce_loss() -> None:
    config: Config = Config(method="bp", width=4, depth=2, inputs=2, outputs=2, batch=4, capacity=4)
    trainer: Trainer = Trainer(config=config)
    state: State = trainer.initialize()
    x: jax.Array = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 1, 0, 1])]
    ids: jax.Array = jnp.arange(4)
    diagnostic: Memorization = Memorization(config=config, objective="mse")
    expected: State = trainer(state=state, x=x, y=y, ids=ids)
    actual: State = diagnostic.step(state=state, x=x, y=y, ids=ids)
    left: jax.Array
    right: jax.Array
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(actual=left, desired=right)
    np.testing.assert_allclose(
        actual=diagnostic.loss(weights=state.weights, x=x, y=y),
        desired=trainer.network.loss(weights=state.weights, x=x, y=y),
    )
    zero: Weights = state.weights._replace(readout=jnp.zeros_like(state.weights.readout))
    ce: Memorization = Memorization(config=config, objective="ce")
    np.testing.assert_allclose(actual=ce.loss(weights=zero, x=x, y=y), desired=np.log(2), rtol=1e-6)
    updated: State = ce.step(state=state, x=x, y=y, ids=ids)
    assert float(ce.loss(weights=updated.weights, x=x, y=y)) < float(
        ce.loss(weights=state.weights, x=x, y=y)
    )


@pytest.mark.parametrize("objective", ["mse", "ce"])
def test_memorization_learns_without_changing_examples(objective: str) -> None:
    config: Config = Config(
        method="bp", width=8, depth=2, inputs=2, outputs=2, batch=4, capacity=4, learning_rate=0.03
    )
    x: np.ndarray = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    examples: Examples = Examples(
        x=x.copy(), y=np.eye(2, dtype=np.float32)[[0, 1, 0, 1]], ids=np.arange(4, dtype=np.int64)
    )
    result: dict = Memorization(config=config, objective=objective, updates=100)(examples=examples)
    assert result["history"][-1]["accuracy"] == 1.0
    assert result["history"][-1]["loss"] < result["history"][0]["loss"]
    assert result["kind"] == "training_only_memorization_not_generalization"
    np.testing.assert_array_equal(actual=examples.x, desired=x)
    with pytest.raises(ValueError, match="full batch"):
        Memorization(config=Config(method="bp", batch=8), objective=objective)(examples=examples)


def test_diagnostic_validation_and_cli(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BP"):
        Memorization(config=Config(), objective="ce")
    with pytest.raises(ValueError, match="BP"):
        Memorization(config=Config(method="bp"), objective="unknown")
    with pytest.raises(ValueError, match="Positive"):
        Memorization(config=Config(method="bp"), objective="mse", updates=0)
    data: Path = tmp_path / "train.npz"
    np.savez(
        data, pixels=np.zeros((64, 12288), dtype=np.uint8), labels=np.arange(64), ids=np.arange(64)
    )
    output: Path = tmp_path / "results"
    run: MagicMock
    with patch.object(Memorization, "__call__", return_value={"ok": True}) as run:
        main(["--data", str(data), "--output", str(output), "--examples", "64", "--updates", "1"])
        assert run.call_count == 4
        assert len(list(output.glob("*.json"))) == 4
        assert run.call_args.kwargs["examples"].x.shape == (64, 12288)
        with pytest.raises(SystemExit):
            main(["--output", str(output)])
        with pytest.raises(SystemExit):
            main(["--data", str(data), "--output", str(tmp_path / "bad"), "--examples", "65"])
