"""Loss gradients and objective-consistent GDI retrieval checks."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geodual.cli import main
from geodual.data import Examples
from geodual.memory import Memory
from geodual.model import Network
from geodual.training import Config, State, Trainer


@pytest.mark.parametrize("objective", ["mse", "ce"])
def test_output_error_is_supervised_gradient(objective: str) -> None:
    network: Network = Network(objective=objective)
    prediction: jax.Array = jnp.asarray([0.3, -0.1, 0.7])
    target: jax.Array = jnp.asarray([0.0, 1.0, 0.0])
    actual: jax.Array = jax.grad(network.supervised)(prediction, target)
    np.testing.assert_allclose(
        actual=network.error(prediction=prediction, y=target), desired=actual, atol=1e-7
    )
    if objective == "ce":
        np.testing.assert_allclose(
            actual=network.supervised(prediction=jnp.zeros(3), y=target),
            desired=np.log(3),
            rtol=1e-6,
        )
        assert np.isfinite(float(network.supervised(prediction=prediction * 10000, y=target)))
    else:
        np.testing.assert_allclose(
            actual=network.supervised(prediction=prediction, y=target),
            desired=0.5 * np.sum((np.asarray(prediction) - np.asarray(target)) ** 2),
        )


@pytest.mark.parametrize("method", ["bp", "alm", "gdi"])
def test_ce_updates_metrics_and_gdi_bank(method: str) -> None:
    trainer: Trainer = Trainer(
        config=Config(
            method=method,
            objective="ce",
            width=4,
            depth=3,
            inputs=2,
            outputs=2,
            batch=4,
            capacity=4,
            steps=16,
        )
    )
    state: State = trainer.initialize()
    x: jax.Array = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 1, 0, 1])]
    ids: jax.Array = jnp.arange(4)
    logits: jax.Array = trainer.network.output(
        weights=state.weights, states=trainer.network(weights=state.weights, x=x)
    )
    expected_errors: jax.Array = jax.nn.softmax(logits, axis=-1) - y
    if method == "gdi":
        lookup: MagicMock
        with patch.object(Memory, "__call__", return_value=jnp.zeros((2, 4, 4))) as lookup:
            trainer.initial(state=state, x=x, y=y, ids=ids)
            np.testing.assert_allclose(
                actual=lookup.call_args.kwargs["errors"], desired=expected_errors
            )
    updated: State = trainer(state=state, x=x, y=y, ids=ids)
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(updated))
    assert not np.array_equal(
        np.asarray(state.weights.readout), np.asarray(updated.weights.readout)
    )
    if method == "gdi":
        np.testing.assert_allclose(
            actual=updated.bank.errors,
            desired=expected_errors / jnp.linalg.norm(expected_errors, axis=-1, keepdims=True),
            atol=1e-7,
        )
    examples: Examples = Examples(x=np.asarray(x), y=np.asarray(y), ids=np.asarray(ids))
    metrics: dict[str, float] = trainer.evaluate(weights=updated.weights, examples=examples)
    assert set(metrics) == {"accuracy", "ce"}
    np.testing.assert_allclose(
        actual=metrics["ce"],
        desired=trainer.network.loss(weights=updated.weights, x=x, y=y),
        rtol=1e-6,
    )


def test_objective_validation_and_cli(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Objective"):
        Config(objective="unknown")
    with pytest.raises(ValueError, match="Objective"):
        Network(objective="unknown")
    trainer: MagicMock
    with patch("geodual.cli.load"), patch("geodual.cli.Trainer") as trainer:
        trainer.return_value.run.return_value = {"ok": True}
        main(["--objective", "ce", "--output", str(tmp_path)])
        assert trainer.call_args.kwargs["config"].objective == "ce"
