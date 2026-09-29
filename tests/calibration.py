"""Exact diagnostic multipliers and disjoint fixed-weight calibration coverage."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geodual.calibration import Adjoints, Calibration
from geodual.data import Examples
from geodual.model import Network, Weights
from geodual.settling import Equilibrium, Settler
from geodual.training import Config, State


@pytest.mark.parametrize("depth", [2, 4])
def test_exact_adjoint_multipliers_recover_bp(depth: int) -> None:
    network: Network = Network(inputs=3, outputs=2, width=4, depth=depth)
    weights: Weights = network.initialize(seed=5)
    x: jax.Array = jax.random.normal(jax.random.PRNGKey(8), shape=(8, 3))
    y: jax.Array = jnp.eye(2)[jnp.arange(8) % 2]
    states: jax.Array = network(weights=weights, x=x)
    duals: jax.Array = Adjoints(network=network)(weights=weights, x=x, y=y)
    actual: Weights = Settler(network=network, steps=1).gradient(
        weights=weights, x=x, y=y, result=Equilibrium(states=states, duals=duals, credit=duals)
    )
    expected: Weights = jax.grad(network.loss)(weights, x, y)
    left: jax.Array
    right: jax.Array
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual=left, desired=right, rtol=1e-5, atol=1e-6)


def test_calibration_preserves_weights_and_uses_disjoint_queries() -> None:
    config: Config = Config(
        method="gdi", inputs=2, outputs=2, width=4, depth=3, batch=8, capacity=32, steps=2
    )
    examples: Examples = Examples(
        x=np.random.default_rng(1).normal(size=(160, 2)).astype(np.float32),
        y=np.eye(2, dtype=np.float32)[np.arange(160) % 2],
        ids=np.arange(160),
    )
    weights: Weights = Network(inputs=2, outputs=2, width=4, depth=3).initialize(seed=0)
    calibration: Calibration = Calibration(config=config)
    state: State = calibration.fill(weights=weights, examples=examples)
    assert int(state.step) == 16
    assert set(np.asarray(state.bank.ids)).isdisjoint(set(examples.ids[128:]))
    left: jax.Array
    right: jax.Array
    for left, right in zip(state.weights, weights, strict=True):
        np.testing.assert_array_equal(actual=left, desired=right)
    rows: dict[str, object] = calibration(weights=weights, examples=examples)
    assert set(rows) == {"zero", "gdi4", "gdi16", "spline16"}
    zero: dict[str, float] = cast(dict[str, float], rows["zero"])
    assert zero["initial_relative_rmse"] == 1.0
    row: object
    for row in rows.values():
        assert all(np.isfinite(value) for value in cast(dict[str, float], row).values())
    with pytest.raises(ValueError, match="Insufficient"):
        calibration(
            weights=weights,
            examples=Examples(x=examples.x[:8], y=examples.y[:8], ids=examples.ids[:8]),
        )
