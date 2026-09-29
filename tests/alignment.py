"""Full-gradient metric and fixed-weight production diagnostic checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geodual.alignment import Alignment, Measurement, cosine
from geodual.data import Examples
from geodual.model import Weights
from geodual.training import Config, Trainer


def test_global_cosine_weights_parameters_instead_of_layers() -> None:
    target: Weights = Weights(
        first=jnp.asarray([3.0]), hidden=jnp.asarray([4.0]), readout=jnp.asarray([0.0])
    )
    actual: Weights = target._replace(hidden=jnp.asarray([-4.0]))
    np.testing.assert_allclose(actual=cosine(actual=actual, target=target), desired=-7 / 25)
    np.testing.assert_allclose(actual=cosine(actual=target, target=target), desired=1)
    np.testing.assert_allclose(
        actual=cosine(actual=jax.tree.map(lambda x: -2 * x, target), target=target), desired=-1
    )
    np.testing.assert_allclose(
        actual=cosine(actual=jax.tree.map(jnp.zeros_like, target), target=target), desired=0
    )


def test_alignment_uses_fixed_weights_and_finite_real_gradients() -> None:
    config: Config = Config(inputs=2, outputs=2, width=4, depth=3, batch=4, capacity=8, steps=4)
    examples: Examples = Examples(
        x=np.random.default_rng(3).normal(size=(80, 2)).astype(np.float32),
        y=np.eye(2, dtype=np.float32)[np.arange(80) % 2],
        ids=np.arange(80),
    )
    weights: Weights = Trainer(config=config).initialize().weights
    before: list[np.ndarray] = [np.asarray(value).copy() for value in weights]
    diagnostic: Alignment = Alignment(config=config, budgets=(4,))
    rows: list[Measurement] = diagnostic(weights=weights, examples=examples)
    assert [row["method"] for row in rows] == ["alm", "gdi", "elastic"]
    for row in rows:
        assert row["steps"] == 4
        assert np.isfinite(row["global_cosine"]) and abs(row["global_cosine"]) <= 1.00001
        assert len(row["layer_cosines"]) == config.depth
        assert np.isfinite(row["layer_cosines"]).all()
    for actual, expected in zip(weights, before, strict=True):
        np.testing.assert_array_equal(actual=actual, desired=expected)
    with pytest.raises(ValueError, match="Insufficient"):
        diagnostic(
            weights=weights,
            examples=Examples(x=examples.x[:4], y=examples.y[:4], ids=examples.ids[:4]),
        )
