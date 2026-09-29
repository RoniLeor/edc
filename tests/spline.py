"""Spline reference parity, affine reproduction, fallback safety, and retrieval integration."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RBFInterpolator

from geodual.cli import main
from geodual.memory import Memory, Records
from geodual.spline import Estimate, Spline
from geodual.training import Config, State, Trainer


def test_spline_matches_scipy_and_reproduces_affine_values() -> None:
    points: np.ndarray = np.random.default_rng(7).normal(size=(16, 4)).astype(np.float32)
    fallback: jax.Array = jnp.full((1, 16), 1 / 16)
    result: Estimate = Spline().weights(
        points=jnp.asarray(points[None]), fallback=fallback, available=jnp.ones((1, 16), dtype=bool)
    )
    reference: np.ndarray = RBFInterpolator(
        y=points, d=np.eye(16), smoothing=0.1, kernel="thin_plate_spline", degree=1
    )(np.zeros((1, 4)))
    assert bool(result.accepted[0])
    np.testing.assert_allclose(actual=result.weights, desired=reference, rtol=3e-5, atol=2e-6)
    values: np.ndarray = points @ np.asarray([2.0, -1.0, 0.2, 0.5]) + 3
    np.testing.assert_allclose(actual=np.asarray(result.weights) @ values, desired=3, atol=2e-6)


def test_projection_is_deterministic_translation_invariant_and_jittable() -> None:
    points: jax.Array = jnp.asarray(
        np.random.default_rng(8).normal(size=(2, 16, 8)), dtype=jnp.float32
    )
    query: jax.Array = jnp.zeros((2, 8))
    fallback: jax.Array = jnp.full((2, 16), 1 / 16)
    available: jax.Array = jnp.ones((2, 16), dtype=bool)
    spline: Spline = Spline()
    result: Estimate = spline(query=query, points=points, fallback=fallback, available=available)
    compiled: Estimate = jax.jit(spline)(
        query=query, points=points, fallback=fallback, available=available
    )
    shifted: Estimate = spline(
        query=query + 2, points=points + 2, fallback=fallback, available=available
    )
    assert np.asarray(result.accepted).all()
    np.testing.assert_allclose(actual=compiled.weights, desired=result.weights, atol=1e-6)
    np.testing.assert_allclose(actual=shifted.weights, desired=result.weights, atol=1e-6)


def test_insufficient_and_degenerate_neighbors_use_exact_fallback() -> None:
    points: jax.Array = jnp.asarray(
        np.random.default_rng(7).normal(size=(1, 16, 4)), dtype=jnp.float32
    )
    fallback: jax.Array = jax.nn.softmax(jnp.arange(16, dtype=jnp.float32))[None]
    missing: Estimate = Spline().weights(
        points=points,
        fallback=fallback,
        available=jnp.ones((1, 16), dtype=bool).at[0, 0].set(False),
    )
    degenerate: Estimate = Spline().weights(
        points=jnp.ones_like(points), fallback=fallback, available=jnp.ones((1, 16), dtype=bool)
    )
    extrapolated: Estimate = Spline().weights(
        points=points + 100, fallback=fallback, available=jnp.ones((1, 16), dtype=bool)
    )
    result: Estimate
    for result in (missing, degenerate, extrapolated):
        assert not bool(result.accepted[0])
        np.testing.assert_array_equal(actual=result.weights, desired=fallback)


def test_spline_memory_uses_fit_and_preserves_filters() -> None:
    rng: np.random.Generator = np.random.default_rng(4)
    keys: jax.Array = jnp.asarray(
        np.asarray([1.0, 0, 0, 0, 0, 0]) + 0.1 * rng.normal(size=(16, 6)), dtype=jnp.float32
    )
    errors: jax.Array = jnp.asarray(
        np.asarray([1.0, 0]) + 0.1 * rng.normal(size=(16, 2)), dtype=jnp.float32
    )
    memory: Memory = Memory(capacity=16, neighbors=16, initializer="spline")
    bank: Records = memory.insert(
        bank=memory.empty(layers=1, width=6, outputs=2),
        keys=keys,
        errors=errors,
        labels=jnp.zeros(16, dtype=int),
        ids=jnp.arange(16),
        duals=jnp.asarray(rng.normal(size=(1, 16, 6)), dtype=jnp.float32),
        step=jnp.asarray(0),
    )
    query: dict[str, jax.Array] = dict(
        keys=jnp.asarray([[1.0, 0, 0, 0, 0, 0]]),
        errors=jnp.asarray([[1.0, 0]]),
        labels=jnp.asarray([0]),
        ids=jnp.asarray([100]),
        step=jnp.asarray(1),
    )
    actual: jax.Array = memory(bank=bank, **query)
    average: jax.Array = replace(memory, initializer="average")(bank=bank, **query)
    assert np.isfinite(np.asarray(actual)).all()
    assert not np.allclose(actual, average)
    changed: Records
    for changed in (
        bank._replace(labels=jnp.ones(16, dtype=int)),
        bank._replace(ids=jnp.full((16,), 100)),
        bank._replace(stamps=jnp.full((16,), -8)),
    ):
        np.testing.assert_array_equal(actual=memory(bank=changed, **query), desired=0)
    np.testing.assert_array_equal(
        actual=memory(bank=bank, **(query | {"keys": -query["keys"]})), desired=0
    )
    singular: Records = bank._replace(
        keys=jnp.tile(query["keys"], (16, 1)), errors=jnp.tile(query["errors"], (16, 1))
    )
    np.testing.assert_array_equal(
        actual=memory(bank=singular, **query),
        desired=replace(memory, initializer="average")(bank=singular, **query),
    )


def test_spline_training_config_validation_and_cli(tmp_path: Path) -> None:
    trainer: Trainer = Trainer(
        config=Config(
            method="gdi",
            steps=4,
            inputs=3,
            outputs=2,
            width=6,
            depth=3,
            batch=8,
            capacity=32,
            neighbors=16,
            initializer="spline",
        )
    )
    state: State = trainer.initialize()
    x: jax.Array = jax.random.normal(jax.random.PRNGKey(1), shape=(8, 3))
    y: jax.Array = jnp.eye(2)[jnp.arange(8) % 2]
    index: int
    for index in range(6):
        state = trainer(state=state, x=x, y=y, ids=jnp.arange(8) + index * 8)
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(state))
    value: float
    for value in (-1.0, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="smoothing"):
            Spline(smoothing=value)
        with pytest.raises(ValueError, match="smoothing"):
            Config(spline_smoothing=value)
        with pytest.raises(ValueError, match="smoothing"):
            Memory(spline_smoothing=value)
    with pytest.raises(ValueError, match="initializer"):
        Config(initializer="bad")
    with pytest.raises(ValueError, match="initializer"):
        Memory(initializer="bad")
    with pytest.raises(ValueError, match="six"):
        Memory(initializer="spline")
    with pytest.raises(ValueError, match="six"):
        Config(initializer="spline")
    with pytest.raises(ValueError, match="GDI"):
        Config(method="bp", neighbors=16, initializer="spline")
    with pytest.raises(ValueError, match="Neighbors"):
        Config(neighbors=513)
    mocked: MagicMock
    with patch("geodual.cli.load"), patch("geodual.cli.Trainer") as mocked:
        mocked.return_value.run.return_value = {"ok": True}
        main(
            [
                "--method",
                "gdi",
                "--neighbors",
                "16",
                "--initializer",
                "spline",
                "--spline-smoothing",
                "0.2",
                "--output",
                str(tmp_path),
            ]
        )
        assert mocked.call_args.kwargs["config"].neighbors == 16
        assert mocked.call_args.kwargs["config"].initializer == "spline"
        assert mocked.call_args.kwargs["config"].spline_smoothing == 0.2
