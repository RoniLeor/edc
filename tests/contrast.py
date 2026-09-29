"""Supervised contrastive loss, coupled gradients, retrieval, and paired augmentations."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geodual.contrast import SupCon, Views
from geodual.model import Network
from geodual.training import Config, State, Trainer


def test_supcon_values_and_coupled_gradient() -> None:
    loss: SupCon = SupCon()
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 0, 1, 1])]
    collapsed: jax.Array = jnp.ones((4, 3))
    np.testing.assert_allclose(actual=loss(prediction=collapsed, y=y), desired=np.log(3), rtol=1e-6)
    separated: jax.Array = jnp.asarray([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    assert float(loss.total(prediction=separated, y=y)) < 0.001
    assert float(loss.total(prediction=separated, y=y[::-1].at[1].set(y[1]))) > 1
    prediction: jax.Array = jnp.asarray([[1.0, 0.3], [0.7, 0.5], [-0.4, 1.0], [-0.9, -0.2]])
    gradient: jax.Array = loss.gradient(prediction=prediction, y=y)
    epsilon: float = 0.001
    finite_difference: float = float(
        (
            loss.total(prediction=prediction.at[0, 1].add(epsilon), y=y)
            - loss.total(prediction=prediction.at[0, 1].add(-epsilon), y=y)
        )
        / (2 * epsilon)
    )
    np.testing.assert_allclose(actual=gradient[0, 1], desired=finite_difference, rtol=0.002)
    network: Network = Network(objective="supcon")
    np.testing.assert_array_equal(
        actual=network.error(prediction=prediction, y=y), desired=gradient
    )
    np.testing.assert_array_equal(
        actual=network.supervised(prediction=prediction, y=y),
        desired=loss(prediction=prediction, y=y),
    )
    assert np.isfinite(np.asarray(loss.gradient(prediction=jnp.zeros((4, 2)), y=y))).all()
    assert np.isfinite(np.asarray(loss(prediction=prediction, y=jnp.eye(4)))).all()


def test_views_are_paired_deterministic_and_padded() -> None:
    x: np.ndarray = np.zeros((3, 28, 28), dtype=np.float32)
    x[:, 10:15, 10:15] = np.arange(1, 4, dtype=np.float32)[:, None, None]
    original: np.ndarray = x.copy()
    views: Views = Views(background=0.0)
    first: np.ndarray = views(x=x.reshape(3, -1), seed=8)
    np.testing.assert_array_equal(actual=first, desired=views(x=x.reshape(3, -1), seed=8))
    assert first.shape == (6, 784)
    np.testing.assert_allclose(actual=first.sum(axis=1), desired=[25, 50, 75, 25, 50, 75])
    assert not np.array_equal(first, views(x=x.reshape(3, -1), seed=9))
    np.testing.assert_array_equal(actual=x, desired=original)
    assert not np.any(first.reshape(6, 28, 28)[:, 0])


@pytest.mark.parametrize("method", ["bp", "alm", "gdi"])
def test_supcon_training_and_retrieval(method: str) -> None:
    config: Config = Config(
        method=method,
        objective="supcon",
        inputs=2,
        outputs=3,
        width=4,
        depth=3,
        batch=4,
        capacity=8,
        steps=4,
    )
    trainer: Trainer = Trainer(config=config)
    state: State = trainer.initialize()
    x: jax.Array = jnp.asarray([[1.0, 0.2], [0.8, 0.3], [-1.0, 0.4], [-0.8, 0.5]])
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 0, 1, 1])]
    ids: jax.Array = jnp.asarray([0, 0, 1, 1])
    prediction: jax.Array = trainer.network.output(
        weights=state.weights, states=trainer.network(weights=state.weights, x=x)
    )
    errors: jax.Array = SupCon().gradient(prediction=prediction, y=y)
    updated: State = trainer(state=state, x=x, y=y, ids=ids)
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(updated))
    assert not np.array_equal(state.weights.readout, updated.weights.readout)
    if method == "gdi":
        np.testing.assert_allclose(
            actual=updated.bank.errors[:4],
            desired=errors / jnp.maximum(jnp.linalg.norm(errors, axis=-1, keepdims=True), 1e-8),
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            actual=trainer.initial(state=updated, x=x, y=y, ids=ids), desired=jnp.zeros((2, 4, 4))
        )
        assert float(jnp.linalg.norm(trainer.initial(state=updated, x=x, y=y, ids=ids + 10))) > 0
    with pytest.raises(ValueError, match="probe"):
        trainer.metrics(weights=updated.weights, x=x, y=y)


def test_zero_strength_supcon_matches_alm() -> None:
    config: Config = Config(
        method="alm",
        objective="supcon",
        inputs=2,
        outputs=3,
        width=4,
        depth=3,
        batch=4,
        capacity=8,
        steps=4,
    )
    alm: Trainer = Trainer(config=config)
    gdi: Trainer = Trainer(config=replace(config, method="gdi", strength=0.0))
    x: jax.Array = jnp.asarray([[1.0, 0.2], [0.8, 0.3], [-1.0, 0.4], [-0.8, 0.5]])
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 0, 1, 1])]
    left: State = alm.initialize()
    right: State = gdi.initialize()
    index: int
    for index in range(3):
        left = alm(state=left, x=x, y=y, ids=jnp.arange(4) + index * 4)
        right = gdi(state=right, x=x, y=y, ids=jnp.arange(4) + index * 4)
    first: jax.Array
    second: jax.Array
    for first, second in zip(left.weights, right.weights, strict=True):
        np.testing.assert_array_equal(actual=first, desired=second)
