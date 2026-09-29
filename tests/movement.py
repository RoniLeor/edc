"""Optimizer movement gating, timestamp timing, and ordinary-GDI regression tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geodual.cli import main
from geodual.memory import Memory, Records
from geodual.model import Weights
from geodual.training import Config, State, Trainer


def test_drift_gate_tracks_model_movement_and_fifo() -> None:
    memory: Memory = Memory(capacity=2, neighbors=1, strength=1, max_age=10, drift_scale=0.01)
    bank: Records = memory.empty(layers=1, width=2, outputs=2)
    vector: jax.Array = jnp.asarray([[1.0, 0.0]])
    bank = memory.insert(
        bank=bank,
        keys=vector,
        errors=vector,
        labels=jnp.asarray([0]),
        ids=jnp.asarray([1]),
        duals=jnp.ones((1, 1, 2)),
        step=jnp.asarray(0),
        movement=jnp.asarray(0.1),
    )
    fresh: jax.Array = memory(
        bank=bank,
        keys=vector,
        errors=vector,
        labels=jnp.asarray([0]),
        ids=jnp.asarray([2]),
        step=jnp.asarray(1),
        movement=jnp.asarray(0.1),
    )
    drifted: jax.Array = memory(
        bank=bank,
        keys=vector,
        errors=vector,
        labels=jnp.asarray([0]),
        ids=jnp.asarray([2]),
        step=jnp.asarray(1),
        movement=jnp.asarray(0.12),
    )
    np.testing.assert_allclose(actual=fresh, desired=np.exp(-0.1), rtol=1e-6)
    np.testing.assert_allclose(actual=drifted, desired=fresh * np.exp(-2), rtol=2e-6)
    expired: jax.Array = memory(
        bank=bank,
        keys=vector,
        errors=vector,
        labels=jnp.asarray([0]),
        ids=jnp.asarray([2]),
        step=jnp.asarray(11),
        movement=jnp.asarray(0.12),
    )
    np.testing.assert_array_equal(actual=expired, desired=0)
    with pytest.raises(ValueError, match="movement is required"):
        memory(
            bank=bank,
            keys=vector,
            errors=vector,
            labels=jnp.asarray([0]),
            ids=jnp.asarray([2]),
            step=jnp.asarray(1),
        )
    with pytest.raises(ValueError, match="movement is required"):
        memory.insert(
            bank=bank,
            keys=vector,
            errors=vector,
            labels=jnp.asarray([0]),
            ids=jnp.asarray([2]),
            duals=jnp.ones((1, 1, 2)),
            step=jnp.asarray(1),
        )
    index: int
    for index in (2, 3):
        bank = memory.insert(
            bank=bank,
            keys=vector,
            errors=vector,
            labels=jnp.asarray([0]),
            ids=jnp.asarray([index]),
            duals=jnp.ones((1, 1, 2)),
            step=jnp.asarray(index),
            movement=jnp.asarray(index / 10),
        )
    np.testing.assert_allclose(actual=bank.movement, desired=[0.3, 0.2], rtol=1e-6)
    assert set(np.asarray(bank.ids)) == {2, 3}


def test_disabled_gate_exactly_preserves_frozen_ordinary_gdi() -> None:
    trainer: Trainer = Trainer(
        config=Config(
            method="gdi",
            objective="ce",
            depth=3,
            width=4,
            inputs=2,
            outputs=2,
            batch=4,
            capacity=8,
            steps=4,
        )
    )
    state: State = trainer.initialize()
    x: jax.Array = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 1, 0, 1])]
    step: int
    for step in range(6):
        state = trainer(state=state, x=x, y=y, ids=jnp.arange(4) + 4 * step)
    with np.load(Path(__file__).parent / "fixtures" / "movement.npz") as expected:
        name: str
        value: jax.Array
        for name, value in zip(("first", "hidden", "readout"), state.weights, strict=True):
            np.testing.assert_array_equal(actual=value, desired=expected[name])
        np.testing.assert_array_equal(actual=state.bank.duals, desired=expected["duals"])
    assert float(state.movement) == 0
    assert state.bank.movement.size == 0


def test_actual_adam_update_distance_and_preupdate_memory_stamp() -> None:
    trainer: Trainer = Trainer(
        config=Config(
            method="gdi",
            objective="ce",
            depth=2,
            width=2,
            inputs=2,
            outputs=2,
            batch=4,
            capacity=4,
            steps=4,
            drift_scale=0.01,
        )
    )
    known: Weights = Weights(
        first=jnp.asarray([[3.0, 4.0]]), hidden=jnp.zeros((0, 1, 1)), readout=jnp.zeros((1, 1))
    )
    updates: Weights = known._replace(first=jnp.asarray([[0.0, 1.0]]))
    np.testing.assert_allclose(actual=trainer.distance(weights=known, updates=updates), desired=0.2)
    np.testing.assert_allclose(
        actual=trainer.distance(weights=known, updates=jax.tree.map(jnp.zeros_like, known)),
        desired=0,
    )
    state: State = trainer.initialize()
    x: jax.Array = jnp.ones((4, 2))
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 1, 0, 1])]
    updated: State = trainer(state=state, x=x, y=y, ids=jnp.arange(4))
    assert float(updated.movement) > 0
    np.testing.assert_array_equal(actual=updated.bank.movement, desired=0)
    next_state: State = trainer(state=updated, x=x, y=y, ids=jnp.arange(4) + 4)
    assert float(next_state.movement) > float(updated.movement)
    np.testing.assert_array_equal(
        actual=next_state.bank.movement, desired=jnp.full((4,), updated.movement)
    )
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(next_state))


def test_gate_validation_and_cli(tmp_path: Path) -> None:
    value: float
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Drift scale"):
            Memory(drift_scale=value)
        with pytest.raises(ValueError, match="Drift scale"):
            Config(drift_scale=value)
    with pytest.raises(ValueError, match="only defined for GDI"):
        Config(method="alm", drift_scale=0.01)
    trainer: MagicMock
    with patch("geodual.cli.load"), patch("geodual.cli.Trainer") as trainer:
        trainer.return_value.run.return_value = {"ok": True}
        main(["--method", "gdi", "--drift-scale", "0.01", "--output", str(tmp_path)])
        assert trainer.call_args.kwargs["config"].drift_scale == 0.01
