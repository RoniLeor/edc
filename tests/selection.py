"""Candidate safety, real probe scores, auxiliary budgets, and train-only updates."""

from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geodual.cli import main
from geodual.data import Dataset, Examples
from geodual.energy import Parameters, RankingState
from geodual.memory import Memory, Records
from geodual.model import Network
from geodual.selection import Choice, Probe, Selector
from geodual.settling import Equilibrium, Settler
from geodual.training import Config, State, Trainer


def test_candidates_preserve_retrieval_and_self_exclusion() -> None:
    memory: Memory = Memory(capacity=8)
    vector: jax.Array = jnp.asarray([[1.0, 0.0]])
    bank: Records = memory.insert(
        bank=memory.empty(layers=1, width=2, outputs=2),
        keys=vector,
        errors=vector,
        labels=jnp.asarray([0]),
        ids=jnp.asarray([3]),
        duals=jnp.ones((1, 1, 2)),
        step=jnp.asarray(0),
    )
    arguments: dict[str, jax.Array] = dict(
        keys=vector,
        errors=vector,
        labels=jnp.asarray([0]),
        ids=jnp.asarray([4]),
        step=jnp.asarray(1),
    )
    candidates: jax.Array = memory.candidates(bank=bank, **arguments)
    assert candidates.shape == (3, 1, 1, 2)
    np.testing.assert_array_equal(actual=candidates[0], desired=0)
    np.testing.assert_array_equal(actual=candidates[1], desired=memory(bank=bank, **arguments))
    np.testing.assert_array_equal(
        actual=candidates[2], desired=replace(memory, neighbors=1)(bank=bank, **arguments)
    )
    np.testing.assert_array_equal(
        actual=memory.candidates(bank=bank, **(arguments | {"ids": jnp.asarray([3])})), desired=0
    )
    with pytest.raises(ValueError, match="average retrieval"):
        replace(memory, neighbors=6, initializer="spline").candidates(bank=bank, **arguments)


def test_probe_merit_and_features_match_real_local_dynamics() -> None:
    trainer: Trainer = Trainer(
        config=Config(
            method="gdi",
            selector="residual",
            width=4,
            depth=3,
            inputs=2,
            outputs=2,
            batch=4,
            capacity=8,
            steps=2,
        )
    )
    state: State = trainer.initialize()
    x: jax.Array = jnp.asarray([[1.0, 0.1], [0.5, 0.3], [-1.0, 0.2], [-0.7, 0.1]])
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 0, 1, 1])]
    initial: jax.Array = jnp.zeros((2, 4, 4))
    probe: Probe = trainer.selection.probe(initial, weights=state.weights, x=x, y=y, steps=2)
    settled: Equilibrium = Settler(network=trainer.network, steps=2)(
        weights=state.weights, x=x, y=y, initial=initial
    )
    residual: jax.Array = trainer.network.residuals(
        weights=state.weights, x=x, states=settled.states
    )
    stationary: jax.Array = (
        jax.grad(trainer.network.energy, argnums=3)(
            state.weights, x, y, settled.states, settled.duals
        )
        * 4
    )
    np.testing.assert_allclose(
        actual=probe.merit, desired=jnp.mean(residual**2 + stationary**2, axis=(0, 2)), rtol=1e-5
    )
    assert probe.features.shape == (4, 12)
    assert np.isfinite(np.asarray(probe.features)).all()
    assessed: Probe = trainer.selection.assess(
        jnp.stack((initial, initial, initial)), weights=state.weights, x=x, y=y, steps=2
    )
    assert assessed.features.shape == (4, 3, 15)
    np.testing.assert_allclose(
        actual=assessed.merit, desired=jnp.repeat(probe.merit[:, None], 3, axis=1), rtol=1e-5
    )


def test_teacher_schedule_warmup_and_evaluation_do_not_leak_updates() -> None:
    config: Config = Config(
        method="gdi",
        selector="learned",
        width=4,
        depth=3,
        inputs=2,
        outputs=2,
        batch=4,
        capacity=8,
        steps=2,
    )
    learned: Trainer = Trainer(config=config)
    ordinary: Trainer = Trainer(config=replace(config, selector="none"))
    state: State = learned.initialize()
    baseline: State = ordinary.initialize()
    x: jax.Array = jnp.asarray([[1.0, 0.1], [0.5, 0.3], [-1.0, 0.2], [-0.7, 0.1]])
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 0, 1, 1])]
    index: int
    for index in range(33):
        state = learned(state=state, x=x, y=y, ids=jnp.arange(4) + index * 4)
        baseline = ordinary(state=baseline, x=x, y=y, ids=jnp.arange(4) + index * 4)
    assert state.selector is not None
    assert int(state.selector.updates) == 2
    np.testing.assert_array_equal(actual=state.selector.choices, desired=[0, 132, 0])
    left: jax.Array
    right: jax.Array
    for left, right in zip(state.weights, baseline.weights, strict=True):
        np.testing.assert_allclose(actual=left, desired=right, atol=1e-6, rtol=1e-6)
    choice: Choice = learned.selection(
        weights=state.weights,
        bank=state.bank,
        x=x,
        y=y,
        ids=jnp.arange(4) + 1000,
        step=jnp.asarray(32),
        movement=state.movement,
        state=state.selector,
        train=False,
    )
    for left, right in zip(
        jax.tree.leaves(choice.state), jax.tree.leaves(state.selector), strict=True
    ):
        np.testing.assert_array_equal(actual=left, desired=right)
    report: dict[str, object] = learned.selection.report(state.selector)
    assert report["teacher_updates"] == 2 and report["training_batches"] == 33
    assert report["extra_state_gradient_evaluations"] == 687
    assert report["scorer_parameters"] == 272


def test_learned_selection_uses_energy_after_warmup() -> None:
    trainer: Trainer = Trainer(
        config=Config(
            method="gdi",
            selector="learned",
            width=4,
            depth=3,
            inputs=2,
            outputs=2,
            batch=4,
            capacity=8,
            steps=2,
        )
    )
    state: State = trainer.initialize()
    x: jax.Array = jnp.asarray([[1.0, 0.1], [0.5, 0.3], [-1.0, 0.2], [-0.7, 0.1]])
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 0, 1, 1])]
    state = trainer(state=state, x=x, y=y, ids=jnp.arange(4))
    assert state.selector is not None
    parameters: Parameters = state.selector.parameters
    chosen: Parameters = parameters._replace(
        first=jnp.zeros_like(parameters.first).at[-1, 0].set(1),
        bias=jnp.zeros_like(parameters.bias),
        readout=jnp.zeros_like(parameters.readout).at[0].set(-1),
    )
    scorer: RankingState = state.selector._replace(parameters=chosen, updates=jnp.asarray(4))
    choice: Choice = trainer.selection(
        weights=state.weights,
        bank=state.bank,
        x=x,
        y=y,
        ids=jnp.arange(4) + 10,
        step=jnp.asarray(1),
        movement=state.movement,
        state=scorer,
        train=True,
    )
    states: jax.Array = trainer.network(weights=state.weights, x=x)
    candidates: jax.Array = trainer.memory.candidates(
        bank=state.bank,
        keys=states[-1],
        errors=trainer.network.error(
            prediction=trainer.network.output(weights=state.weights, states=states), y=y
        ),
        labels=jnp.argmax(y, axis=-1),
        ids=jnp.arange(4) + 10,
        step=jnp.asarray(1),
    )
    np.testing.assert_allclose(actual=choice.initial, desired=candidates[2], rtol=1e-6)
    np.testing.assert_array_equal(actual=choice.state.choices - scorer.choices, desired=[0, 0, 4])
    assert int(choice.state.updates) == 4


def test_residual_selection_and_energy_run_artifacts(tmp_path: Path) -> None:
    config: Config = Config(
        method="gdi",
        selector="residual",
        width=4,
        depth=3,
        inputs=2,
        outputs=2,
        batch=4,
        capacity=8,
        steps=2,
        epochs=1,
    )
    trainer: Trainer = Trainer(config=config)
    state: State = trainer.initialize()
    x: jax.Array = jnp.asarray([[1.0, 0.1], [0.5, 0.3], [-1.0, 0.2], [-0.7, 0.1]])
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 0, 1, 1])]
    state = trainer(state=state, x=x, y=y, ids=jnp.arange(4))
    assert state.selector is not None
    assert int(state.selector.updates) == 0
    assert int(state.selector.choices.sum()) == 4
    assert trainer.selection.report(state.selector)["extra_state_gradient_evaluations"] == 9
    examples: Examples = Examples(x=np.asarray(x), y=np.asarray(y), ids=np.arange(4))
    validation: Examples = Examples(
        x=(np.asarray(x) * 0.9).astype(np.float32), y=np.asarray(y), ids=np.arange(4) + 4
    )
    test: Examples = Examples(
        x=(np.asarray(x) * 1.1).astype(np.float32), y=np.asarray(y), ids=-1 - np.arange(4)
    )
    summary: dict[str, object] = Trainer(config=replace(config, selector="learned")).run(
        data=Dataset(train=examples, validation=validation, test=test), output=tmp_path
    )
    metadata: dict[str, object] = cast(dict[str, object], summary["selector"])
    assert metadata["teacher_updates"] == 1
    assert metadata["training_batches"] == 1
    assert (tmp_path / "selector.npz").exists()
    with np.load(tmp_path / "selector.npz") as arrays:
        assert set(arrays.files) == {"first", "bias", "readout"}
        assert all(np.isfinite(arrays[name]).all() for name in arrays.files)


def test_selector_validation_and_cli(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown energy"):
        Config(selector="bad")
    with pytest.raises(ValueError, match="squared-error GDI"):
        Config(method="bp", selector="learned")
    with pytest.raises(ValueError, match="squared-error GDI"):
        Config(objective="ce", selector="learned")
    with pytest.raises(ValueError, match="squared-error GDI"):
        Config(initializer="spline", neighbors=16, selector="learned")
    with pytest.raises(ValueError, match="squared-error"):
        Selector(network=Network(), memory=Memory(), mode="bad")
    mocked: MagicMock
    with patch("geodual.cli.load"), patch("geodual.cli.Trainer") as mocked:
        mocked.return_value.run.return_value = {"ok": True}
        main(["--method", "gdi", "--selector", "learned", "--output", str(tmp_path)])
        assert mocked.call_args.kwargs["config"].selector == "learned"
