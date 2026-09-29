"""Numerical parity, local dynamics, and retrieval safety checks."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geodual.memory import Memory, Records
from geodual.model import Network, Weights
from geodual.settling import Equilibrium, Settler


def test_reference_parity() -> None:
    """Match upstream initialization, forward pass, settling, and all weight gradients."""
    with np.load(Path(__file__).parent / "fixtures" / "reference.npz") as expected:
        model: Network = Network(width=3, depth=4, inputs=5, outputs=2)
        weights: Weights = model.initialize(seed=7)
        name: str
        value: jax.Array
        for name, value in zip(("first", "hidden", "readout"), weights, strict=True):
            np.testing.assert_allclose(actual=value, desired=expected[name], atol=1e-6)
        x: jax.Array = jnp.asarray(expected["x"])
        y: jax.Array = jnp.asarray(expected["y"])
        states: jax.Array = model(weights=weights, x=x)
        np.testing.assert_allclose(actual=states, desired=expected["forward"], atol=1e-6)
        np.testing.assert_allclose(
            actual=model.residuals(weights=weights, x=x, states=states), desired=0, atol=1e-6
        )
        solver: Settler = Settler(network=model, steps=8, rate=0.2)
        result: Equilibrium = solver(weights=weights, x=x, y=y, initial=jnp.zeros_like(states))
        np.testing.assert_allclose(actual=result.states, desired=expected["states"], atol=2e-6)
        np.testing.assert_allclose(actual=result.credit, desired=expected["credit"], atol=2e-6)
        gradient: Weights = solver.gradient(weights=weights, x=x, y=y, result=result)
        for name, value in zip(("first", "hidden", "readout"), gradient, strict=True):
            np.testing.assert_allclose(actual=value, desired=expected["grad_" + name], atol=2e-6)
        cosine: jax.Array
        residual: jax.Array
        stationarity: jax.Array
        cosine, residual, stationarity = solver.diagnostics(
            weights=weights, x=x, y=y, result=result
        )
        assert cosine.shape == (4,)
        assert np.isfinite(np.asarray(cosine)).all()
        assert float(residual) >= 0 and float(stationarity) >= 0
        assert float(model.loss(weights=weights, x=x, y=y)) > 0


def test_batch_scaling_timing_and_zero_dual_pc() -> None:
    """Activity dynamics are independent of batch duplication; final credit is pre-dual."""
    model: Network = Network(width=2, depth=2, inputs=2, outputs=2)
    weights: Weights = model.initialize(seed=5)
    x: jax.Array = jnp.ones((1, 2))
    y: jax.Array = jnp.zeros((1, 2))
    solver: Settler = Settler(network=model, steps=1, rate=0.1)
    initial: jax.Array = jnp.zeros((1, 1, 2))
    single: Equilibrium = solver(weights=weights, x=x, y=y, initial=initial)
    double: Equilibrium = solver(
        weights=weights,
        x=jnp.repeat(x, repeats=2, axis=0),
        y=jnp.repeat(y, repeats=2, axis=0),
        initial=jnp.repeat(initial, repeats=2, axis=1),
    )
    np.testing.assert_allclose(actual=single.states[:, 0], desired=double.states[:, 0])
    np.testing.assert_array_equal(actual=single.credit, desired=initial)
    np.testing.assert_allclose(
        actual=single.duals,
        desired=model.residuals(weights=weights, x=x, states=single.states),
        atol=float(np.finfo(np.float32).eps),  # XLA fusion and eager arithmetic differ by one ULP.
    )
    pc: Equilibrium = Settler(network=model, steps=3, alpha=0)(
        weights=weights, x=x, y=y, initial=initial
    )
    np.testing.assert_array_equal(actual=pc.duals, desired=initial)
    assert "Settler" in repr(solver) and "Network" in repr(model)


@pytest.mark.parametrize("objective", ["mse", "ce"])
def test_exact_equilibrium_recovers_bp(objective: str) -> None:
    """At feasible activities, negative supervised state adjoints give exact BP credit."""
    model: Network = Network(width=3, depth=4, inputs=5, outputs=2, objective=objective)
    weights: Weights = model.initialize(seed=9)
    x: jax.Array = jnp.ones((2, 5))
    y: jax.Array = jnp.eye(2)
    states: jax.Array = model(weights=weights, x=x)
    # A frozen upstream linear algebra oracle is unnecessary here: JAX differentiates
    # the production full loss, while the production settling rule differentiates local energy.
    solver: Settler = Settler(network=model, steps=1500, rate=0.1)
    result: Equilibrium = jax.jit(solver)(weights=weights, x=x, y=y, initial=jnp.zeros_like(states))
    actual: Weights = solver.gradient(weights=weights, x=x, y=y, result=result)
    expected: Weights = jax.grad(model.loss)(weights, x, y)
    left: jax.Array
    right: jax.Array
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual=left, desired=right, atol=2e-5, rtol=2e-4)


def test_memory_filters_age_labels_ids_and_empty() -> None:
    memory: Memory = Memory(capacity=4, neighbors=1, strength=1, max_age=2)
    bank: Records = memory.empty(layers=1, width=2, outputs=2)
    keys: jax.Array = jnp.asarray([[1.0, 0.0]])
    errors: jax.Array = jnp.asarray([[0.0, 1.0]])
    labels: jax.Array = jnp.asarray([1])
    ids: jax.Array = jnp.asarray([5])
    empty: jax.Array = memory(
        bank=bank, keys=keys, errors=errors, labels=labels, ids=ids, step=jnp.asarray(0)
    )
    np.testing.assert_array_equal(actual=empty, desired=0)
    bank = memory.insert(
        bank=bank,
        keys=keys,
        errors=errors,
        labels=labels,
        ids=ids,
        duals=jnp.ones((1, 1, 2)),
        step=jnp.asarray(0),
    )
    retrieved: jax.Array = memory(
        bank=bank,
        keys=keys,
        errors=errors,
        labels=labels,
        ids=jnp.asarray([6]),
        step=jnp.asarray(1),
    )
    np.testing.assert_allclose(actual=retrieved, desired=np.exp(-0.5))
    label: int
    query_id: int
    step: int
    for label, query_id, step in ((1, 5, 1), (0, 6, 1), (1, 6, 3)):
        excluded: jax.Array = memory(
            bank=bank,
            keys=keys,
            errors=errors,
            labels=jnp.asarray([label]),
            ids=jnp.asarray([query_id]),
            step=jnp.asarray(step),
        )
        np.testing.assert_array_equal(actual=excluded, desired=0)
    zero: Memory = Memory(capacity=4, neighbors=1, strength=0)
    np.testing.assert_array_equal(
        actual=zero(
            bank=bank,
            keys=keys,
            errors=errors,
            labels=labels,
            ids=jnp.asarray([6]),
            step=jnp.asarray(1),
        ),
        desired=0,
    )
    mismatch: Memory = Memory(capacity=4, neighbors=1, mismatch=True)
    wrong: jax.Array = mismatch(
        bank=bank,
        keys=keys,
        errors=errors,
        labels=jnp.asarray([0]),
        ids=jnp.asarray([6]),
        step=jnp.asarray(1),
    )
    assert float(jnp.linalg.norm(wrong)) > 0
    assert "Memory" in repr(memory)


def test_memory_rollover_and_low_similarity() -> None:
    memory: Memory = Memory(capacity=2, neighbors=2)
    bank: Records = memory.empty(layers=1, width=2, outputs=2)
    index: int
    for index in range(3):
        bank = memory.insert(
            bank=bank,
            keys=jnp.ones((1, 2)),
            errors=jnp.ones((1, 2)),
            labels=jnp.asarray([0]),
            ids=jnp.asarray([index]),
            duals=jnp.ones((1, 1, 2)),
            step=jnp.asarray(index),
        )
    assert set(np.asarray(bank.ids).tolist()) == {1, 2}
    result: jax.Array = memory(
        bank=bank,
        keys=-jnp.ones((1, 2)),
        errors=jnp.ones((1, 2)),
        labels=jnp.asarray([0]),
        ids=jnp.asarray([3]),
        step=jnp.asarray(3),
    )
    np.testing.assert_array_equal(actual=result, desired=0)
    with pytest.raises(ValueError):
        memory.insert(
            bank=bank,
            keys=jnp.ones((3, 2)),
            errors=jnp.ones((3, 2)),
            labels=jnp.zeros(3),
            ids=jnp.arange(3),
            duals=jnp.ones((1, 3, 2)),
            step=jnp.asarray(3),
        )


def test_invalid_configurations() -> None:
    with pytest.raises(ValueError):
        Network(depth=1)
    with pytest.raises(ValueError):
        Settler(network=Network(), steps=0)
    with pytest.raises(ValueError):
        Memory(neighbors=0)
    with pytest.raises(ValueError):
        Memory(max_age=0)
