"""Symmetric dual transport, fade schedule, unchanged baselines and CLI wiring."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geodual.cli import main
from geodual.model import Network, Weights
from geodual.settling import Equilibrium, Settler
from geodual.training import Config, State, Trainer


@pytest.mark.parametrize("layers", [4, 5])
def test_pair_updates_preserve_sum_and_credit_and_fade(layers: int) -> None:
    network: Network = Network(width=2, depth=layers + 1, inputs=2, outputs=2)
    weights: Weights = network.initialize(seed=0)
    x: jax.Array = jnp.ones((2, 2))
    y: jax.Array = jnp.eye(2)
    duals: jax.Array = jnp.arange(layers * 4, dtype=jnp.float32).reshape(layers, 2, 2) / 10
    carry: Equilibrium = Equilibrium(
        states=network(weights=weights, x=x), duals=duals, credit=jnp.zeros_like(duals)
    )
    baseline: Settler = Settler(network=network, steps=16)
    elastic: Settler = replace(baseline, elastic=0.05)
    index: int
    for index in (0, 3, 7, 8, 15):
        plain: Equilibrium = baseline.step(index=index, carry=carry, weights=weights, x=x, y=y)
        paired: Equilibrium = elastic.step(index=index, carry=carry, weights=weights, x=x, y=y)
        np.testing.assert_array_equal(actual=paired.states, desired=plain.states)
        np.testing.assert_array_equal(actual=paired.credit, desired=duals)
        np.testing.assert_allclose(
            actual=paired.duals + paired.duals[::-1],
            desired=plain.duals + plain.duals[::-1],
            atol=5e-7,
        )
        if layers % 2:
            np.testing.assert_array_equal(
                actual=paired.duals[layers // 2], desired=plain.duals[layers // 2]
            )
        if index >= 7:
            np.testing.assert_array_equal(actual=paired.duals, desired=plain.duals)
        else:
            expected_strength: float = 0.05 if index == 0 else 0.05 * 4 / 7
            np.testing.assert_allclose(
                actual=paired.duals[0] - plain.duals[0],
                desired=expected_strength * (duals[-1] - duals[0]),
                atol=1e-7,
            )
            np.testing.assert_allclose(
                actual=paired.duals[1] - plain.duals[1],
                desired=expected_strength * (duals[-2] - duals[1]),
                atol=1e-7,
            )


def test_elastic_changes_real_jitted_training_and_is_finite() -> None:
    config: Config = Config(
        method="gdi", width=4, depth=5, inputs=2, outputs=2, batch=4, capacity=8, steps=16
    )
    plain: Trainer = Trainer(config=config)
    coupled: Trainer = Trainer(config=replace(config, elastic=0.05))
    assert coupled.settler.elastic == 0.05
    a: State = plain.initialize()
    b: State = coupled.initialize()
    x: jax.Array = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    y: jax.Array = jnp.eye(2)[jnp.asarray([0, 1, 0, 1])]
    index: int
    for index in range(5):
        a = plain(state=a, x=x, y=y, ids=jnp.arange(4) + index * 4)
        b = coupled(state=b, x=x, y=y, ids=jnp.arange(4) + index * 4)
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(b))
    assert not np.array_equal(np.asarray(a.bank.duals), np.asarray(b.bank.duals))
    assert not np.array_equal(np.asarray(a.weights.hidden), np.asarray(b.weights.hidden))
    assert int(b.step) == 5
    assert b.selector is None


def test_elastic_validation_and_cli(tmp_path: Path) -> None:
    network: Network = Network(width=2, depth=3, inputs=2, outputs=2)
    value: float
    for value in (-0.1, 0.51, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Elastic strength"):
            Config(elastic=value)
        with pytest.raises(ValueError, match="Elastic strength"):
            Settler(network=network, steps=16, elastic=value)
    with pytest.raises(ValueError, match="at least four"):
        Config(elastic=0.05, steps=3)
    with pytest.raises(ValueError, match="at least four"):
        Settler(network=network, steps=3, elastic=0.05)
    with pytest.raises(ValueError, match="dual ascent"):
        Settler(network=network, steps=16, elastic=0.05, alpha=0)
    with pytest.raises(ValueError, match="requires GDI"):
        Config(method="bp", elastic=0.05)
    with pytest.raises(ValueError, match="no selector"):
        Config(selector="learned", elastic=0.05)
    trainer: MagicMock
    with patch("geodual.cli.load"), patch("geodual.cli.Trainer") as trainer:
        trainer.return_value.run.return_value = {"ok": True}
        main(["--method", "gdi", "--steps", "16", "--elastic", "0.05", "--output", str(tmp_path)])
        assert trainer.call_args.kwargs["config"].elastic == 0.05
