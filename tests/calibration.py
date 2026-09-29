"""Fixed-weight bank construction and held-out query separation."""

import jax
import numpy as np

from edc.calibration import Calibration
from edc.data import Examples
from edc.model import Network, Weights
from edc.training import Config, State


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
