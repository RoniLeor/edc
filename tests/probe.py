"""Frozen probe training, train-only normalization, and validation selection."""

import numpy as np

from geodual.data import Examples
from geodual.probe import REGULARIZATIONS, Probe, fit


def test_probe_separates_classes_and_handles_constant_features() -> None:
    train: Examples = Examples(
        x=np.asarray([[-2.0, 7.0], [-1.0, 7.0], [1.0, 7.0], [2.0, 7.0]], dtype=np.float32),
        y=np.eye(2, dtype=np.float32)[[0, 0, 1, 1]],
        ids=np.arange(4),
    )
    validation: Examples = Examples(
        x=np.asarray([[-3.0, 7.0], [3.0, 7.0]], dtype=np.float32),
        y=np.eye(2, dtype=np.float32),
        ids=np.asarray([4, 5]),
    )
    original: np.ndarray = train.x.copy()
    probe: Probe = fit(train=train, validation=validation)
    assert probe.validation_accuracy == 1.0
    assert probe.regularization == REGULARIZATIONS[0]
    np.testing.assert_array_equal(actual=probe(validation.x), desired=[0, 1])
    np.testing.assert_array_equal(actual=probe.center, desired=[0.0, 7.0])
    assert np.isfinite(probe.weights).all()
    np.testing.assert_array_equal(actual=train.x, desired=original)
