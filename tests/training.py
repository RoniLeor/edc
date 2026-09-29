"""Small real-implementation training and CLI integration tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geodual.cli import main
from geodual.data import Dataset, Examples
from geodual.training import Config, State, Trainer


@pytest.mark.parametrize("method", ["bp", "pc", "alm", "gdi", "mismatch"])
def test_training_methods_and_artifacts(method: str, tmp_path: Path) -> None:
    trainer: Trainer = Trainer(
        config=Config(method=method, depth=2, width=2, steps=2, batch=4, capacity=4, epochs=2)
    )
    rng: np.random.Generator = np.random.default_rng(4)
    examples: Examples = Examples(
        x=rng.normal(size=(5, 784)).astype(np.float32),
        y=np.eye(10, dtype=np.float32)[[0, 1, 2, 3, 4]],
        ids=np.arange(5, dtype=np.int64),
    )
    validation: Examples = Examples(x=examples.x.copy(), y=examples.y.copy(), ids=examples.ids + 10)
    test: Examples = Examples(x=examples.x.copy(), y=examples.y.copy(), ids=-examples.ids - 1)
    data: Dataset = Dataset(train=examples, validation=validation, test=test)
    state: State = trainer.initialize()
    initial: jax.Array = trainer.initial(
        state=state,
        x=jnp.asarray(examples.x),
        y=jnp.asarray(examples.y),
        ids=jnp.asarray(examples.ids),
    )
    np.testing.assert_array_equal(actual=initial, desired=0)
    updated: State = trainer(
        state=state,
        x=jnp.asarray(examples.x[:4]),
        y=jnp.asarray(examples.y[:4]),
        ids=jnp.asarray(examples.ids[:4]),
    )
    assert int(updated.step) == 1
    assert not np.array_equal(
        np.asarray(state.weights.readout), np.asarray(updated.weights.readout)
    )
    assert int(state.step) == 0
    assert "Trainer" in repr(trainer)
    result: dict[str, object] = trainer.run(data=data, output=tmp_path)
    assert (tmp_path / "weights.npz").is_file()
    assert result["train_examples"] == 5
    assert result["best_epoch"] in (1, 2)
    assert isinstance(result["history"], list)
    assert len(result["history"]) == 2
    if method in {"gdi", "mismatch"}:
        assert set(np.asarray(updated.bank.ids).tolist()) == {0, 1, 2, 3}
    else:
        np.testing.assert_array_equal(actual=updated.bank.ids, desired=-1)


def test_zero_gate_equals_alm() -> None:
    zero: Trainer = Trainer(
        config=Config(method="gdi", depth=3, width=2, steps=3, batch=4, capacity=4, strength=0)
    )
    baseline: Trainer = Trainer(
        config=Config(method="alm", depth=3, width=2, steps=3, batch=4, capacity=4)
    )
    a: State = zero.initialize()
    b: State = baseline.initialize()
    x: jax.Array = jax.random.normal(jax.random.PRNGKey(1), shape=(4, 784))
    y: jax.Array = jnp.eye(10)[:4]
    step: int
    for step in range(3):
        a = zero(state=a, x=x, y=y, ids=jnp.arange(4) + step * 4)
        b = baseline(state=b, x=x, y=y, ids=jnp.arange(4) + step * 4)
        left: jax.Array
        right: jax.Array
        for left, right in zip(a.weights, b.weights, strict=True):
            np.testing.assert_allclose(actual=left, desired=right, atol=1e-7)


def test_invalid_training_and_nonfinite_evaluation() -> None:
    with pytest.raises(ValueError):
        Config(method="unknown")
    with pytest.raises(ValueError):
        Config(epochs=0)
    with pytest.raises(ValueError):
        Config(capacity=1)
    trainer: Trainer = Trainer(config=Config(depth=2, width=2))
    examples: Examples = Examples(
        x=np.full((1, 784), np.nan, dtype=np.float32),
        y=np.eye(10, dtype=np.float32)[:1],
        ids=np.asarray([0], dtype=np.int64),
    )
    with pytest.raises(FloatingPointError):
        trainer.evaluate(weights=trainer.initialize().weights, examples=examples)


def test_cli_writes_summary_and_preserves_outputs(tmp_path: Path) -> None:
    run: MagicMock
    with (
        patch("geodual.cli.load"),
        patch("geodual.cli.Trainer.run", return_value={"ok": True}) as run,
    ):
        main(["--dataset", "mnist", "--method", "bp", "--output", str(tmp_path)])
        assert run.call_count == 1
        assert (tmp_path / "summary.json").read_text() == '{\n  "ok": true\n}\n'
        with pytest.raises(SystemExit):
            main(["--output", str(tmp_path)])


def test_training_preserves_frozen_ordinary_gdi() -> None:
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
