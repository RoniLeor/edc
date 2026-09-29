"""Finite-candidate energy normalization, proxy targets, and auxiliary learning."""

import jax
import jax.numpy as jnp
import numpy as np

from geodual.energy import Parameters, Ranker, RankingState


def test_energy_probabilities_and_teacher_targets() -> None:
    ranker: Ranker = Ranker(inputs=3)
    state: RankingState = ranker.initialize(seed=0)
    features: jax.Array = jnp.stack((jnp.eye(3), jnp.eye(3)[::-1]))
    np.testing.assert_allclose(
        actual=ranker.probabilities(parameters=state.parameters, features=features),
        desired=1 / 3,
        rtol=1e-6,
    )
    merit: jax.Array = jnp.asarray([[1.0, 4.0, 9.0], [9.0, 4.0, 1.0]])
    target: jax.Array = ranker.targets(merit)
    np.testing.assert_allclose(actual=target.sum(axis=-1), desired=1, rtol=1e-6)
    np.testing.assert_array_equal(actual=jnp.argmax(target, axis=-1), desired=[0, 2])
    np.testing.assert_allclose(actual=ranker.targets(jnp.ones((2, 3))), desired=1 / 3, rtol=1e-6)
    np.testing.assert_allclose(actual=ranker.targets(jnp.zeros((2, 3))), desired=1 / 3, rtol=1e-6)
    assert float(ranker.loss(state.parameters, features, target)) > 0
    assert np.isfinite(np.asarray(ranker(parameters=state.parameters, features=features))).all()


def test_scorer_learns_conditional_candidate_preferences() -> None:
    ranker: Ranker = Ranker(inputs=3)
    state: RankingState = ranker.initialize(seed=7)
    initial: Parameters = state.parameters
    features: jax.Array = jnp.stack((jnp.eye(3), jnp.eye(3)[::-1]))
    merit: jax.Array = jnp.asarray([[1.0, 4.0, 9.0], [9.0, 4.0, 1.0]])
    target: jax.Array = ranker.targets(merit)
    before: float = float(ranker.loss(state.parameters, features, target))
    _index: int
    for _index in range(200):
        state = jax.jit(ranker.update)(state=state, features=features, merit=merit)
    after: float = float(ranker.loss(state.parameters, features, target))
    assert after < before - 0.2
    np.testing.assert_array_equal(
        actual=jnp.argmax(
            ranker.probabilities(parameters=state.parameters, features=features), axis=-1
        ),
        desired=[0, 2],
    )
    assert int(state.updates) == 200
    assert not np.array_equal(state.parameters.readout, initial.readout)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(state))
