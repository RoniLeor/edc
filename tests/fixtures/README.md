# Reference oracle

`reference.npz` was generated directly by the public SakanaAI/pc-alm implementation at commit
`660747f61a8a7e547c0ecd2c48c8883380a7d1f6`, without changing its source.

Settings: parameter seed 7, input seed 8, input shape (2,5), one-hot targets eye(2), depth 4,
width 3, input dimension 5, output dimension 2, ReLU, float32, state_lr=0.2, rho=1, alpha=1,
budget=8, inner_steps=1, weight_credit_timing='pre_dual_energy'.

Arrays: first/hidden/readout parameters, x/y, forward hidden states, final settled states,
pre-final-dual weight-credit multipliers, and each parameter group's PC-ALM gradient.
Upstream `init_params`, `forward`, `run_pcalm`, and `method_grad` produced these arrays;
only the hidden-layer lists were stacked to match EDC's vectorized representation.

The test compares production outputs against this frozen oracle. Floating-point tolerances
account for different XLA fusion and reduction order; no expected values are generated from
EDC itself. See THIRD_PARTY_LICENSE at the project root for the upstream MIT notice.
