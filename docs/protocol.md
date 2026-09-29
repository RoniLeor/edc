# Measurement protocol

## Accuracy

Residual ReLU MLP width32/depth32, original MSE, Adam0.001, activity rate0.23588,
alpha1, batch64, seeds0/1/2. MNIST/Fashion train55k, tuning5k (stratified split
seed42), official test10k. The best validation-accuracy checkpoint is selected
within each five- or ten-epoch run; test is evaluated after training. Hyperparameters
were fixed before these comparisons. These exploratory experiments are not a
preregistered benchmark, and the project has also examined other methods.

PC-ALM starts with zero duals. Ordinary GDI uses four neighbors, capacity512,
max age8 batches, strength0.5, temperature0.1, cosine confidence threshold0.5,
same-label filtering and original-ID self-exclusion. Elastic adds the mirrored
correction at strength0.05, fading to zero halfway through settling. No spline,
drift gate or energy selector. There are4,300 outer updates at five epochs and
8,600 at ten. No extra state-gradient evaluations are added by coupling.

The bundle has72 matched run summaries plus6 ordinary-GDI10/T16 controls.
Six elastic10/T16 measurements were reused from an earlier completed experiment;
the other30 ten-epoch comparisons were newly trained. All first-five validation
accuracy trajectories reproduced their five-epoch equivalents. All saved model
weights were finite, all settings/checkpoint selections were audited, and source
hashes were verified locally. The manifest hashes the exported evidence files;
it does not include datasets or all training checkpoints.

## Controlled gradients

Six validation-selected BP checkpoints from ten-epoch runs (three seeds per
dataset) are published with their original summaries and hashes. They are a
common reference, not each compared method's own final model. At each checkpoint:

1. Keep model weights fixed throughout the diagnostic.
2. Fill a fresh ordinary-GDI16 bank with the first16 batches of the training split.
   Store final duals and forward keys with the original GDI timing. No optimizer step.
3. Query the next four disjoint batches of64 training images. Query gradients
   and labels are never fed back into the bank or used to train model parameters.
4. Compare zero-initialized PC-ALM, ordinary GDI and elastic GDI at16/32/64 steps.
   All methods use identical weights, queries and the same fixed bank where applicable.
5. Compute exact BP on the same batch and weights. The main metric is cosine
   over all parameter entries, equivalent to flattening the entire gradient.
   Average the four batch cosines for each checkpoint; then report mean and
   sample SD across the three checkpoint means. Per-layer cosines are separate.

The frozen bank is deliberately held fixed to isolate the immediate update;
it does not replicate evolving memory during training. Four query batches are
a diagnostic sample, not a full dataset evaluation. Modest cosine gains here do
not demonstrate why accuracy improved or establish a generalization mechanism.
A zero-norm gradient returns cosine0 through the1e-30 denominator floor.

## Files and reproduction

`benchmarks/manifest.json` lists78 training summaries and six reference checkpoints.
`benchmarks/gradients.json` records54 method/budget/checkpoint combinations,
each averaged over four query batches. `experiments/gradients.py` recomputes them;
`experiments/figures.py` only renders saved measurements. No generated image model
is used: the architecture and all data figures are deterministic Matplotlib vectors.

The upstream comparison uses the same gradient metric definition, not identical
checkpoints or the paper's one-epoch/full60k training protocol. The core is tested
against a frozen upstream numerical oracle, including pre-dual credit timing.
