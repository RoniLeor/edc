#!/bin/sh
# Re-run the complete matched matrix into a fresh directory (72 training runs).
set -eu
output=${1:-results/reproduction}
for epochs in 5 10; do
  for dataset in mnist fashion; do
    for seed in 0 1 2; do
      for steps in 16 32 64; do
        for method in alm gdi; do
          elastic=0
          if [ "$method" = gdi ]; then elastic=0.05; fi
          uv run geodual --dataset "$dataset" --method "$method" --objective mse \
            --epochs "$epochs" --seed "$seed" --steps "$steps" --elastic "$elastic" \
            --output "$output/$dataset-$method-$steps-$epochs-$seed"
        done
      done
    done
  done
done
