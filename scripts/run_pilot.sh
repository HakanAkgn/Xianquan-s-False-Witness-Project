#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Sequential by default, avoiding hidden parallel resource consumption.
for task in original typed; do
  for seed in 11 29 47; do
    python -m fw.run --task "$task" --seed "$seed" --threads 1 --output outputs
  done
done
for seed in 11 29 47; do
  for step in 1000 2000 4000; do
    python -m fw.balanced "outputs/original/seed_${seed}/model_${step}.pt" \
      --output "outputs/original/seed_${seed}/balanced_${step}"
  done
  for step in 2000 4000; do
    python -m fw.orbits "outputs/original/seed_${seed}/model_${step}.pt" \
      --output "outputs/original/seed_${seed}/orbit_${step}"
  done
  python -m fw.orbits "outputs/typed/seed_${seed}/model_4000.pt" \
    --output "outputs/typed/seed_${seed}/orbit_4000"
  for arm in iid augmentation consistency; do
    python -m fw.repair "outputs/original/seed_${seed}/model_2000.pt" \
      --output "outputs/repair/${arm}/seed_${seed}" --arm "$arm"
  done
done
python -m fw.summarize --outputs outputs --destination results
