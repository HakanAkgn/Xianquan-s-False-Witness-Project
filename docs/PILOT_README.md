# False Witness: token-balanced counterfactual experiments

**Independent implementation, executed pilot, and reproducibility package.** This repository extends the random-function framework described in *False Witnesses: When Irrelevant Facts Change Transformer Answers*. The original author repository and checkpoints were not found; this is **not** a fork or verified reproduction. See the [source audit](docs/SOURCE_AUDIT.md) for recovered settings and unverified implementation choices.

## Read the results

[Executed research report](docs/RESULTS.md) · [All numerical summaries](results/) · [Original-format balanced construction](docs/BALANCED_ORIGINAL.md) · [Complete-cube analysis and proofs](docs/COUNTERFACTUAL_CUBES.md) · [Repair protocol](docs/REPAIR_PROTOCOL.md)

At 2,000 updates, the token-balanced witness raises wrong-suggestion following by **0.7813, 39.5264 and 0.0732 percentage points** across seeds 11,29,47. IID no-hint accuracy is **99.9756%, 99.8291% and 100%**, respectively. The effect is heterogeneous, not an eight-model replication of the original caption.

Four independently movable witnesses form 16 contexts with identical tokens and country answers. In seed 29, **30.1873%** of nonconstant wrong-versus-correct logit-margin energy is higher-order; singleton-calibrated additive predictions perform worse than a constant baseline. This is an empirical interaction diagnostic, not a new Fourier theorem or a complete causal model.

With matched extra updates and supervised contexts, one-witness counterfactual augmentation raises seed 29's incorrect-hint four-witness cube robustness from **6.0547% to 99.6094%**, compared with ordinary-data continuation. Other seeds and no-hint conditions show important differences and tradeoffs. Adding a consistency penalty is not uniformly better. The typed two-relation extension remains underlearned and is not evidence of high-competence binding failure.

## Install and test

Tested: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0. The environment's installed CPU wheel reports a `+cpu` suffix. For the matching package release versions:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.10.0' 'numpy==2.3.5' 'scipy==1.17.0'
python -m pip install -e '.[test]'
python -m pytest -q
```

The completed suite has **31 passing tests**. Other supported dependency versions may change numerical trajectories; exact percentages are not guaranteed across hardware or versions.

## Reproduce

```bash
# Entire fixed seed/checkpoint grid, controls, repair arms and summary tables:
bash scripts/run_pilot.sh

# Or run one original-format model and its exact-token controls:
python -m fw.run --task original --seed 29 --output outputs
python -m fw.balanced outputs/original/seed_29/model_2000.pt \
  --output outputs/original/seed_29/balanced_2000
python -m fw.orbits outputs/original/seed_29/model_2000.pt \
  --output outputs/original/seed_29/orbit_2000

# Counterfactual training; use iid/augmentation/consistency as separate arms:
python -m fw.repair outputs/original/seed_29/model_2000.pt \
  --arm augmentation --output outputs/repair/augmentation/seed_29

# Aggregate after the entire grid exists; missing required runs fail explicitly:
python -m fw.summarize --outputs outputs --destination results
# A fresh clone stores the complete CSV grid in a compressed data archive:
python -m zipfile -e results/tables.zip results
python -m fw.plot --results results --destination results/figures
```

Defaults use CPU, one thread, no external model API and no paid service. Existing output paths are overwritten when rerun; copy an execution before changing its configuration. `torch.load(..., weights_only=False)` reads optimizer-containing local checkpoints: **load only trusted checkpoints**. Do not run this loader on an untrusted download.

## Files and evidence

`fw/data.py` provides random tables, original and typed interventions, exact token checks, and relevant changes. `fw/model.py` is the patchable two-layer transformer. `fw/evaluate.py` evaluates paired outcomes and all-head second-layer Q/K/V interventions. `fw/orbits.py` constructs complete cubes, decomposes response variance, and tests singleton-to-combination predictions. `fw/repair.py` performs compute-matched training comparisons.

`results/tables.zip` contains the complete CSV grid and input manifest, retaining every seed/checkpoint/arm, including small effects and underlearned models. `balanced_2000_event_audit.json` stores compact per-example events for the main balanced comparison; `fw.evaluate.unpack_outcomes` decodes them. Codes preserve correct/suggested-wrong/other status, not the identity of other answers. Full categorical outputs, probability/margin arrays, optimizer checkpoints and execution logs are retained in the accompanying execution archive and regenerated under `outputs/`; large binaries are not required in this Git repository.

The central balanced edit preserves **all person-country answers**, not every possible query about changed tables. Complete-cube robustness applies to the tested finite orbit, not all logically equivalent prompts. This pilot does not establish transfer to pretrained language models, mathematics, formal proofs, or a generally effective learned provenance mechanism. Scientific priority has not been exhaustively assessed.
