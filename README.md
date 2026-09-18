# False Witnesses: reproduce the paper before extending it

**The paper has not yet been exactly reproduced. New extension runs are stopped.**

This branch separates the earlier, non-matching pilot from a manuscript-aligned baseline implementation and a strict author-artifact replay workflow. The first eight-model diagnostic was actually executed, but gives **15.0909% -> 36.0779%**, not the paper's **27.4% -> 54.8%**. A similar phenomenon is not an exact reproduction.

**[Current status and complete comparison](docs/PAPER_REPRODUCTION_STATUS.md)** · [Author-archive requirements](docs/AUTHOR_ARCHIVE_CONTRACT.md) · [Baseline-first sequence](configs/replication_stages.json) · [Saved diagnostic results](reproduction_results/)

## What changed

The new `reproduction/` package uses 75% two-step questions, the native PyTorch encoder with identically initialized but independently trained layer clones, native attention initialization, seeds 10--17, and the Appendix D.2 rule excluding BOTH compared companies from every other employee row. Exact GELU, causal masks, the primary suffix layouts, and the first-layer access masks are tested explicitly.

It does **not** recover the author's original GPU random streams by reusing the same seed integers. The original source, checkpoints, fixed evaluation inputs and reference outputs are still missing. The local diagnostic used PyTorch 2.10.0 CPU; the manuscript reports PyTorch 2.6.0 on RTX A5000. Constructor/random-call order also remains unverified.

## Commands

Run from the repository root. The recorded diagnostic environment is listed in `requirements-reproduction-tested.txt`; those versions are not asserted to be the author's complete environment.

```bash
python -m pytest tests_reproduction -q

# Expected to exit with code 2 until real author artifacts are supplied.
python -m reproduction.replay \
  --manifest configs/author_replay_manifest.json \
  --output reproduction_results/author_replay_status.json

# Optional fresh-data diagnostic, NOT an exact author replication.
# A new output directory is required; existing executions are never overwritten.
python -m reproduction.diagnostic \
  --output diagnostic_execution \
  --workers 4 --acknowledge-regenerated-data

# Independently replay all eight final outputs and seed 10's complete training stream.
python -m reproduction.audit_diagnostic \
  --root diagnostic_execution \
  --output reproduction_results/internal_replay.json
```

The replay verifier currently implements only the initial Figure 1/Table 3 comparison. Even a successful replay of that stage does not certify the entire manuscript, replay training, or unlock extensions. Remaining stages are enumerated explicitly, not silently declared complete. The supplied empty author manifest fails closed.

## Evidence retained

The eight CPU diagnostic models each received 2,000 updates with batch size 128. Evaluations at 0, 500, 1,000, 1,500 and 2,000 updates use common fixed diagnostic inputs. The execution archive retains all initialization/final checkpoints, final optimizer/RNG states, 2,048,000 ordered training examples, all saved output arrays and hashes. Git retains the compact comparison, summary and verification files; full plans, traces and artifact hashes are in the execution archive. These are **our diagnostic artifacts**, not author artifacts.

The new test suite has 37 passing tests. Replaying the eight final checkpoints reproduces all 294,912 checked classes and logits exactly. Replaying seed 10's 2,000 updates from its stored initial state and training stream reproduces its final weights bitwise. These establish internal reproducibility of this diagnostic, not agreement with the paper.

## Earlier pilot

The previous `fw/`, `results/` and existing experiment documents remain unchanged. Its README is preserved in [docs/PILOT_README.md](docs/PILOT_README.md). That 50/50-task, independently initialized three-seed pilot must not be used as the paper baseline. Its extensions have **not** been rerun on the new baseline.
