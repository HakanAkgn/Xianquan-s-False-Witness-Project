# Baseline verification continued — 19 September 2026

**The trained-model results have not yet been exactly reproduced. No new extension experiments were run.** This continuation verifies parts of the paper that do not require missing author checkpoints: the arithmetic of Table 3/Figure 1 and the mathematical statements in Appendix F.1 and F.3.

Base inspected: `reproduction/paper-baseline-first`, commit `191df75c3de1146c3ce69565643759c83a1f1c67`. The previous pilot, baseline training code, diagnostic results, and replay gate are unchanged.

## 1. What was completed

| Baseline component | Result | What this does not establish |
|---|---|---|
| Published Table 3 arithmetic | All 32 rows and 128 rounded fields admit unique compatible integer counts; all 32 effects agree with their recovered marginals. | No checkpoint, individual prediction, paired transition count, or training run is recovered. |
| Figure 1 group mean | 27.435302734375% control; 54.8004150390625% witness; +27.3651123046875 percentage points. | This reconstructs the reported figure from the table; it is not independently generated model output. |
| Appendix F.1 | Matching-record identities checked on all maps for N=2,3; all 216 valid N=3 equal-function constructions pass. No N=2 construction satisfies the constraints. | This is an identity of the stated symbolic computation, not a claim that every transformer implements it. |
| Appendix F.3, small domains | 13,282 directly enumerated cases; exact agreement between direct Bayesian summation and the formulas; 745 zero-evidence cases identified. | These are newly specified verification cases, not the paper's archived 145,720-case suite. |
| Appendix F.3, main parameters | All 6,435 feasible sufficient-statistic combinations with at least one match pass at N=8, alpha=3/4. | This tests the query-averaged candidate solver, not actual model suggestion-following rates. |
| New verification tests | 26 tests pass in ordinary Python and under `python -O`. | The existing model-training test suites and full training runs were not rerun in this continuation. |

The source CSV copied for local verification matches its GitHub Git-blob SHA exactly: `fca89af9f803e2438db877f496c6add6e068eb34`. The report JSON records SHA-256 hashes for the source CSV and executed script. The paper was accessed through file-library text excerpts; original PDF bytes and a direct visual table audit were not available. Table reconstruction is conditional on the correctness of that transcription, the stated 4,096-example populations, and rounding to three decimal places.

## 2. Why exact counts can be recovered from a rounded table

For 4,096 examples, one additional counted answer changes a percentage by `100/4096 = 0.0244140625` percentage points. Table 3 prints three decimal places, so its rounding interval is only 0.001 percentage points wide. That interval can contain at most one admissible integer count.

The verifier finds that count using rational arithmetic, rather than guessing by floating-point rounding. Impossible, ambiguous, incomplete, duplicated, and arithmetically inconsistent rows are rejected.

For Figure 1's company-first, suggestion-last group at 2,000 updates, the recovered counts are:

| Seed | Control follows wrong suggestion | Witness follows wrong suggestion | Population in each condition |
|---|---:|---:|---:|
| 10 | 1,032 | 2,962 | 4,096 |
| 11 | 0 | 0 | 4,096 |
| 12 | 2,116 | 3,877 | 4,096 |
| 13 | 2,425 | 3,920 | 4,096 |
| 14 | 0 | 1 | 4,096 |
| 15 | 1,170 | 3,266 | 4,096 |
| 16 | 2 | 3 | 4,096 |
| 17 | 2,245 | 3,928 | 4,096 |

Equal weighting of these eight models gives the paper's rounded **27.4% → 54.8%**. This recovers exact marginal counts, not the identity of the examples that changed predictions. In particular, paired confidence intervals cannot be reconstructed from these marginals alone.

## 3. What the mathematical checks mean

### Matching a company can return its country without selecting the queried person

Appendix F.1 defines one-hot lookup matrices `F` and `G`. Their product `H = FG` gives every person's country. The verifier constructs those matrices explicitly. When a selection averages over people employed at the suggested company, the selected country is that company's country, independent of the queried person. All 1,563 nonempty matching selections in the exhaustive N=2,3 populations satisfy Equations (13) and (14).

The same verification checks all 216 valid N=3 Appendix D.2 constructions. Swapping the irrelevant employee between two companies in the same country leaves the complete person-country answer function unchanged. With N=2, assigning the two distinct companies the same country makes the company-country function constant, so different true and suggested countries cannot coexist.

These are confirmations of statements already in the paper, not new research findings.

### A query-averaged Bayesian solver can favor the suggested country

Appendix F.3 asks what an idealized solver would do with the tables and suggestion if it discarded the queried person's identity. Two independent calculations are compared: summation over every possible hidden queried person, and the paper's formula using country counts and the number of suggestion matches.

Every probability is represented as an exact fraction. For N=2 and N=3, all function pairs, all suggestions, and the explicitly listed reliability values are enumerated. This gives 13,282 cases, including 745 cases in which the evidence has zero probability and the conditional distribution is undefined. The mode and tie condition is checked in 7,702 applicable cases, including 764 ties. There are no mismatches.

At the paper's N=8 and alpha=3/4, the paper's likelihood ratio is R=21. The calculation depends only on the eight country counts and the number of matching employees. Country-label symmetry lets the suggested country be fixed to label zero. All 6,435 feasible combinations of these sufficient statistics with at least one match are realized by explicit tables and checked by direct summation. Every case has a unique suggested-country mode; the smallest unnormalized lead over any other country is 14.

This does **not** predict that every trained model follows the suggestion. The paper itself distinguishes the candidate calculation from the actual learned models. The model-comparison measurements in Appendix F.3, such as 61.77% mode agreement, still require the author's actual outputs.

## 4. What remains missing for the requested exact reproduction

The existing eight-model CPU diagnostic remains **15.0909% → 36.0779%**, not the paper's result. Nothing in this continuation relabels it as successful reproduction. Matching the mathematical identity or reconstructing a table cannot repair a mismatch in the trained models.

Appendix H describes an accompanying archive with original checkpoints, fixed evaluation inputs, output arrays, training streams, and reconstruction/extraction code. That archive is still not supplied by this repository; its release list is empty. The earlier CPU diagnostic archive is not a substitute for the author archive.

The minimal material for the first exact inference stage is the eight company-first, suggestion-last checkpoints at update 2,000, their original evaluation arrays, and canonical exact-GELU reference outputs. Exact training replay additionally needs the original source, initial states, ordered random/input streams, optimizer state where applicable, and environment details. Later figures require their corresponding archived groups as well.

The existing `reproduction.replay` checks remain authoritative for checkpoint replay. This new script does not change the replay gate, claim full-paper completion, or authorize extensions. No typed task or new training distribution has been substituted.

## 5. Files and commands

New files:

- `reproduction/paper_checks.py`: exact-arithmetic table and Appendix F checks; standard library only.
- `tests/test_paper_checks.py`: 26 validation tests, including failure cases.
- `reproduction_results/paper_checks_20260919.json`: all recovered marginal counts, enumeration summaries, scope flags, and hashes.
- `reproduction_results/paper_checks_tests_20260919.txt`: the executed new-test log.

From the repository root:

```bash
python -m unittest discover -s tests -p test_paper_checks.py -v
python -O -m unittest discover -s tests -p test_paper_checks.py
python -m reproduction.paper_checks \
  --table configs/paper_table3.csv \
  --output reproduction_results/paper_checks_local_rerun.json
```

Use a new output filename for another run: the checker refuses to overwrite an existing result. No third-party package, paid service, email access, or model download is needed for these checks.

## Sources

1. *False Witnesses: When Irrelevant Facts Change Transformer Answers*, `false_witnesses_v2.pdf`: Appendix F.1, Equations (13)–(14); Appendix F.3, Equations (15)–(17); Appendix H; Appendix J, Table 3. These are the existing paper's results being checked.
2. Repository base commit `191df75c3de1146c3ce69565643759c83a1f1c67`: `configs/paper_table3.csv`, `README.md`, and `reproduction/replay.py`.
3. New execution output: `reproduction_results/paper_checks_20260919.json` and the accompanying test log. Checkpoint replay and full-paper reproduction remain false.
