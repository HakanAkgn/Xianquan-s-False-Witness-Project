# Baseline-first reproduction: executed status

**Exact reproduction is not established. No new extensions were run.**

## What was actually executed

The earlier pilot is preserved, not relabeled. A separate native-PyTorch baseline implementation was written against the recovered manuscript specification. Eight models, seeds 10--17, were trained for 2,000 updates with batch size 128 in the company-first, suggestion-last setting. Each model saw 256,000 fresh examples. The training mixture is 25% one-step and 75% two-step; suggestions are present in half of examples and correct with probability 0.75. The model has 69,576 parameters.

Evaluations were saved at 0, 500, 1,000, 1,500 and 2,000 updates on the same 4,096 diagnostic examples per condition across seeds. The diagnostic inputs are newly generated and are not the author's archived inputs. The original main group trains to 4,000 updates and spans both table orders and both primary suffix layouts; those later stages were not run here because the first-stage exact comparison is not established.

## Corrections relative to the previous pilot

| Component | Previous pilot | New baseline code |
|---|---|---|
| Training task mixture | 50/50 | 25/75, one-step/two-step |
| Layer initialization | Separately initialized blocks | Native encoder clones, equal initial tensors with separate storage |
| Attention initialization | Plain Linear QKV defaults | Native MultiheadAttention defaults |
| Main model seeds | 11, 29, 47 | 10--17 |
| Equal-function inputs | Other rows could use r | Every other row excludes both r and h |
| Reproduction claims | Independent pilot | Explicit author-replay checks; no fallback pass |

Core architecture, suffixes, exact GELU, all-parameter AdamW and clipping follow Appendix C. The first-layer-only 64-edge access masks are implemented and unit-tested but not claimed as a reproduced training experiment. These corrections do not reconstruct undocumented random-call order, original GPU streams, or missing checkpoints.

## Initial comparison: all eight models

Both columns concern suggestion-following rates, not clean-task accuracy. The paper columns are transcribed from Table 3; the diagnostic columns are freshly measured. Populations and training streams differ, so these differences are not a controlled experiment isolating a particular implementation detail.

| Seed | Paper control (%) | Paper witness (%) | Diagnostic control (%) | Diagnostic witness (%) | Diagnostic effect (points) |
|---|---:|---:|---:|---:|---:|
| 10 | 25.195 | 72.314 | 0.0488 | 0.0488 | 0.0000 |
| 11 | 0.000 | 0.000 | 44.6777 | 98.5352 | 53.8574 |
| 12 | 51.660 | 94.653 | 42.1875 | 98.7793 | 56.5918 |
| 13 | 59.204 | 95.703 | 0.1465 | 0.1953 | 0.0488 |
| 14 | 0.000 | 0.024 | 0.0488 | 0.0488 | 0.0000 |
| 15 | 28.564 | 79.736 | 30.5908 | 86.7920 | 56.2012 |
| 16 | 0.049 | 0.073 | 0.1465 | 0.1465 | 0.0000 |
| 17 | 54.810 | 95.898 | 2.8809 | 4.0771 | 1.1963 |

The diagnostic group mean is **15.0909% -> 36.0779%**, a **20.9869-point** effect. The paper reports **27.4% -> 54.8%**. No seed matches all four Table 3 measurements at the published precision. Model-level behavior differs sharply: for example, seed 11 is robust in the paper but vulnerable in this CPU diagnostic. This is not an exact reproduction, and similarity of the group-level phenomenon does not change that conclusion.

The corrected diagnostic is not an estimate of what changing only the task mixture or initialization does. Several mismatches were corrected together, and the author data/environment remain unavailable. The eight predefined seeds and all their outputs are retained; no seed was selected, resampled, or retuned to match the published mean.

## What is verified

All 37 new unit tests pass. They check parameter counts; cloned-but-independent layers; native attention bias initialization; the training mixture; all four suffix encodings in both orders; first-layer-only access masks; Appendix D.2 constraints; and refusal of missing, altered, malformed or diagnostic substitutes for author artifacts.

An independent execution audit reloaded all eight final checkpoints and exactly reproduced their saved logits and all 294,912 checked class predictions. A complete replay of seed 10's 2,000 training updates from its stored initialization and 256,000 ordered examples reproduces its final weights bitwise. Float64 native-versus-explicit-QKV forward checks also pass. These checks establish internal reproducibility of our diagnostic, not replication of the author runs.

## Why exact replay remains blocked

Appendix H describes an existing reproduction archive containing author inputs, checkpoints, outputs, streams and extraction code. It has not been found. The supplied GitHub project contains the earlier independent work, not that archive. The original manuscript was available through file-library text retrieval but not as local PDF bytes; no source-file hash or direct figure/table visual audit is claimed in this attempt.

The local execution environment is Python 3.13.5, PyTorch 2.10.0+cpu and NumPy 2.3.5. The manuscript reports PyTorch 2.6.0 and NVIDIA RTX A5000. CPU random streams seeded at 35600+s are not the corresponding CUDA streams. Reusing a seed does not establish identical initial weights, batches, or training trajectories. The exact constructor/random-call order and primary evaluation sampler stream have not been verified.

`reproduction/replay.py` rejects the supplied incomplete manifest. It requires genuine hash-checked author checkpoints, fixed inputs and reference logits, then tests every class and the Appendix H logit tolerance. Its implemented scope is only the first Figure 1/Table 3 group; even a pass cannot certify all remaining figures. The full-paper status therefore remains false, and the extension status remains blocked.

## Same-setting extensions: not yet executed

Once baseline replay is established, token balancing and four-witness combinations must be evaluated on the verified checkpoint tensors without changing the architecture or reinitializing. Retraining must use those same starting checkpoints and optimizer states, the corrected ordinary 25/75 mixture, and matched control arms. New test cohorts remain distinct from the original fixed evaluation cohorts. The extra typed-relation task changes the setting and cannot serve as a same-setting extension.

The old pilot's percentages have not been carried over. No claim about the token-balanced effect, nonadditivity or counterfactual repair under the paper's exact runs is made here.

## Source and evidence map

The source specification is `false_witnesses_v2.pdf`, Appendices C, D.2, H and J (Table 3). `configs/paper_table3.csv` is a transcription of reported, rounded values, not an original output array. `reproduction_results/diagnostic_comparison.csv` and `diagnostic_summary.json` contain our new measurements. `internal_replay.json` and `test_output.txt` contain executed validation results. `author_replay_status.json` records the blocked author replay. The full execution archive contains the checkpoint, optimizer, input, output and ordered training-stream files referred to by its included `diagnostic_artifact_hashes.json`.

Official numerical background: PyTorch 2.6 documentation, *Reproducibility* and *TransformerEncoder*. Its warnings about release/device reproducibility and cloned initialization motivate verification against archived tensors; they do not imply that the paper's reported findings are invalid.
