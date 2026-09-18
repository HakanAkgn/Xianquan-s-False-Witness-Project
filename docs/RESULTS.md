# Executed research results: token-balanced false witnesses

**Status: independent reconstruction and three-seed exploratory pilot, not an author-code replication or a confirmed general breakthrough.** Executed on CPU on 18 September 2026. The supplied repository was empty. The manuscript framework was recovered from stored document passages, but authoritative upstream source, exact manuscript bytes, and author checkpoints were not located. Reconstruction assumptions are itemized in [SOURCE_AUDIT.md](SOURCE_AUDIT.md).

Six models were trained from scratch (three original-format and three two-relation models), with all 18 requested checkpoints retained. Nine continued-training runs compare three repair arms at identical supervised-context and update budgets. Eighteen complete four-bit cube evaluations cover the original checkpoints, final typed models and repair arms. Each cube evaluation includes three hint conditions. No seed was dropped because its effect was small or its competence was low.

## 1. A stronger, original-format control

Change an irrelevant employee's company from r to h with g(r)=g(h). Compensate the token change by exchanging h to r in the country table at an unused company d. No employee works at d, including after the swap. Then every person's country g(f(i)), the queried employer f(x), the query, the suggestion, input length and complete literal token histogram remain fixed. This requires no additional vocabulary or relation tokens.

This construction uses the shared symbolic token alphabet; it does not directly extend to disjoint company/country vocabularies. This is global token balance, not balance separately within each relation table. The unused company's country changes, so standalone company-country answers are not invariant. Other employees' one-step employer answers also need not stay fixed. The paired effect cannot be attributed exclusively to the employee edit without its compensation controls; all four combinations of the two edits are evaluated.

At **2,000 training updates**, with **4,096 paired examples per model**:

| Seed | IID no-hint accuracy (%) | Wrong-suggestion following: control (%) | Balanced witness (%) | Change (percentage points) | Within-model paired 95% interval (points) |
|---|---:|---:|---:|---:|---:|
| 11 | 99.9756 | 87.7441 | 88.5254 | 0.7812 | [0.2196, 1.3429] |
| 29 | 99.8291 | 50.6836 | 90.2100 | 39.5264 | [38.0198, 41.0330] |
| 47 | 100.0000 | 0.2441 | 0.3174 | 0.0732 | [-0.0337, 0.1802] |

The 39.5264-point effect in seed 29 survives exact token balancing while IID no-hint accuracy is 99.8291%. On the balanced cohort itself, no-hint control accuracy is 99.7803%. The compensation-only change in the wrong-minus-correct logit margin is -0.0743, the employee-only change is +1.7147, their interaction is +0.0709, and the balanced total is +1.7113. Thus the compensation is not inert, but its isolated mean effect does not explain the positive balanced effect.

Seed 11 has a much smaller paired effect despite already following the incorrect suggestion frequently in both conditions. Seed 47 is largely robust. The mean balanced effect across the three seeds is 13.4603 points, but its descriptive Student-t 95% interval is [-42.6233, 69.5438] points. These seeds do not establish a precise population effect. The within-model intervals in the table are normal paired sampling intervals conditional on each fixed model, not independent-training replication intervals.

The compact public event record `results/balanced_2000_event_audit.json` reproduces the paired contingency counts exactly. Full eight-class categorical predictions are retained in the execution archive. The original caption's 27.4% to 54.8% is not substituted for any of these new measurements.

## 2. Complete, fixed-token counterfactual cubes

Four disjoint swaps generate 16 contexts for each underlying world. Within a cube every context has the same input-token multiset and every person's country answer is identical. Unlike independently generated count-specific cohorts, all witness counts from zero through four are evaluated on the same underlying worlds. Evaluation uses 1,024 worlds (16,384 contexts) per model and hint condition.

Let D(z) be the wrong-minus-correct logit margin on the four-bit cube. Standard Walsh analysis decomposes its variance into additive and higher-order components. The degree-at-least-two energy fraction below uses logits, not only softmax probabilities, and is reported with absolute variance. It does not identify a unique internal mechanism: normalization and nonlinear processing can also produce such interactions.

The additive predictor is calibrated on only five contexts per world: zero witnesses and four singletons. Its error is evaluated on the remaining eleven combinations. This is a within-world intervention-combination test, not generalization to new worlds without calibration.

| Seed | Margin variance | Higher-order share (%) | Held-out additive RMSE | Constant-baseline RMSE | Correct on all 16 contexts (%) |
|---|---:|---:|---:|---:|---:|
| 11 | 0.101884 | 7.9541 | 0.321225 | 0.642455 | 0.7812 |
| 29 | 0.304821 | 30.1873 | 2.201845 | 1.977894 | 0.4883 |
| 47 | 0.307626 | 12.7581 | 0.838899 | 1.119303 | 97.4609 |

In seed 29, 30.1873% of nonconstant margin energy lies above degree one. The singleton-calibrated additive predictor has RMSE 2.201845, worse than the constant baseline's 1.977894. This rejects this particular additive account on these counterfactual cubes; it does not reject every possible mechanistic model. In seeds 11 and 47, the additive predictor improves on the constant baseline.

Wrong-suggestion following across zero through four witnesses is [83.5938, 95.9229, 97.4772, 98.1201, 98.5352]% in seed 29. In seed 11 it is [86.8164, 88.9160, 89.4694, 88.6963, 86.7188]%, which is nonmonotonic. A universal monotone multiplicity law is therefore not supported by this pilot. Conditioning on four unused companies changes the evaluation distribution; these rates should not be directly compared with one-swap rates as though only count differed between the cohorts.

Finite-orbit robust accuracy means correctness on every one of these 16 contexts. It is not a certificate over all answer-preserving rewrites. The oracle orbit-average baseline and its Jensen loss inequality are included as diagnostics, not as a practical learned defense. Walsh/Parseval/Poincare identities and group averaging are established mathematics, not claimed as new theorems; see [COUNTERFACTUAL_CUBES.md](COUNTERFACTUAL_CUBES.md).

## 3. Actual, compute-matched training interventions

All arms start from the corresponding original model's 2,000-update checkpoint and optimizer state. Each receives 500 additional updates with 512 supervised contexts per update. The ordinary-data arm continues IID training. Counterfactual augmentation includes incorrect-hint controls, their one-witness token-balanced counterparts, and relevant query changes with changed labels, plus ordinary examples. Consistency uses exactly the same augmented data plus a weight-one Jensen-Shannon penalty. The extra penalty adds a small arithmetic cost; FLOPs are not asserted to be exactly identical.

Training constructs one witness. Testing includes complete four-witness cubes, independently generated test examples, IID no-/correct-/incorrect-hint accuracy, and relevant changes. Inference uses the normal transformer only; no symbolic oracle is supplied at test time.

| Seed | Arm | IID no-hint accuracy (%) | IID correct-hint accuracy (%) | Relevant-change accuracy (%) | Incorrect-hint cube robust accuracy (%) | No-hint cube robust accuracy (%) |
|---|---|---:|---:|---:|---:|---:|
| 11 | iid | 99.9268 | 99.8291 | 99.9268 | 7.6172 | 99.6094 |
| 11 | augmentation | 99.3164 | 98.9502 | 99.1211 | 43.2617 | 96.3867 |
| 11 | consistency | 99.0479 | 97.2412 | 99.0479 | 44.6289 | 98.0469 |
| 29 | iid | 100.0000 | 99.5850 | 100.0000 | 6.0547 | 99.6094 |
| 29 | augmentation | 99.8291 | 99.9756 | 99.9268 | 99.6094 | 98.8281 |
| 29 | consistency | 99.7559 | 99.9756 | 99.8535 | 99.7070 | 96.2891 |
| 47 | iid | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 47 | augmentation | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 47 | consistency | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |

For seed 29, counterfactual augmentation raises incorrect-hint finite-cube robust accuracy from 6.0547% under matched IID continuation to 99.6094%, while IID no-hint accuracy is 99.8291% and relevant-change accuracy is 99.9268%. The consistency arm reaches 99.7070% on the incorrect-hint cube, but its no-hint cube robustness falls to 96.2891%, compared with 98.8281% under augmentation and 99.6094% under IID continuation.

Seed 11 improves substantially but remains imperfect: augmentation and consistency reach 43.2617% and 44.6289% incorrect-hint cube robustness, respectively, versus 7.6172% for IID continuation. Both incur competence/robustness costs in other conditions. Seed 47 reaches 100% finite-cube robustness in all three arms; ordinary training already suffices there. These results support augmentation as a useful pilot intervention, not a universal fix. They do not establish a distinct, consistent advantage for the JS penalty over augmentation alone. Relevant-change controls rule out simply producing a single constant answer, but do not by themselves prove a learned provenance mechanism.

## 4. An important negative result

The extended two-relation task reaches only 40.6494%, 41.6016% and 40.3076% IID no-hint accuracy at 4,000 updates for seeds 11,29,47. Its generators, paired tests and complete cubes are implemented and executed, but the competence requirement failed. Its distractor effects are not evidence that an otherwise proficient model loses relation binding. All outputs are retained rather than presenting only the largest effects.

Pretrained-language-model transfer, genuinely false claims against fixed authoritative facts, mathematical applicability tasks, formal proof search, new relation-type transfer, and a predictive Q/K/V theory have not been established. Existing Q/K/V factorial probes patch all positions and heads in the second layer; they must not be described as localized mediation or a complete causal explanation.

## 5. Validation and reproducibility

The test suite passes 31 tests, covering answer/token invariants, exact witness counts, encoder layouts, relevant-change sensitivity, Q/K/V identity patches, compression round trips, Fourier identities and invalid-input guards. Across completed cube evaluations, the largest Parseval residual is 2.84e-14. The finite-difference gradient check has maximum error 2.23e-11 over 1,000 random trials. These validate implementation identities, not scientific novelty or model generalization.

`results/tables.zip` contains all seed/checkpoint CSV summary tables and input hashes. The `results/` directory also contains raw paired event decisions for the central comparison, numerical checks, and the repair summary. The execution archive additionally contains all checkpoints, full categorical predictions, cube probability/margin arrays, training metadata, training traces and logs. Plan hashes are content-integrity records, not external preregistration. The exploratory addenda state when they were designed relative to inspected results.

To reproduce with the tested environment, use the repository README and `bash scripts/run_pilot.sh`. All computations default to CPU. No large-model API, paid compute service or external proof engine is required.

## Implication for the manuscript

The strongest supported addition is a sequence of increasingly demanding tests: a token-balanced effect at high clean competence, a complete-cube test exposing nonadditive response, and a supervised counterfactual intervention assessed against ordinary training and relevant-change controls. The scientific framing should concern counterfactual robustness within an answer-equivalence class, with explicit limitations. The original author implementation and more independent training seeds are still needed before treating these findings as a robust extension of the original paper rather than an independent pilot.
