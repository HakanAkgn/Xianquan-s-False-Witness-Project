# Finalization audit: measured pilot, manuscript alignment, and remaining scope

This audit examines research commit `bda185e70ac45712463ed80f7f8e059bdcbf70ac`. It preserves its code and results rather than silently replacing their training recipe. The research branch and draft PR #1 already existed when this audit began. No claim is made here that their training was rerun in this audit.

## 1. What was independently verified now

The actual uploaded manuscript `false_witnesses_v2.pdf` is now available locally: 33 pages, SHA-256 `f0b220c21c1b5f23936923a11de69a35cf01e66eb53e313347833619b7f2f08b`. Appendix C on PDF page 20 was extracted and visually inspected. Historical statements in SOURCE_AUDIT.md that the earlier implementation session lacked the bytes should not be read as the current situation. The manuscript itself is not redistributed.

The public event archive was retrieved from GitHub and its exact Git blob identity verified: `943d13713f41cc5bd86f88b7bf174725eaedd791`. The new standard-library-only script `scripts/audit_public_events.py` independently decodes its three arrays, each containing four conditions and 4,096 paired examples. It recalculates following rates, paired changes, contingency tables, and fixed-model normal sampling intervals. It does not import the original evaluator.

| Seed | Control suggested-answer count | Balanced suggested-answer count | Net count change | Effect (percentage points) |
|---|---:|---:|---:|---:|
| 11 | 3594 | 3626 | 32 | 0.7812500000 |
| 29 | 2076 | 3695 | 1619 | 39.5263671875 |
| 47 | 10 | 13 | 3 | 0.0732421875 |

For seed 29, 1,625 examples gain the suggested wrong answer and six lose it. Of the 1,055 initially correct examples, 863 change to the suggested wrong answer; three change to another answer. Thus the event archive supports a substantial directed effect in this model, not only a difference between rounded aggregate percentages. Seeds 11 and 47 do not show a comparably large paired effect.

The six new audit tests pass. They check source identity, all three effects, contingency totals and marginals, net paired changes, the reported conditional interval, and rejection of altered input bytes. The earlier report states that 31 framework tests passed; those are a separate historical test run, not 31 tests rerun here. The full checkpoint inference, training histories, complete-cube outputs, and repair experiments were not reexecuted during this audit. Their results remain those reported in docs/RESULTS.md and the existing tables.

Reproduce this limited independent check from the repository root:

```bash
python scripts/audit_public_events.py --output results/finalization_event_replay.json
python -m unittest discover -s tests -p test_public_event_audit.py -v
```

The normal intervals are conditional on each fixed model and can be inaccurate for very sparse changes. They are retained to audit the published calculation, not to endorse them as exact intervals. Thousands of test examples do not substitute for more independent training seeds.

## 2. Concrete differences from the manuscript

The uploaded Appendix C.2 resolves several previously unknown settings. The pilot has the same stated width and parameter count, but parameter-count agreement does not establish implementation parity.

| Component | Uploaded manuscript, Appendix C.2 | Audited pilot source |
|---|---|---|
| Training task mixture | 25% one-hop, 75% two-hop | `fw/data.py:draw` uses `two = torch.rand(...) < 0.5`: 50% each |
| Layer initialization | PyTorch encoder clones the initial layer; parameters then train separately | `fw/model.py` independently constructs each `Block` in a ModuleList |
| Q/K/V initialization | PyTorch encoder attention defaults | A new `nn.Linear(width, 3*width)` uses Linear defaults, rather than the encoder MultiheadAttention projection initialization |
| Primary training seeds | 10 through 17, GPU generator seed 35600+s | CPU pilot seeds 11, 29, 47 and its own generator procedure |
| Ordinary attention mask | Causal lower-triangular mask, explicitly defined in C.4 | Causal mask agrees for the ordinary company-first setting |

In particular, the library MultiheadAttention initializes its packed input-projection weights with Xavier uniform and its input-projection biases to zero; a newly constructed ordinary Linear uses a different default weight/bias initialization. The source-level distinction does not imply any particular magnitude or direction of its effect on learning.

No stored numerical result was changed to make it appear to come from the manuscript's recipe. These pilot measurements remain valid as records of the implemented model, but they must not be treated as a verified reproduction of the original training population. A manuscript-aligned run should be a separately named experiment, retaining this pilot for comparison. Even matching every written setting would not establish bitwise parity with unavailable original checkpoints or random streams.

## 3. Where the associated repositories fit

The source search identifies two relevant public implementations:

* Guo et al., *How Do LLMs Perform Two-Hop Reasoning in Context?*: https://github.com/GuoTianYu2000/twohopIC ; the repository is linked in the paper, https://arxiv.org/html/2502.13913v2 . This is related prior work, not the original False Witnesses reproduction archive.
* Feng and Steinhardt, *How Do Language Models Bind Entities in Context?*: https://github.com/jiahai-feng/binding-iclr ; linked in https://arxiv.org/html/2310.17191v2 . This is a source for binding experiments, not an implementation of the present manuscript.

The uploaded manuscript specifies its own two-layer table model and refers to accompanying reproduction materials in Appendix H without giving their location in the supplied PDF. No authoritative False Witnesses repository or original checkpoint archive was located in this audit. The two repositories above must not be attributed as its upstream code.

## 4. What the current contribution supports

The existing branch provides three useful exploratory additions: token-balanced answer-preserving edits; complete finite cubes of interacting witnesses; and ordinary-data versus counterfactual-training comparisons. Its strongest individual balanced result survives preservation of the global literal-token histogram. That controls token counts, not token positions or all representational properties. Compensation-only and employee-only conditions remain necessary because the balanced intervention changes both tables.

The complete-cube report describes higher-order interactions and failure of a particular singleton-calibrated additive predictor in one seed. The repair table describes a large improvement for that seed, partial improvement for another, and a seed already robust under ordinary continuation. Neither result establishes a universal law or a universally successful repair. The effect of ignoring hints versus learning better evidence selection is not resolved by those accuracy numbers.

## 5. What has NOT been completed

The motivating discussion concerned graded similarity between irrelevant and relevant reasoning fragments, and the geometry of binding or retrieval representations. The current executed pilot does not establish those hypotheses. In particular:

* Multiple exact witnesses are not a graded approximate-match experiment.
* Token balance does not establish a shared internal mechanism with GSM-IC distractibility.
* Whole-layer Q/K/V factorial patches are not the localized crossed-payload or binding-ID interventions described in the manuscript and binding literature.
* The two-relation extension remains underlearned; it cannot support a claim about interference in an otherwise competent relation-binding model.
* No pretrained-language-model, natural-language mathematics, or formal-proof-search result has been established.

A focused next experiment should separate input similarity, selection-score compatibility, and downstream answer influence. Prespecify similarity independently of the observed error, keep the target answer and nuisance variables controlled, compare score-changing key perturbations with query-orthogonal controls, and cross selection changes with independently varied values. These are proposed experiments, not measured results in this repository.

## Final status

The research package is a reviewable independent pilot. Its central paired arithmetic has now been independently checked against exact public event bytes, and its training deviations from the available manuscript have been identified. It is not yet the completed graded-similarity / AI-for-mathematics extension requested in the discussion, nor a verified reproduction or confirmed general breakthrough. Keep the draft PR unmerged until the collaborators have reviewed these distinctions.
