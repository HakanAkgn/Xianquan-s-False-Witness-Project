# Frozen pilot design

This document is written before the training runs. This is an independent implementation, not an upstream-code reproduction. No original weights have been acquired.

Train seeds 11, 29, 47 for 4,000 updates each, reporting checkpoints 1,000, 2,000, 4,000 without selecting the most favorable. Evaluate 4,096 matched examples per setting. Main task: N=8, two-layer width-64, four-head, FF-128 pre-LN transformer, AdamW 0.003, weight decay 0.01, batch 128, norm clipping 1. Model defaults not explicitly recovered from the paper are listed in SOURCE_AUDIT.md.

1. Original-style equal-function edit: g o f unchanged for all people; f(j) changes from r to h, g(r)=g(h), j!=x. Does NOT preserve token counts.
2. Typed relation exchange: two relations, randomly queried; exchange (f_q(j),f_other(j))=(r,h) with (h,r). Both complete composed answer functions, all country-label counts, sequence length, all tokens and their global counts remain exactly fixed. Input positions are not preserved; query-relation stratification is mandatory.
3. Report incorrect hint, no hint, relevant-change and other-relation-at-query controls. Clean i.i.d. evaluations distinguish task competence from adversarial robustness. Invariance is measured only for two-step labels, never claimed for one-step labels whose underlying facts really change.
4. Witness-count sweeps at fixed N, context length, and paired token counts. Different counts use separately conditioned pairs, not one globally identical world across all counts; this limitation is explicit.
5. Cross second-layer Q/K/V replacements while keeping the recipient residual stream. Report factorial selection/payload interaction; this is an interventional decomposition, not by itself an out-of-sample prediction.
6. Derived identities: softmax mass and signed susceptibility; finite-difference/exhaustive tests only. These are not claimed novel general-transformer theorems.

No pretrained-model, Lean, verifier, or transfer-learning result is to be claimed unless actually run. Three seeds are a pilot, not a publication-strength independent replication count. Conditional world sampling differs from i.i.d. training and can affect absolute performance. Confidence intervals over examples and over seeds must be separated. Negative and null findings are retained.
