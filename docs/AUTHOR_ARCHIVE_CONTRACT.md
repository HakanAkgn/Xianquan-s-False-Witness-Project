# Author artifacts required for an exact reproduction

Source: `false_witnesses_v2.pdf`, Appendices C, D, E, G, H and J. Appendix H says accompanying materials contain evaluation inputs, checkpoints, output arrays, training streams, reconstruction code, and source-linked figure data. Those original materials have not been located in the accessible repository or file context.

## Obtain the existing archive, rather than generate a replacement

The necessary material is the author's code/release and version identifier, checkpoints for the reported seeds/updates, exact semantic training streams or generator implementation with environment/RNG states, fixed evaluation inputs, exact-GELU reference logits/predictions, and the compact figure-data file with its extraction script and hashes. The original GPU/PyTorch environment and initialization order are needed for a training replay. A checkpoint replay can be checked independently on CPU against the archived outputs.

A repository link containing only our reconstruction is not this archive. Guo et al.'s two-hop task code, cited by the paper, is related work and is not automatically the source of these experiments. The source PDF currently exists only as searchable file-library passages in this runtime; its exact bytes and a manuscript hash were not available for this attempt.

## Initial replay adapter: explicit file contract

`configs/author_replay_manifest.json` is deliberately incomplete. Fill it only after locating and inspecting genuine author material. Paths are relative to the manifest's directory. Each file entry requires a SHA-256 digest. Preserve the original archive unchanged and document the adapter/export script used to produce the following tensor-only files; do not substitute newly generated inputs or newly computed reference outputs.

`origin` identifies the author release and records how its origin was checked. Hashes verify file integrity, not authorship. A declaration in this manifest is not an independent provenance certificate.

The initial verifier expects eight models: seeds 10--17, company-first, suggestion-last, 2,000 updates. For every model supply its checkpoint, exact-GELU reference outputs, the state-dictionary container key (null for a raw tensor dictionary), and an optional explicit target-to-author parameter-name mapping. The loader does not guess parameter associations and rejects diagnostic/pilot checkpoint metadata. Only `torch.load(..., weights_only=True)` is used.

The common evaluation NPZ contains integer arrays of shape (4096,19): `one_absent`, `one_correct`, `one_incorrect`, `two_absent`, `two_correct`, `two_incorrect`, `pair_control`, and `pair_witness`. Natural cases share their underlying tables and question; the one-/two-step incorrect conditions share suggestions. The pair arrays must meet Appendix D.2 exactly, including both-company exclusion at every unedited row. If the original archive uses a different organization, adapt its existing arrays transparently rather than regenerate examples.

For each model the reference-output NPZ contains an array named `<case>_logits` of shape (4096,8) for every case above. The reference values must come from the author's canonical exact-GELU outputs. The verifier performs CPU float64 inference, checks every predicted class exactly, and uses a maximum absolute logit tolerance of 5e-5, matching the independently checked comparison tolerance reported in Appendix H. It also checks the initial model's Table 3 values to the table's published rounding precision. Do not relax tolerances, remove seeds, or alter data to obtain a pass.

## Scope of a pass

This verifier tests the initial Figure 1/Table 3 group only. It does not implement all remaining figure protocols. A pass leaves `full_paper_reproduced=false` and `extensions_allowed=false`. Extend the replay adapters to the remaining primary, access, information-control, reliability and activation-intervention arrays, then independently validate them. `configs/replication_stages.json` lists what is pending.

## Requirements for the extensions after baseline verification

Use the verified checkpoint tensors and the baseline encoder without reinitialization. Token-balanced pairs and four-witness cubes change only evaluation populations, which must remain separate from the original evaluation arrays. Counterfactual retraining must start from the verified 2,000-update checkpoint and optimizer, use the same baseline architecture and 25/75 ordinary-task mixture, and compare ordinary continuation, augmentation, and augmentation plus consistency at matched update/example budgets. Its augmented distribution is the explicit intervention, not a hidden baseline change. Retain no-/correct-/incorrect-hint and relevant-answer-change controls. The typed second-relation task is a different setting and is not part of this same-setting extension.

No new extension has been run in this baseline-first attempt. The older `fw.repair` defaults are not suitable for rerunning it unchanged because its ordinary sampler still uses the historical 50/50 mixture.
