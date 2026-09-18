# Compute-matched repair experiment

This is an exploratory follow-up designed after seeing original-style and first balanced-pair effects, but before training any repair arm. Keep all three original seeds 11,29,47 and start every arm from each seed's fixed 2,000-update checkpoint plus optimizer state. Do not select by effect or competence.

Run 500 additional updates, 512 supervised contexts per update, otherwise original AdamW, clipping and learning rate. Arms:

* **iid:** 512 new ordinary training contexts, same original hint statistics.
* **augmentation:** 128 ordinary contexts, 128 balanced controls with an incorrect suggestion, their 128 token-balanced witness contexts, and 128 relevant changes that set the actual queried employer to the suggestion. Equal average cross entropy over all 512 contexts.
* **consistency:** identical contexts and cross entropy as augmentation, plus JS divergence of the two 128-example paired answer distributions, weight 1.

The first 128 iid contexts, model initialization, and optimizer starting state are shared across arms within a seed. Augmentation and consistency also share identical counterfactual draws. All arms receive 512 supervised contexts and the same update count; consistency has a small additional arithmetic cost. This does not imply identical FLOPs to floating-point precision. The counterfactual arms deliberately alter the training distribution; the augmentation arm is necessary to distinguish that effect from the consistency penalty.

Train with one witness. Evaluate untouched held-out one-witness examples plus complete four-witness cubes (all combinations). Thus the multi-witness test is a transfer in number/combinations of interventions, not a claim of transfer to new relation types or natural-language tasks. Also report no-hint and correct-hint i.i.d. accuracy, relevant-change accuracy, original-style pairs and no-hint/correct-hint cube outcomes. Invariance alone cannot pass the relevant-change test. All counterfactual generation uses ground-truth synthetic functions during training; inference on the trained model uses only its normal token input.

The augmentation and consistency arms may perform equally. A null benefit of JS must be reported, not hidden. Three seeds constitute a pilot. No learning-rate or weight search will be performed for this report.
