# Complete counterfactual cubes: from individual witnesses to interactions

## Construction

Fix a query (x,q), two relation tables f_0,f_1, and a company-country map g. Choose k distinct people j_1,...,j_k different from x. At each selected person choose two distinct companies r,h with g(r)=g(h), and set (f_q(j),f_{1-q}(j))=(r,h). Let tau_j exchange the two entries. These swaps commute, are involutions, and act freely on the selected assignments. They generate a cube G=(Z/2Z)^k with 2^k distinct contexts.

For every z in G, every person i, and both relations r,

    g(f_r^z(i)) = g(f_r^0(i)).

The query's intermediate company f_q(x), suggestion, query token, all country answers, sequence length, and **the complete token multiset** also stay fixed. Thus witness-count curves across this cube do not add any suggestion mentions or change the background world. The earlier separate-count paired experiment does not have this stronger across-count guarantee.

The construction requires g-equivalent values, a second relation, and distinct nonquery rows. It is a new task extension, not a claim that original checkpoints accept the new vocabulary/length. Randomly query either relation and report strata to check position/role confounding.

## Exact decomposition

Let D(z)=logit_wrong(z)-logit_correct(z) be a scalar model response. Use logits, not only probabilities, to avoid attributing sigmoid saturation to an internal interaction. Define

    D_hat(S) = 2^(-k) sum_z D(z) (-1)^(sum_{j in S} z_j).

Standard Walsh orthogonality gives

    Var_G D = sum_{S nonempty} D_hat(S)^2.

Degree-one terms measure additive sensitivity. Terms of degree at least two measure interactions on this cube. The fraction of nonconstant energy at degree >=2 is reported together with its absolute variance; a large fraction of vanishing variance is not an important behavioral effect.

For an edge flip e_j,

    E_G [D(z)-D(z xor e_j)]^2 = 4 sum_{S containing j} D_hat(S)^2.

Consequently,

    Var_G D <= (1/4) sum_j E_G [D(z)-D(z xor e_j)]^2 <= k Var_G D.

**Proof.** Expand in the orthonormal Walsh basis. Flipping j reverses the sign of precisely those basis functions containing j, producing a factor two. Squaring and averaging removes cross terms. Summing over j weights each nonconstant coefficient by |S|, which lies between 1 and k. This establishes both bounds.

This is the standard discrete-cube Poincare/Parseval argument applied to answer-preserving contexts, not a newly discovered general mathematical inequality. In particular, one measured edge at a single starting context does NOT bound the full variance. The function F(z1,z2)=z1*z2 has zero change from (0,0) under either isolated edit and a nonzero change when both edits are combined.

## Predictive test

Calibrate an additive response using only the no-witness context and the k singleton-witness contexts:

    D_pred(z) = D(0) + sum_j z_j [D(e_j)-D(0)].

Evaluate RMSE on the remaining 2^k-k-1 contexts, which were not used for calibration. Compare to the constant D(0) baseline. This is explicitly **within-world singleton-to-combination prediction**: it is not a predictor of new worlds without calibration, and it is not a fitted causal transformer model.

## Invariant projection and its limits

Let p(z) be the model probability vector. The finite group average p_bar=E_G p(z) is unchanged by any cube swap. It is the least-squares projection onto responses constant on the cube. For a fixed true answer y, Jensen's inequality gives

    -log p_bar(y) <= E_G[-log p(z)(y)].

The guarantee concerns mean cross entropy under the uniform cube, not accuracy, calibration for another distribution, or improvement over the clean representative. The implementation reports accuracy and loss separately. The orbit generator is an external symbolic oracle; this baseline must not be presented as an end-to-end learned defense. A practical learned remedy would minimize edge consistency on training cubes while also supervising relevant changes, then be tested on unseen cubes/relations.

## Pilot protocol

Use the already frozen baseline seeds and checkpoints. Add k=4, 1,024 independent underlying worlds, all 16 assignments per world, and incorrect/absent/correct suggestion conditions. No training or checkpoint selection uses cube outcomes. Report every seed, query-relation strata, robust accuracy over the finite orbit, ordinary mean accuracy, coefficient energies, held-out additive RMSE, and the projection's loss/accuracy. The cube experiment is an exploratory extension designed after the first 1,000-update learning diagnostics were inspected, but before any cube response was evaluated. It is not retrospectively called preregistered.

## Attribution

Walsh analysis, orthogonality, Parseval and influence identities are standard; see Ryan O'Donnell, *Analysis of Boolean Functions*, Cambridge University Press (2014), updated author version https://arxiv.org/abs/2105.10386 . Group-orbit averaging for data augmentation is also established: Shuxiao Chen, Edgar Dobriban, Jane H. Lee, *A Group-Theoretic Framework for Data Augmentation*, JMLR 21(245):1-71 (2020), https://jmlr.org/papers/v21/20-163.html . The proposed research contribution is the token- and answer-preserving relational cube plus its empirical application, not these underlying identities. No exhaustive priority claim is made.
