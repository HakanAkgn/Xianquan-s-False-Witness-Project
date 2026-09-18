# A token-balanced test requiring no vocabulary or architecture change

The two-relation extension is a harder task and its early checkpoints did not reach high clean competence. To separate an architectural/data shift from the witness phenomenon, this additional exploratory test uses the original sequence format. It was designed after inspecting the 1,000/2,000-update original-style results, before evaluating the balanced test. All baseline seeds/checkpoints are retained.

Choose an irrelevant person j, distinct companies r,h with g(r)=g(h), and a company d outside the image of f. Initially let f(j)=r and g(d)=h. Swap these two literal symbols:

    f(j): r -> h
    g(d): h -> r.

The construction uses the shared 0..N-1 token alphabet of the symbolic framework: h and r can occupy a country-output position as country symbols. It does not directly apply to disjoint natural-language company/country vocabularies.

The model input has exactly the same symbols with exactly the same multiplicities and exactly the same length. The first change leaves g(f(j)) fixed; the second affects no person's answer because no person works at d. Exclude h,r,d from the queried intermediate as appropriate and choose j!=x. Therefore g o f is identical pointwise and f(x) is unchanged. The compensating country symbols h and r are chosen different from both the correct country and the suggested country, avoiding an answer-token-count manipulation. The g table itself changes at unused companies; company-country queries would NOT be invariant. Nor are every person's one-step employer answers invariant.

Compare four contexts: control, f-edit only, g-edit only, both edits. All preserve every person's country answer; the control/both pair additionally preserves the literal token histogram. Report compensation-only influence and the 2x2 interaction, rather than silently attributing every balanced-pair change solely to the employer edit. This control is token-balanced across roles, not separately within the employer and country tables. The two-relation construction tests a complementary form of typed attachment.

For k distinct irrelevant employees and k distinct unused companies, these disjoint exchanges commute and generate 2^k contexts. g remains unchanged at r,h and all active companies. Thus every node has identical composed answers and tokens. Four-bit cubes fit N=8, but conditioning on four unused companies necessarily reduces the set of active employers; report ordinary no-suggestion accuracy on those same contexts as well as i.i.d. clean accuracy.

Protocol: 4,096 one-swap pairs per original checkpoint, three hint conditions, full factorial controls; four-bit complete cubes with 1,024 underlying worlds, 16 contexts each, same three hint conditions, all seeds. Cube spectra, additive-singleton held-out prediction and oracle averaging follow COUNTERFACTUAL_CUBES.md. These are independently generated model experiments, not author-checkpoint results.
