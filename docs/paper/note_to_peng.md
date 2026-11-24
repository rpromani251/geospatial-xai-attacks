Hi Peng,

Attached is the current draft of the geography paper, **"Geographic awareness is no defence: scaffold attacks hide a
racial price penalty from LIME, KernelSHAP and GeoShapley."** Since the last version it covers two cities, ten seeds
for the main King County results, and what a typical audit report would conclude. Every number in the PDF is generated
from the result files, and an independent recomputation of 952 printed numbers and rule outcomes agrees with it.

A longer project record (all results with numbers, a dated account of everything tried since the start of the
project, corrected claims, and the decisions I need from you) is in the shared doc.

**Main findings**

- **All three explainers are fooled, LIME completely.** A scaffold with a detector trained on each explainer's own
  queries hides a penalty that halves predicted prices in tracts where Black and Latino residents exceed a quarter of
  the population. LIME shows 0.01 of the penalty while 97.8% of penalised homes still pay it.
  KernelSHAP and GeoShapley show 0.70 and 0.74 of it while
  89–90% of penalised homes still pay.
- **It replicates in Cook County, Illinois** (5 of 6 explainer–design classes the same; the
  exception is a noise artefact explained in the paper). There nearly half the homes are penalised, so even the
  unhidden explanation shows a smaller penalty (−0.15 against −0.24 in
  log10 price for the same penalty): a Shapley value compares a home with a reference population that is itself
  penalised.
- **Geographic awareness buys little, and can cost.** GeoShapley keeps about 4 points more than KernelSHAP in King
  County's feature design, nothing detectable in its location design, and **less** than KernelSHAP in Cook County's
  location design (−0.08).
- **Two propositions explain why** (Section 4.6, proofs in Appendix E): an exact split of the share kept, which
  predicts the result at a smaller penalty to within 0.009, and a routing floor, valid for any detector,
  set by the dependence between features that an explainer's queries break.
- **What an audit would conclude.** Under the scaffold, the global importance plot misses the protected attribute in
  all 10 seeds with LIME, 7 with KernelSHAP and
  4 with GeoShapley. A test restricted to the penalised homes still catches it with the Shapley
  explainers but not with LIME.
- **Robustness.** The same results with random forests and neural nets, a continuous penalty, a smaller penalty, other
  Shapley backgrounds and a background of homes the attacker never saw (which, if anything, helps the attacker).
  Switching LIME to its Gaussian sampler recovers part of the penalty. A local Shapley background resists the attack
  but sees less of a neighbourhood-wide penalty to begin with.

**Where I'd value your view**

1. Venue, and how much of the theory stays in the main text.
2. Whether Cook County gets its own section or stays one table and a section in Results.
3. Whether the tests added after the plan (local and unseen backgrounds, continuous penalty, audit checks) stay in the
   main text, labelled as such, or move to an exploratory section.
4. The outcome check, which clears its bar narrowly (AUC 0.81 against 0.80).
5. Literature to position against, and authorship.

**Not in this draft:** stronger attacker variants, attackers that adapt to defences, and a full citation check.

The routing floor also links to the theory project I'd like to discuss with you: detecting hidden bias when a model
only answers honestly near the data.

Best,
Robert
