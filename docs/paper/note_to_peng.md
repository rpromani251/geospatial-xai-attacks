Hi Peng,

Attached is draft v1 of the geography paper, working title **"Geographic awareness is no defence: scaffold attacks
hide a racial price penalty from LIME, KernelSHAP and GeoShapley."** All results are from King County. Every number is
generated from the result files, and an independent recomputation of the printed numbers and the pre-specified rule
outcomes agrees with the PDF.

**Main findings**

- **All three explainers are fooled, LIME completely.**
  - A scaffold with a detector trained on each explainer's own queries hides a penalty that halves predicted prices
    in tracts where Black and Latino residents exceed a quarter of the population.
  - LIME shows 2% of the penalty while 97% of penalised homes still pay it.
  - KernelSHAP and GeoShapley show 67% and 72% of it, while 86–88% of penalised homes still pay.
- **What is hidden from the Shapley explainers is limited.** The scaffold hides the size of the penalty and its
  global importance rank. On the penalised homes the trigger stays among the top three attributions. Hiding more
  costs the attacker bias delivered.
- **Geographic awareness buys little.** GeoShapley keeps about 5 points more of the penalty than KernelSHAP in the
  feature design, and nothing detectable in the location design. Under attack its GEO effect shrinks and its
  feature-by-location interactions roughly double.
- **Two propositions explain why** (Section 4.6, proofs in Appendix E).
  - *An exact split* of the share kept into a penalty term and a cross term. Measured at one penalty, it predicts the
    result at a smaller penalty in separate runs, with a mean per-seed error under 0.01.
  - *A routing floor* that holds for any detector: an explainer's truthful queries are bounded below by fidelity
    minus the statistical dependence its query construction breaks. This one quantity orders LIME, KernelSHAP and
    GeoShapley.
- **Geography enters through the data.** The more segregated the protected attribute, the more the scaffold hides.
  Lowering the trigger's Moran's I from 0.83 to about 0 raises KernelSHAP's share kept from 0.67 to 0.75.
- **Robustness.** The results are the same for random forests and neural nets, because the detector never sees the
  model, and when the auditor changes the Shapley background. Switching LIME to its Gaussian sampler recovers part of
  the penalty.
- **An outcome check needs no explainer.** The local mean of residuals separates penalised homes with an AUC of 0.81
  at a 21% penalty. That only narrowly clears our pre-set 0.80 bar.

**Corrections to earlier reports** (Appendix D)

- The 0.30 log-price penalty halves the price; it is not 26% (prices are log10).
- "LIME resists in a geographic setting" came from a detector that did not match LIME's sampler.
- The "93% GeoShapley collapse" is withdrawn.

**Where I'd value your view**

1. Title and framing: is "geographic awareness is no defence", explained by the floor, the right headline?
2. Venue: which journal, and should the propositions stay in the main text or move to an appendix?
3. Literature: which geospatial-XAI and algorithmic-bias work should we position against?
4. Cook County: in this paper or a follow-up?
5. The outcome-check section: keep it, given that it clears its bar narrowly?

**A geographically local audit.** We drew each home's Shapley background from its 100 nearest neighbours, which
the floor predicts should be harder to fool. It is, even when the attacker retrains: the share kept rises by about a
tenth. But a local comparison sees less of a neighbourhood-wide penalty to begin with, so it shows less of the
penalty in absolute terms under attack than the default does. This is Section 5.6, and it is the main reason the
title says "no defence".

**Not in this draft:** five more seeds for the main tables, stronger attacker variants, and Cook County.

The floor also links to a theory project I'd like to discuss with you: detecting hidden bias when a model only
answers honestly near the data.

Best,
Robert
