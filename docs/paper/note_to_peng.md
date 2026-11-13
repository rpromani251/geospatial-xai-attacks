Hi Peng,

Attached is draft v1 of the geography-journal paper on scaffold attacks against geospatial explainers. All the results
are from King County. Every number is generated from the result files: the tables and macros are produced by scripts,
and an independent recomputation of about 670 printed numbers and every pre-specified rule outcome agrees with the PDF.

**Main findings**

- **All three explainers are fooled.**
  - A scaffold with a detector trained on each explainer's own queries hides a penalty that halves predicted prices
    in tracts where Black and Latino residents exceed a quarter of the population.
  - LIME is fully fooled: it shows 2% of the penalty while 97% of penalised homes still pay it.
  - KernelSHAP and GeoShapley show 67% and 72% of it, with 86–88% of penalised homes still paying.
- **What the scaffold hides from the Shapley explainers is limited.** It hides the size of the penalty and the
  trigger's rank in global importance. On the penalised homes themselves the trigger stays among the top three
  attributions. Hiding more costs the attacker bias delivered: at 50% fidelity they still show about a quarter of the
  penalty.
- **The estimator, not the geography.**
  - Vulnerability follows where an explainer queries the model. Efficiency, together with near-real coalition
    hybrids, keeps the penalty inside Shapley explanations.
  - The share kept is close to the average, over coalition sizes, of the share of hybrids the detector routes to the
    biased model.
  - GeoShapley's joint location player keeps its hybrids at real locations. That buys it about 5 points of the
    penalty in the feature design and nothing detectable in the location design.
  - Under attack its GEO effect shrinks and its feature-by-location interactions roughly double.
- **Geography enters through the data.** The more spatially clustered the protected attribute, the more the scaffold
  hides: lowering Moran's I from 0.83 to about 0 raises the share kept from 0.67 to 0.75.
- **Robustness.** The results hold for random forests and neural nets; by construction the detector does not depend on
  the model. They also hold when the auditor changes the Shapley background. Switching LIME to its Gaussian sampler
  recovers part of the penalty.
- **An outcome check needs no explainer.** The local mean of residuals separates penalised homes with an AUC of 0.81
  at a 21% penalty. That clears our pre-set 0.80 bar only narrowly.

**Corrections to earlier reports** (Appendix D and CHANGELOG.md)

- The 0.30 log-price penalty halves the price. It is not 26%: prices are log10.
- "LIME resists in a geographic setting" came from a detector that did not match LIME's sampler.
- The "93% GeoShapley collapse" is withdrawn.

**Where I'd value your view**

1. Is the framing right: GeoShapley is "protected little" rather than "no harder to fool"?
2. Should "what is hidden is size and global rank, not presence on penalised homes" be a headline?
3. Which geospatial-XAI and algorithmic-bias literature should I position this against?
4. Should Cook County go into the submission version or a follow-up?

**Not in this draft:** five more seeds for the main tables, attacker-strength variants, any defence evaluation, and
Cook County.

Best,
Robert
