"""
Choosing the Right Model from Scratch: Evidence, BIC, AIC, Cross-Validation and WAIC scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Choosing the right model from scratch: evidence, BIC, AIC, cross-validation and WAIC.

Story (Murphy, Probabilistic Machine Learning, Section 5.2): training fit always prefers the
most flexible model, so model choice needs something else. The marginal likelihood averages the
fit over the prior and carries its own Occam penalty; BIC and MDL approximate it; AIC, leave-one-
out cross-validation and WAIC estimate predictive accuracy instead. On the book's polynomial
example they agree when the data are plentiful and part ways when they are not, and averaging
over models beats committing to one when the data are scarce.
"""
import math
import numpy as np


def main() -> None:
    print("1. Bayes factors for a coin (N = 5, uniform prior; Figure 5.4)")
    for r in bayes_factor_table(5):
        n1 = r["n1"]
        print(f"   {n1} heads: log10 BF(1,0) = {r['log10_bf']:+.3f} ({r['label']}), BIC = {coin_bic(n1, 5 - n1):.3f} nats")
    print("   Two or three heads favor the fair coin even though a biased coin always fits at least as well.")

    print("\n2. Posterior over polynomial degrees (alpha = 0.01, beta = 0.1; Figures 5.5 and 5.6)")
    for n in (5, 30):
        x, y = make_poly_data(n)
        r = degree_posterior(x, y, [1, 2, 3], 0.01, 0.1)
        print(f"   N = {n:>2}: log evidence {r['log_evidence']} | p(m|D) {r['posterior']} | MAP degree {r['map_degree']}")
        eb = degree_posterior(x, y, [0, 1, 2, 3, 4], center=5.0, scale=5.0)
        capped = sum(1 for a, b in eb["hyper"] if a >= 1e6)
        if capped:
            print(f"           empirical Bayes, degrees 0-4: alpha hits the cap for {capped} of 5 degrees (the data look like noise around zero) | p(m|D) {eb['posterior']}")
        else:
            print(f"           empirical Bayes, degrees 0-4: p(m|D) {eb['posterior']} | MAP degree {eb['map_degree']}")
    print("   (The book's figure titles subtract M/2 log 2pi instead of N/2 log 2pi; the winners are the same.)")
    print("   Empirical Bayes fits a looser or tighter prior per degree and moves some mass to higher degrees (see section 7).")

    print("\n3. Occam's razor on the 30 book points: best fit keeps rising, evidence peaks")
    x30, y30 = make_poly_data(30)
    for r in occam_curve(x30, y30, range(7), 0.01, 0.1):
        print(f"   degree {r['degree']}: train log lik {r['train_ll']:>8.2f} | log evidence {r['log_evidence']:>8.2f} | log Occam factor {r['occam']:>7.2f}")

    print("\n4. Information criteria on the 30 book points (losses, lower is better)")
    for d in range(5):
        ic = information_criteria(poly_design(x30, d), y30)
        print(f"   degree {d}: BIC {ic['bic']:>7.2f} | AIC {ic['aic']:>7.2f} | MDL {ic['mdl_bits']:>6.1f} bits")
    X2 = poly_design(x30, 2)
    print(f"   Laplace = exact for this model: {laplace_log_evidence(X2, y30, 0.01, 0.1):.4f} vs {blr_log_evidence(X2, y30, 0.01, 0.1):.4f}")
    print(f"   data-space evidence check: {gaussian_log_evidence(X2, y30, 0.01, 0.1):.4f}")
    print("   BIC error per data point shrinks with N (degree 2, true noise precision 0.04):")
    for r in bic_vs_evidence([10, 30, 100, 300, 1000], 2, 0.01, 0.04):
        print(f"     N = {r['n']:>4}: log evidence {r['log_evidence']:>9.2f} | J_BIC {r['j_bic']:>9.2f} | gap per point {r['gap_per_point']:+.4f}")

    print("\n5. Cross-validation and the prequential view (30 book points, alpha = 0.01, beta = 0.1)")
    for d in (1, 2, 3):
        X = poly_design(x30, d)
        print(f"   degree {d}: evidence {blr_log_evidence(X, y30, 0.01, 0.1):>8.2f} | prequential {prequential_log_likelihood(X, y30, 0.01, 0.1):>8.2f}"
              f" | prequential plug-in {prequential_log_likelihood(X, y30, 0.01, 0.1, plugin=True):>8.2f}"
              f" | LOO {loo_log_predictive(X, y30, 0.01, 0.1):>8.2f} | LOO plug-in {loo_log_predictive(X, y30, 0.01, 0.1, plugin=True):>8.2f}")
    print("   The exact prequential sum is the evidence (eq 5.52); the plug-in versions are not.")

    print("\n6. WAIC from 4000 posterior samples (30 book points, empirical-Bayes precisions per degree)")
    for d in range(5):
        X = poly_design(x30, d, 5.0, 5.0)
        a, b = fit_alpha_beta(X, y30)
        m, S = blr_posterior(X, y30, a, b)
        w = waic(X, y30, sample_posterior(m, S, 4000, d), b)
        print(f"   degree {d}: lppd {w['lppd']:>8.2f} | p_waic {w['p_waic']:>5.2f} (weights {d + 1}) | WAIC {w['waic']:>7.2f}")
    print("   The penalty stays below the weight count (the prior shrinks the extra weights), and degrees 2-4")
    print("   predict new points about equally well, so WAIC barely separates them.")

    print("\n7. How often each criterion recovers the true degree 2 (degrees 0-4, 20 data sets per cell)")
    print("   noise    N | evidence   BIC   AIC   LOO  WAIC")
    for r in agreement_study([6, 12, 30, 100], [2.0, 5.0, 10.0], [0, 1, 2, 3, 4], 20):
        print(f"   {r['noise']:>5.1f} {r['n']:>4} |   {r['evidence']:.2f}    {r['bic']:.2f}  {r['aic']:.2f}  {r['loo']:.2f}  {r['waic']:.2f}")
    print("   Rates move by about 0.1 between seeds at 20 data sets; read only larger differences.")

    print("\n8. Averaging versus choosing: held-out mean log predictive density (noise 5, degrees 0-4, 60 data sets)")
    for n in (6, 12, 30):
        r = bma_vs_map(n, 5.0, [0, 1, 2, 3, 4], 60)
        print(f"   N = {n:>2}: mean BMA {r['bma']:>9.3f} | MAP degree {r['map']:>12.3f} | AIC plug-in {r['aic_plugin']:>14.3f}")
        print(f"           median BMA {r['bma_median']:>7.3f} | MAP degree {r['map_median']:>12.3f} | AIC plug-in {r['aic_plugin_median']:>14.3f} | BMA >= MAP on {r['bma_beats_map']:.2f} of data sets")
    print("   Averaging rarely loses much and never collapses; committing to one degree occasionally does.")


if __name__ == "__main__":
    main()

