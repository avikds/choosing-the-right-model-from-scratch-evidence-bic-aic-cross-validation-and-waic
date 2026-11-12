# Choosing the Right Model from Scratch: Evidence, BIC, AIC, Cross-Validation and WAIC

A polynomial of degree 3 always fits five points at least as well as a straight line, so training fit alone can never tell you which model to believe. Murphy's Probabilistic Machine Learning answers the question five ways: the marginal likelihood (evidence) with its built-in Bayesian Occam's razor, the BIC and MDL penalties that approximate it, AIC, leave-one-out cross-validation and its prequential cousin, and WAIC computed from posterior samples. This project builds all of them from scratch in numpy on the book's own example. You start with Bayes factors for a coin and Jeffreys' scale, build Bayesian linear regression on polynomial features, compute the log evidence two independent ways and reproduce the posterior over degrees from the book (degree 1 wins at N = 5, degree 2 at N = 30), fit the prior and noise precisions by empirical Bayes, watch training likelihood climb while the evidence peaks, check that the Laplace approximation is exact for this model and that BIC's error per data point vanishes, prove numerically that the prequential log likelihood equals the evidence, and estimate WAIC by Monte Carlo. The finale is a study across sample sizes and noise levels: how often each criterion recovers the true degree, and whether averaging over degrees beats picking one when you score held-out predictive density.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** coin_log_evidence

---

Built on Deep-ML.
