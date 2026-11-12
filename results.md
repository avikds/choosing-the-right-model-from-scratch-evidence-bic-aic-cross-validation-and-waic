```
1. Bayes factors for a coin (N = 5, uniform prior; Figure 5.4)
   0 heads: log10 BF(1,0) = +0.727 (moderate for M1), BIC = -0.805 nats
   1 heads: log10 BF(1,0) = +0.028 (weak for M1), BIC = -3.307 nats
   2 heads: log10 BF(1,0) = -0.273 (weak for M0), BIC = -4.170 nats
   3 heads: log10 BF(1,0) = -0.273 (weak for M0), BIC = -4.170 nats
   4 heads: log10 BF(1,0) = +0.028 (weak for M1), BIC = -3.307 nats
   5 heads: log10 BF(1,0) = +0.727 (moderate for M1), BIC = -0.805 nats
   Two or three heads favor the fair coin even though a biased coin always fits at least as well.

2. Posterior over polynomial degrees (alpha = 0.01, beta = 0.1; Figures 5.5 and 5.6)
   N =  5: log evidence [-19.04, -22.48, -25.86] | p(m|D) [0.9677, 0.0312, 0.0011] | MAP degree 1
           empirical Bayes, degrees 0-4: alpha hits the cap for 4 of 5 degrees (the data look like noise around zero) | p(m|D) [0.2, 0.2, 0.2, 0.2, 0.2]
   N = 30: log evidence [-171.88, -97.18, -101.11] | p(m|D) [0.0, 0.9807, 0.0193] | MAP degree 2
           empirical Bayes, degrees 0-4: p(m|D) [0.0, 0.0, 0.3491, 0.2355, 0.4154] | MAP degree 4
   (The book's figure titles subtract M/2 log 2pi instead of N/2 log 2pi; the winners are the same.)
   Empirical Bayes fits a looser or tighter prior per degree and moves some mass to higher degrees (see section 7).

3. Occam's razor on the 30 book points: best fit keeps rising, evidence peaks
   degree 0: train log lik  -183.06 | log evidence  -186.30 | log Occam factor   -3.24
   degree 1: train log lik  -164.99 | log evidence  -171.88 | log Occam factor   -6.89
   degree 2: train log lik   -83.71 | log evidence   -97.18 | log Occam factor  -13.47
   degree 3: train log lik   -82.39 | log evidence  -101.11 | log Occam factor  -18.72
   degree 4: train log lik   -81.85 | log evidence  -107.63 | log Occam factor  -25.78
   degree 5: train log lik   -81.34 | log evidence  -115.27 | log Occam factor  -33.93
   degree 6: train log lik   -81.03 | log evidence  -124.09 | log Occam factor  -43.06

4. Information criteria on the 30 book points (losses, lower is better)
   degree 0: BIC  223.64 | AIC  220.83 | MDL  161.3 bits
   degree 1: BIC  222.18 | AIC  217.98 | MDL  160.3 bits
   degree 2: BIC  178.76 | AIC  173.16 | MDL  129.0 bits
   degree 3: BIC  180.27 | AIC  173.26 | MDL  130.0 bits
   degree 4: BIC  182.87 | AIC  174.46 | MDL  131.9 bits
   Laplace = exact for this model: -97.1801 vs -97.1801
   data-space evidence check: -97.1801
   BIC error per data point shrinks with N (degree 2, true noise precision 0.04):
     N =   10: log evidence    -35.22 | J_BIC    -35.63 | gap per point +0.0403
     N =   30: log evidence    -93.22 | J_BIC    -89.38 | gap per point -0.1279
     N =  100: log evidence   -303.51 | J_BIC   -300.28 | gap per point -0.0323
     N =  300: log evidence   -922.96 | J_BIC   -921.87 | gap per point -0.0036
     N = 1000: log evidence  -3059.18 | J_BIC  -3058.14 | gap per point -0.0010

5. Cross-validation and the prequential view (30 book points, alpha = 0.01, beta = 0.1)
   degree 1: evidence  -171.88 | prequential  -171.88 | prequential plug-in  -216.32 | LOO  -175.14 | LOO plug-in  -184.24
   degree 2: evidence   -97.18 | prequential   -97.18 | prequential plug-in  -112.58 | LOO   -87.77 | LOO plug-in   -88.96
   degree 3: evidence  -101.11 | prequential  -101.11 | prequential plug-in  -114.02 | LOO   -87.47 | LOO plug-in   -88.57
   The exact prequential sum is the evidence (eq 5.52); the plug-in versions are not.

6. WAIC from 4000 posterior samples (30 book points, empirical-Bayes precisions per degree)
   degree 0: lppd  -108.47 | p_waic  0.95 (weights 1) | WAIC  218.83
   degree 1: lppd  -106.00 | p_waic  2.22 (weights 2) | WAIC  216.45
   degree 2: lppd   -82.91 | p_waic  2.87 (weights 3) | WAIC  171.57
   degree 3: lppd   -82.24 | p_waic  3.38 (weights 4) | WAIC  171.25
   degree 4: lppd   -82.07 | p_waic  3.53 (weights 5) | WAIC  171.20
   The penalty stays below the weight count (the prior shrinks the extra weights), and degrees 2-4
   predict new points about equally well, so WAIC barely separates them.

7. How often each criterion recovers the true degree 2 (degrees 0-4, 20 data sets per cell)
   noise    N | evidence   BIC   AIC   LOO  WAIC
     2.0    6 |   0.45    0.20  0.25  0.60  0.55
     2.0   12 |   0.85    0.70  0.70  0.85  0.75
     2.0   30 |   0.95    0.90  0.85  0.90  0.85
     2.0  100 |   0.85    0.90  0.65  0.65  0.60
     5.0    6 |   0.15    0.20  0.25  0.35  0.35
     5.0   12 |   0.70    0.70  0.70  0.80  0.80
     5.0   30 |   0.80    0.90  0.85  0.90  0.90
     5.0  100 |   0.70    0.90  0.65  0.55  0.55
    10.0    6 |   0.15    0.15  0.15  0.15  0.05
    10.0   12 |   0.25    0.40  0.50  0.60  0.50
    10.0   30 |   0.55    0.90  0.85  0.80  0.80
    10.0  100 |   0.50    0.90  0.65  0.55  0.55
   Rates move by about 0.1 between seeds at 20 data sets; read only larger differences.

8. Averaging versus choosing: held-out mean log predictive density (noise 5, degrees 0-4, 60 data sets)
   N =  6: mean BMA    -3.813 | MAP degree    -3313.386 | AIC plug-in    -353483.225
           median BMA  -3.538 | MAP degree       -3.631 | AIC plug-in       -147.885 | BMA >= MAP on 0.40 of data sets
   N = 12: mean BMA    -3.330 | MAP degree       -3.349 | AIC plug-in         -6.378
           median BMA  -3.204 | MAP degree       -3.222 | AIC plug-in         -3.544 | BMA >= MAP on 0.62 of data sets
   N = 30: mean BMA    -3.122 | MAP degree       -3.128 | AIC plug-in         -3.196
           median BMA  -3.102 | MAP degree       -3.103 | AIC plug-in         -3.121 | BMA >= MAP on 0.65 of data sets
   Averaging rarely loses much and never collapses; committing to one degree occasionally does.
```
