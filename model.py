"""
Choosing the Right Model from Scratch: Evidence, BIC, AIC, Cross-Validation and WAIC

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - coin_log_evidence
import math

def log_beta(a, b):
    """
    Compute the natural logarithm of the Beta function:
        log B(a, b) = log Gamma(a) + log Gamma(b) - log Gamma(a+b)

    Uses math.lgamma() for numerical stability.
    """
    if a <= 0 or b <= 0:
        raise ValueError("Beta parameters must be strictly positive.")

    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def coin_log_evidence(n1, n0, a1=1.0, a0=1.0):
    """
    Compute the log marginal likelihoods for two coin models.

    Model 0: Fair coin (theta = 0.5).
    Model 1: Biased coin with a Beta(a1, a0) prior.

    Parameters:
        n1: Number of observed heads.
        n0: Number of observed tails.
        a1: Beta prior parameter for heads.
        a0: Beta prior parameter for tails.

    Returns:
        A tuple containing:
        (log p(D | fair coin), log p(D | Beta-Bernoulli coin))
    """
    if n1 < 0 or n0 < 0:
        raise ValueError("Number of heads and tails cannot be negative.")

    if a1 <= 0 or a0 <= 0:
        raise ValueError("Beta prior parameters must be strictly positive.")

    # Total number of coin tosses.
    N = n1 + n0

    # Model 0: Fair coin.
    # Each toss has probability 0.5, independently.
    log_evidence_fair = N * math.log(0.5)

    # Model 1: Beta-Bernoulli coin.
    # Integrating out the unknown head probability gives:
    # B(a1 + n1, a0 + n0) / B(a1, a0).
    log_evidence_beta = (
        log_beta(a1 + n1, a0 + n0)
        - log_beta(a1, a0)
    )

    return log_evidence_fair, log_evidence_beta

# Step 2 - jeffreys_label
def jeffreys_label(bf):
    """
    Translate a Bayes factor into qualitative evidence
    using Jeffreys' scale.

    bf = p(D | M1) / p(D | M0)
    """

    # Evidence in favor of M0
    if bf < 1 / 100:
        return "decisive for M0"
    elif bf < 1 / 10:
        return "strong for M0"
    elif bf < 1 / 3:
        return "moderate for M0"
    elif bf < 1:
        return "weak for M0"

    # Evidence in favor of M1
    elif bf < 3:
        return "weak for M1"
    elif bf < 10:
        return "moderate for M1"
    elif bf < 100:
        return "strong for M1"
    else:
        return "decisive for M1"

# Step 3 - bayes_factor_table
def bayes_factor_table(n, a1=1.0, a0=1.0):
    """
    Reproduce the Bayesian coin comparison for every possible
    number of heads in n tosses.

    Parameters:
        n: Total number of coin tosses.
        a1: Beta prior parameter for heads.
        a0: Beta prior parameter for tails.

    Returns:
        A list of dictionaries containing:
        - n1: Number of heads.
        - log10_bf: Logarithm (base 10) of the Bayes factor.
        - label: Jeffreys' interpretation of the Bayes factor.
    """

    if n < 0:
        raise ValueError("The number of tosses cannot be negative.")

    results = []

    # Iterate over every possible number of heads.
    for n1 in range(n + 1):

        # Number of tails.
        n0 = n - n1

        # Obtain log evidence for both coin models.
        log_fair, log_biased = coin_log_evidence(
            n1, n0, a1=a1, a0=a0
        )

        # Calculate the log Bayes factor:
        # log(BF) = log p(D | M1) - log p(D | M0)
        log_bf = log_biased - log_fair

        # Convert the natural logarithm into base-10 logarithm.
        log10_bf = log_bf / math.log(10)

        # Calculate the Bayes factor itself for Jeffreys' scale.
        # Handle overflow for extremely large Bayes factors.
        try:
            bf = math.exp(log_bf)
        except OverflowError:
            bf = math.inf

        # Store the result in the required dictionary format.
        results.append({
            "n1": n1,
            "log10_bf": round(log10_bf, 4),
            "label": jeffreys_label(bf)
        })

    return results

