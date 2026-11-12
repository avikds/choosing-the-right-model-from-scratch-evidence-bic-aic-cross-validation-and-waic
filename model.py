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

# Step 4 - coin_bic
def coin_bic(n1, n0):
    """
    Approximate the biased coin's log evidence using BIC.

    Parameters:
        n1: Number of observed heads.
        n0: Number of observed tails.

    Returns:
        BIC approximation of the log evidence in nats.
    """

    if n1 < 0 or n0 < 0:
        raise ValueError("Number of heads and tails cannot be negative.")

    # Total number of tosses.
    N = n1 + n0

    if N == 0:
        raise ValueError("Total number of tosses must be greater than zero.")

    # Maximum likelihood estimate of the probability of heads.
    theta_hat = n1 / N

    # Log likelihood at the maximum likelihood estimate.
    # Apply the convention 0 * log(0) = 0.
    log_likelihood = 0.0

    if n1 > 0:
        log_likelihood += n1 * math.log(theta_hat)

    if n0 > 0:
        log_likelihood += n0 * math.log(1.0 - theta_hat)

    # BIC penalty: (number of parameters / 2) * log(N).
    # The biased coin model has one free parameter.
    bic_penalty = 0.5 * math.log(N)

    # Return the BIC approximation in natural logarithm units (nats).
    return log_likelihood - bic_penalty

# Step 5 - make_poly_data
import numpy as np


def make_poly_data(n, seed=1, noise=5.0):
    """
    Generate the polynomial-regression dataset.

    True underlying function:
        y = (x - 4)^2

    Gaussian noise is added with the specified standard deviation.

    Parameters:
        n: Number of data points.
        seed: Random seed for reproducibility.
        noise: Standard deviation of Gaussian noise.

    Returns:
        A tuple (x, y) containing two NumPy float arrays
        of shape (n,).
    """

    # Initialize the random number generator.
    rs = np.random.RandomState(seed)

    # Generate input values uniformly between 0 and 10.
    x = rs.uniform(0, 10, n)

    # Generate independent standard normal noise.
    eps = rs.normal(size=n)

    # Generate the polynomial output with Gaussian noise.
    y = (x - 4) ** 2 + noise * eps

    return x, y

# Step 6 - poly_design
def poly_design(x, degree, center=0.0, scale=1.0):
    """
    Build the polynomial design matrix.

    Parameters:
        x: Input values (list or NumPy array).
        degree: Maximum polynomial degree.
        center: Value subtracted from each input.
        scale: Value used to scale the centered inputs.

    Returns:
        A float NumPy array of shape (len(x), degree + 1),
        with columns 1, z, z^2, ..., z^degree.
    """

    # Validate the polynomial degree.
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("Degree must be a non-negative integer.")

    # Avoid division by zero during feature scaling.
    if scale == 0:
        raise ValueError("Scale must be non-zero.")

    # Convert the input into a NumPy float array.
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input x must be a one-dimensional array.")

    # Center and scale the input values.
    z = (x - center) / scale

    # Construct polynomial features in ascending degree order.
    # The first column is the constant term (1).
    Phi = np.vander(z, N=degree + 1, increasing=True)

    return Phi.astype(float, copy=False)

# Step 7 - blr_posterior
def blr_posterior(X, y, alpha, beta):
    """
    Compute the Gaussian posterior over Bayesian linear
    regression weights.

    Prior:
        w ~ N(0, alpha^(-1) I)

    Likelihood:
        y ~ N(Xw, beta^(-1) I)

    Parameters:
        X: Design matrix of shape (N, M).
        y: Target vector of shape (N,).
        alpha: Prior precision.
        beta: Observation noise precision.

    Returns:
        m: Posterior mean of shape (M,).
        S: Posterior covariance matrix of shape (M, M).
    """

    # Convert inputs to NumPy float arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of observations.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be strictly positive.")

    # Number of regression weights.
    M = X.shape[1]

    # Compute the posterior precision matrix.
    precision = alpha * np.eye(M) + beta * (X.T @ X)

    # Compute the posterior covariance matrix:
    # S = (alpha * I + beta * X.T @ X)^(-1)
    # Using solve is more numerically stable than explicitly
    # calculating the matrix inverse.
    S = np.linalg.solve(precision, np.eye(M))

    # Compute the posterior mean:
    # m = beta * S * X.T * y
    m = beta * (S @ X.T @ y)

    return m, S

# Step 8 - blr_predict
def blr_predict(Xs, m, S, beta):
    """
    Compute the posterior predictive distribution for new inputs.

    Parameters:
        Xs: Test design matrix of shape (T, M).
        m: Posterior mean vector of shape (M,).
        S: Posterior covariance matrix of shape (M, M).
        beta: Observation noise precision.

    Returns:
        mean: Predictive mean array of shape (T,).
        var: Predictive variance array of shape (T,).
    """

    # Convert inputs into NumPy float arrays.
    Xs = np.asarray(Xs, dtype=float)
    m = np.asarray(m, dtype=float).reshape(-1)
    S = np.asarray(S, dtype=float)

    # Validate the observation noise precision.
    if beta <= 0:
        raise ValueError("Beta must be strictly positive.")

    # Validate input dimensions.
    if Xs.ndim != 2:
        raise ValueError("Xs must be a two-dimensional test design matrix.")

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square covariance matrix.")

    if Xs.shape[1] != m.shape[0]:
        raise ValueError("Xs and m have incompatible dimensions.")

    if S.shape != (m.shape[0], m.shape[0]):
        raise ValueError("S and m have incompatible dimensions.")

    # Compute the posterior predictive mean:
    # mean = Xs @ m
    mean = Xs @ m

    # Compute the uncertainty contributed by the posterior weights:
    # diag(Xs @ S @ Xs.T)
    weight_uncertainty = np.sum((Xs @ S) * Xs, axis=1)

    # Add observation noise variance:
    # var = 1/beta + diag(Xs @ S @ Xs.T)
    var = (1.0 / beta) + weight_uncertainty

    return mean, var

# Step 9 - blr_log_evidence
def blr_log_evidence(X, y, alpha, beta):
    """
    Compute the log marginal likelihood of a Bayesian
    linear regression model in weight space.

    Parameters:
        X: Design matrix of shape (N, M).
        y: Target vector of shape (N,).
        alpha: Prior precision.
        beta: Observation noise precision.

    Returns:
        Log marginal likelihood as a Python float (in nats).
    """

    # Convert inputs into NumPy float arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")

    N, M = X.shape

    if N == 0 or M == 0:
        raise ValueError("X must contain observations and features.")

    if y.shape[0] != N:
        raise ValueError("X and y must contain the same number of observations.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be strictly positive.")

    # Construct the posterior precision matrix:
    # A = alpha * I + beta * X.T @ X
    A = alpha * np.eye(M) + beta * (X.T @ X)

    # Compute the Cholesky factorization:
    # A = L @ L.T
    L = np.linalg.cholesky(A)

    # Compute the posterior mean:
    # m = beta * A^(-1) @ X.T @ y
    # Solve the linear systems instead of explicitly inverting A.
    rhs = beta * (X.T @ y)

    m = np.linalg.solve(
        L.T,
        np.linalg.solve(L, rhs)
    )

    # Compute log determinant of A using its Cholesky factor:
    # log|A| = 2 * sum(log(diag(L)))
    log_det_A = 2.0 * np.sum(np.log(np.diag(L)))

    # Compute the residual between observations and predictions.
    residual = y - X @ m

    # Compute the energy term:
    # E(m) = (beta / 2) * ||y - Xm||^2
    #       + (alpha / 2) * m.T @ m
    E_m = (
        0.5 * beta * (residual @ residual)
        + 0.5 * alpha * (m @ m)
    )

    # Compute the log marginal likelihood:
    # log p(y | X, alpha, beta)
    log_evidence = (
        0.5 * M * math.log(alpha)
        + 0.5 * N * math.log(beta)
        - E_m
        - 0.5 * log_det_A
        - 0.5 * N * math.log(2.0 * math.pi)
    )

    return float(log_evidence)

# Step 10 - gaussian_log_evidence
def gaussian_log_evidence(X, y, alpha, beta):
    """
    Compute the log marginal likelihood in data space.

    The weights are integrated out, giving:
        y ~ N(0, C)

    where:
        C = I/beta + X @ X.T / alpha

    Parameters:
        X: Design matrix of shape (N, M).
        y: Target vector of shape (N,).
        alpha: Prior precision.
        beta: Observation noise precision.

    Returns:
        Log marginal likelihood as a Python float (in nats).
    """

    # Convert inputs into NumPy float arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")

    N, M = X.shape

    if N == 0 or M == 0:
        raise ValueError("X must contain observations and features.")

    if y.shape[0] != N:
        raise ValueError("X and y must contain the same number of observations.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be strictly positive.")

    # Construct the covariance matrix in data space:
    # C = I/beta + X @ X.T / alpha
    C = (1.0 / beta) * np.eye(N) + (X @ X.T) / alpha

    # Compute the Cholesky factorization:
    # C = L @ L.T
    L = np.linalg.cholesky(C)

    # Calculate log determinant using the Cholesky factor:
    # log|C| = 2 * sum(log(diag(L)))
    log_det_C = 2.0 * np.sum(np.log(np.diag(L)))

    # Solve L @ z = y.
    # This gives y.T @ inv(C) @ y = z.T @ z.
    z = np.linalg.solve(L, y)

    # Compute the quadratic form.
    quadratic_term = z @ z

    # Compute the log Gaussian marginal likelihood:
    log_evidence = (
        -0.5 * N * math.log(2.0 * math.pi)
        -0.5 * log_det_C
        -0.5 * quadratic_term
    )

    return float(log_evidence)

# Step 11 - model_posterior
def model_posterior(log_evidences, log_prior=None):
    """
    Compute the posterior probabilities over models using
    their log marginal likelihoods and log prior probabilities.

    Parameters:
        log_evidences: List or NumPy array of log p(D | m).
        log_prior: Optional array of log p(m).
                   Uniform prior is assumed when None.

    Returns:
        A NumPy float array containing posterior probabilities
        that sum to 1.
    """

    # Convert log evidences into a NumPy float array.
    log_evidences = np.asarray(log_evidences, dtype=float)

    # Validate the input.
    if log_evidences.ndim != 1 or log_evidences.size == 0:
        raise ValueError("log_evidences must be a non-empty one-dimensional array.")

    if np.any(np.isnan(log_evidences)) or np.any(np.isposinf(log_evidences)):
        raise ValueError("Log evidences cannot contain NaN or positive infinity.")

    # Use uniform model priors when no log prior is provided.
    if log_prior is None:
        log_joint = log_evidences.copy()
    else:
        log_prior = np.asarray(log_prior, dtype=float)

        if log_prior.shape != log_evidences.shape:
            raise ValueError("log_prior and log_evidences must have the same shape.")

        if np.any(np.isnan(log_prior)) or np.any(np.isposinf(log_prior)):
            raise ValueError("Log priors cannot contain NaN or positive infinity.")

        # Combine log evidence and log prior.
        log_joint = log_evidences + log_prior

    # Ensure at least one model has non-zero posterior support.
    if not np.any(np.isfinite(log_joint)):
        raise ValueError("At least one model must have finite log probability.")

    # Subtract the maximum for numerical stability.
    max_log_joint = np.max(log_joint)

    # Convert log probabilities into unnormalized probabilities.
    probabilities = np.exp(log_joint - max_log_joint)

    # Normalize so that the posterior probabilities sum to 1.
    posterior = probabilities / np.sum(probabilities)

    return np.asarray(posterior, dtype=float)

# Step 12 - fit_alpha_beta
def fit_alpha_beta(X, y, alpha=1.0, beta=1.0, iters=200, tol=1e-8):
    """
    Estimate the prior precision (alpha) and noise precision (beta)
    using empirical Bayes and MacKay's fixed-point updates.

    Parameters:
        X: Design matrix of shape (N, M).
        y: Target vector of shape (N,).
        alpha: Initial prior precision.
        beta: Initial observation noise precision.
        iters: Maximum number of iterations.
        tol: Convergence tolerance.

    Returns:
        A tuple (alpha, beta) of Python floats.
    """

    # Convert inputs to NumPy float arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")

    N, M = X.shape

    if N == 0 or M == 0:
        raise ValueError("X must contain observations and features.")

    if y.shape[0] != N:
        raise ValueError("X and y must contain the same number of observations.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be strictly positive.")

    if iters < 0:
        raise ValueError("Number of iterations cannot be negative.")

    if tol < 0:
        raise ValueError("Tolerance cannot be negative.")

    # Precompute eigenvalues of X.T @ X.
    # Clipping small negative values handles floating-point
    # roundoff because X.T @ X is positive semidefinite.
    eigvals = np.maximum(
        np.linalg.eigvalsh(X.T @ X),
        0.0
    )

    # Precompute X.T @ X and X.T @ y.
    XtX = X.T @ X
    Xty = X.T @ y

    # MacKay's iterative fixed-point updates.
    for _ in range(iters):

        # Step 1: Compute posterior precision matrix.
        A = alpha * np.eye(M) + beta * XtX

        # Compute posterior mean:
        # m = beta * A^(-1) @ X.T @ y
        m = np.linalg.solve(A, beta * Xty)

        # Eigenvalues of beta * X.T @ X.
        lam = beta * eigvals

        # Step 2: Compute effective number of parameters:
        # gamma = sum(lambda_i / (lambda_i + alpha))
        gamma = np.sum(lam / (lam + alpha))

        # Step 3: Update prior precision alpha.
        # alpha_new = gamma / (m.T @ m)
        alpha_new = min(
            gamma / max(float(m @ m), 1e-300),
            1e6
        )

        # Compute residuals.
        residual = y - X @ m

        # Sum of squared residuals.
        residual_sq = float(residual @ residual)

        # Update noise precision beta.
        # beta_new = (N - gamma) / ||y - Xm||^2
        beta_new = min(
            (N - gamma) / max(residual_sq, 1e-300),
            1e8
        )

        # Step 4: Check convergence using relative changes
        # expressed as absolute log ratios.
        alpha_change = abs(math.log(alpha_new / alpha))
        beta_change = abs(math.log(beta_new / beta))

        # Update both precision parameters.
        alpha, beta = alpha_new, beta_new

        # Stop when both parameters have converged.
        if alpha_change < tol and beta_change < tol:
            break

    return float(alpha), float(beta)

# Step 13 - degree_posterior
def degree_posterior(x, y, degrees, alpha=None, beta=None,
                     center=0.0, scale=1.0):
    """
    Compute the posterior probabilities over polynomial degrees.

    Parameters:
        x: Input data.
        y: Target values.
        degrees: Candidate polynomial degrees.
        alpha: Prior precision. If None, empirical Bayes is used.
        beta: Noise precision. Required when alpha is specified.
        center: Centering value for polynomial features.
        scale: Scaling value for polynomial features.

    Returns:
        Dictionary containing:
            degrees: Candidate polynomial degrees.
            log_evidence: Log evidence for each degree, rounded to 2 decimals.
            posterior: Posterior probabilities, rounded to 4 decimals.
            map_degree: Degree with the highest posterior probability.
            hyper: List of (alpha, beta) pairs, rounded to 6 decimals.
    """

    # Convert candidate degrees into a list.
    degrees = list(degrees)

    if len(degrees) == 0:
        raise ValueError("At least one polynomial degree is required.")

    # When alpha is provided, beta must also be provided.
    if alpha is not None and beta is None:
        raise ValueError("Beta must be provided when alpha is specified.")

    log_evidences = []
    hyper = []

    # Evaluate each candidate polynomial degree.
    for d in degrees:

        # Construct the polynomial design matrix.
        X = poly_design(x, d, center=center, scale=scale)

        # Estimate hyperparameters using empirical Bayes
        # when alpha is None.
        if alpha is None:
            a, b = fit_alpha_beta(X, y)
        else:
            # Use the provided hyperparameters for every degree.
            a, b = float(alpha), float(beta)

        # Compute the log marginal likelihood.
        log_ev = blr_log_evidence(X, y, a, b)

        # Store the evidence and hyperparameters.
        log_evidences.append(float(log_ev))
        hyper.append((float(a), float(b)))

    # Compute posterior probabilities with a uniform prior
    # over the candidate polynomial degrees.
    posterior = model_posterior(log_evidences)

    # Identify the MAP degree.
    # np.argmax returns the first occurrence when values are tied.
    map_index = int(np.argmax(posterior))
    map_degree = int(degrees[map_index])

    # Return the required dictionary with the specified rounding.
    return {
        "degrees": [int(d) for d in degrees],
        "log_evidence": [
            round(float(v), 2) for v in log_evidences
        ],
        "posterior": [
            round(float(p), 4) for p in posterior
        ],
        "map_degree": map_degree,
        "hyper": [
            (round(float(a), 6), round(float(b), 6))
            for a, b in hyper
        ]
    }

# Step 14 - occam_curve
def occam_curve(x, y, degrees, alpha, beta, center=0.0, scale=1.0):
    """
    Compare maximum training likelihood with Bayesian model
    evidence for different polynomial degrees.

    Parameters:
        x: Input data.
        y: Target values.
        degrees: List of polynomial degrees to evaluate.
        alpha: Prior precision.
        beta: Observation noise precision.
        center: Centering value for polynomial features.
        scale: Scaling value for polynomial features.

    Returns:
        A list of dictionaries containing:
            degree: Polynomial degree.
            train_ll: Maximum training log likelihood.
            log_evidence: Bayesian log marginal likelihood.
            occam: Log Occam factor (log_evidence - train_ll).
    """

    # Convert inputs into NumPy float arrays.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate inputs.
    if x.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be one-dimensional arrays of equal length.")

    if len(y) == 0:
        raise ValueError("The dataset cannot be empty.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be strictly positive.")

    N = len(y)

    results = []

    # Evaluate each polynomial degree.
    for d in degrees:

        # Construct the polynomial design matrix.
        X = poly_design(x, d, center=center, scale=scale)

        # Find the least-squares weights using the
        # Moore-Penrose pseudoinverse.
        w_hat = np.linalg.lstsq(X, y, rcond=None)[0]

        # Compute residuals at the least-squares solution.
        residual = y - X @ w_hat

        # Compute the maximum training log likelihood:
        # train_ll = (N/2) * log(beta / (2*pi))
        #            - (beta/2) * ||y - Xw_hat||^2
        train_ll = (
            0.5 * N * math.log(beta / (2.0 * math.pi))
            - 0.5 * beta * (residual @ residual)
        )

        # Compute Bayesian log marginal likelihood
        # using the previously implemented function.
        log_ev = blr_log_evidence(X, y, alpha, beta)

        # Compute the log Occam factor.
        occam = log_ev - train_ll

        # Store results in the required format.
        results.append({
            "degree": int(d),
            "train_ll": round(float(train_ll), 2),
            "log_evidence": round(float(log_ev), 2),
            "occam": round(float(occam), 2)
        })

    return results

# Step 15 - information_criteria
def information_criteria(X, y):
    """
    Score a Gaussian linear regression model using BIC, AIC, and MDL.

    Parameters:
        X: Design matrix of shape (N, M).
        y: Target vector of shape (N,).

    Returns:
        Dictionary containing:
            loglik: Maximized Gaussian log likelihood.
            n_params: Number of model parameters (M + 1).
            bic: Bayesian Information Criterion.
            aic: Akaike Information Criterion.
            mdl_bits: Minimum Description Length in bits.
    """

    # Convert inputs into NumPy float arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")

    N, M = X.shape

    if N == 0 or M == 0:
        raise ValueError("X must contain observations and features.")

    if y.shape[0] != N:
        raise ValueError("X and y must contain the same number of observations.")

    # Step 1: Estimate regression weights using maximum likelihood.
    w_hat = np.linalg.lstsq(X, y, rcond=None)[0]

    # Step 2: Compute residuals and residual sum of squares.
    residual = y - X @ w_hat
    rss = float(residual @ residual)

    # Step 3: Estimate the maximum-likelihood noise variance.
    # Apply the specified lower bound to avoid log(0).
    sigma2 = max(rss / N, 1e-300)

    # Step 4: Compute the maximized Gaussian log likelihood.
    # loglik = -N/2 * (log(2*pi*sigma^2) + 1)
    loglik = -0.5 * N * (
        math.log(2.0 * math.pi * sigma2) + 1.0
    )

    # Number of parameters:
    # M regression weights + 1 noise variance parameter.
    n_params = M + 1

    # Step 5: Calculate Bayesian Information Criterion.
    bic = -2.0 * loglik + n_params * math.log(N)

    # Step 6: Calculate Akaike Information Criterion.
    aic = -2.0 * loglik + 2.0 * n_params

    # Step 7: Calculate Minimum Description Length in bits.
    mdl_bits = (
        -loglik + 0.5 * n_params * math.log(N)
    ) / math.log(2.0)

    # Return the results using the required Python types.
    return {
        "loglik": float(loglik),
        "n_params": int(n_params),
        "bic": float(bic),
        "aic": float(aic),
        "mdl_bits": float(mdl_bits)
    }

# Step 16 - laplace_log_evidence
def laplace_log_evidence(X, y, alpha, beta):
    """
    Compute the log marginal likelihood using the Laplace approximation.

    For Bayesian linear regression, the log joint distribution is
    quadratic in the weights, making this approximation exact.

    Parameters:
        X: Design matrix of shape (N, M).
        y: Target vector of shape (N,).
        alpha: Prior precision.
        beta: Observation noise precision.

    Returns:
        Log marginal likelihood as a Python float (in nats).
    """

    # Convert inputs into NumPy float arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")

    N, M = X.shape

    if N == 0 or M == 0:
        raise ValueError("X must contain observations and features.")

    if y.shape[0] != N:
        raise ValueError("X and y must contain the same number of observations.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be strictly positive.")

    # Obtain the MAP estimate of the regression weights.
    # For this Gaussian model, the MAP estimate equals the posterior mean.
    m, _ = blr_posterior(X, y, alpha, beta)

    # Compute residuals at the MAP estimate.
    residual = y - X @ m

    # Log likelihood at the MAP estimate:
    # log p(y | X, m, beta)
    log_likelihood = (
        -0.5 * N * math.log(2.0 * math.pi)
        + 0.5 * N * math.log(beta)
        - 0.5 * beta * (residual @ residual)
    )

    # Log prior probability at the MAP estimate:
    # log p(m | alpha)
    log_prior = (
        0.5 * M * math.log(alpha)
        - 0.5 * M * math.log(2.0 * math.pi)
        - 0.5 * alpha * (m @ m)
    )

    # Construct the Hessian of the negative log joint:
    # H = alpha * I + beta * X.T @ X
    H = alpha * np.eye(M) + beta * (X.T @ X)

    # Calculate log|H| using the sign and log-absolute-determinant.
    sign, log_det_H = np.linalg.slogdet(H)

    if sign <= 0:
        raise ValueError("The Hessian must be positive definite.")

    # Apply the Laplace approximation:
    # log p(D) ≈ log p(D | m) + log p(m)
    #            - 0.5 * log|H| + (M/2) * log(2*pi)
    log_evidence = (
        log_likelihood
        + log_prior
        - 0.5 * log_det_H
        + 0.5 * M * math.log(2.0 * math.pi)
    )

    return float(log_evidence)

# Step 17 - bic_vs_evidence
def bic_vs_evidence(ns, degree, alpha, beta, seed=1, noise=5.0,
                    center=5.0, scale=5.0):
    """
    Compare exact Bayesian log evidence with the BIC approximation
    for different dataset sizes.

    Parameters:
        ns: Iterable of sample sizes.
        degree: Polynomial degree.
        alpha: Prior precision.
        beta: Observation noise precision.
        seed: Random seed for reproducibility.
        noise: Standard deviation of Gaussian noise.
        center: Centering value for polynomial features.
        scale: Scaling value for polynomial features.

    Returns:
        A list of dictionaries containing:
            n: Sample size (integer).
            log_evidence: Exact log marginal likelihood.
            j_bic: BIC approximation to log evidence.
            gap_per_point: Difference per observation.
    """

    results = []

    # Evaluate each requested sample size.
    for n in ns:

        n = int(n)

        if n <= 0:
            raise ValueError("Sample sizes must be positive.")

        # Generate the dataset using the specified random seed.
        x, y = make_poly_data(n, seed=seed, noise=noise)

        # Construct the polynomial design matrix.
        X = poly_design(
            x,
            degree,
            center=center,
            scale=scale
        )

        # Compute the exact Bayesian log marginal likelihood.
        log_evidence = blr_log_evidence(X, y, alpha, beta)

        # Compute the BIC approximation:
        # J_BIC = -0.5 * BIC loss.
        ic = information_criteria(X, y)
        j_bic = -0.5 * ic["bic"]

        # Calculate the approximation error per data point.
        gap_per_point = (log_evidence - j_bic) / n

        # Store the results with the required rounding.
        results.append({
            "n": int(n),
            "log_evidence": round(float(log_evidence), 2),
            "j_bic": round(float(j_bic), 2),
            "gap_per_point": round(float(gap_per_point), 4)
        })

    return results

# Step 18 - loo_log_predictive
def gaussian_logpdf(y, mean, var):
    """
    Compute the elementwise log probability density
    of a Gaussian distribution.

    Parameters:
        y: Observed values.
        mean: Gaussian predictive mean.
        var: Gaussian predictive variance.

    Returns:
        Elementwise Gaussian log densities.
    """

    # Convert inputs to NumPy float arrays.
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    var = np.asarray(var, dtype=float)

    # Variance must be strictly positive.
    if np.any(var <= 0):
        raise ValueError("Gaussian variance must be strictly positive.")

    # Gaussian log probability density:
    # log N(y | mean, var)
    log_density = (
        -0.5 * np.log(2.0 * math.pi * var)
        - (y - mean) ** 2 / (2.0 * var)
    )

    return log_density


def loo_log_predictive(X, y, alpha, beta, plugin=False):
    """
    Compute the leave-one-out predictive log density
    for Bayesian linear regression.

    For every observation, fit the posterior using all
    other observations and predict the held-out observation.

    Parameters:
        X: Design matrix of shape (N, M).
        y: Target vector of shape (N,).
        alpha: Prior precision.
        beta: Observation noise precision.
        plugin: If True, use only the observation noise
                variance (1/beta). Otherwise, use the
                full posterior predictive variance.

    Returns:
        Sum of leave-one-out predictive log densities
        as a Python float.
    """

    # Convert inputs into NumPy float arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")

    N, M = X.shape

    if N == 0 or M == 0:
        raise ValueError("X must contain observations and features.")

    if y.shape[0] != N:
        raise ValueError("X and y must contain the same number of observations.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be strictly positive.")

    # Initialize the cumulative log predictive density.
    total_log_predictive = 0.0

    # Leave out each observation one at a time.
    for n in range(N):

        # Create a mask that excludes the current observation.
        mask = np.ones(N, dtype=bool)
        mask[n] = False

        # Training data excluding observation n.
        X_train = X[mask]
        y_train = y[mask]

        # Fit the Gaussian posterior using the remaining observations.
        m, S = blr_posterior(
            X_train,
            y_train,
            alpha,
            beta
        )

        # Predict the held-out observation using the full posterior.
        mean, var = blr_predict(
            X[n:n + 1],
            m,
            S,
            beta
        )

        # Use the full predictive variance or the plug-in
        # noise variance, depending on the selected option.
        if plugin:
            predictive_var = 1.0 / beta
        else:
            predictive_var = var[0]

        # Calculate the log predictive density of the held-out
        # observation and add it to the running total.
        log_density = gaussian_logpdf(
            y[n],
            mean[0],
            predictive_var
        )

        total_log_predictive += float(log_density)

    return float(total_log_predictive)

# Step 19 - prequential_log_likelihood
def prequential_log_likelihood(X, y, alpha, beta, plugin=False, order=None):
    """
    Compute the prequential log likelihood as a sum of
    one-step-ahead predictive log densities.

    Parameters:
        X: Design matrix of shape (N, M).
        y: Target vector of shape (N,).
        alpha: Prior precision.
        beta: Observation noise precision.
        plugin: If True, use only the noise variance (1/beta).
                Otherwise, use the full posterior predictive variance.
        order: Optional permutation of observation indices.
               Defaults to [0, 1, ..., N-1].

    Returns:
        Sum of sequential predictive log densities as a Python float.
    """

    # Convert inputs into NumPy float arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")

    N, M = X.shape

    if N == 0 or M == 0:
        raise ValueError("X must contain observations and features.")

    if y.shape[0] != N:
        raise ValueError("X and y must contain the same number of observations.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be strictly positive.")

    # Determine the order in which observations are visited.
    if order is None:
        visit_order = list(range(N))
    else:
        order_array = np.asarray(order)

        if order_array.ndim != 1:
            raise ValueError("Order must be a one-dimensional sequence.")

        if not np.issubdtype(order_array.dtype, np.integer):
            raise ValueError("Order must contain integer row indices.")

        visit_order = order_array.tolist()

        # Ensure that order is a valid permutation of all observations.
        if sorted(visit_order) != list(range(N)):
            raise ValueError(
                "Order must contain every row index exactly once."
            )

    # Initialize the cumulative prequential log likelihood.
    total_log_likelihood = 0.0

    # Visit each observation sequentially.
    for k, idx in enumerate(visit_order):

        if k == 0:
            # First observation:
            # No training observations are available.
            # Use the prior distribution:
            # m = 0
            # S = (1/alpha) * I
            m = np.zeros(M, dtype=float)
            S = np.eye(M, dtype=float) / alpha

        else:
            # Use only the observations visited before the current one.
            previous_indices = visit_order[:k]

            X_previous = X[previous_indices]
            y_previous = y[previous_indices]

            # Compute the posterior using the previous observations.
            m, S = blr_posterior(
                X_previous,
                y_previous,
                alpha,
                beta
            )

        # Predict the current observation using its posterior predictive
        # distribution.
        mean, var = blr_predict(
            X[idx:idx + 1],
            m,
            S,
            beta
        )

        # Select the predictive variance.
        if plugin:
            # Plug-in predictive variance:
            # Only observation noise is considered.
            predictive_var = 1.0 / beta
        else:
            # Full posterior predictive variance:
            # Includes observation noise and weight uncertainty.
            predictive_var = float(var[0])

        # Compute the log predictive density of the current observation.
        log_density = gaussian_logpdf(
            y[idx],
            mean[0],
            predictive_var
        )

        # Add the one-step-ahead log predictive density.
        total_log_likelihood += float(log_density)

    return float(total_log_likelihood)

# Step 20 - sample_posterior
def sample_posterior(m, S, n_samples, seed):
    """
    Draw weight vectors from a Gaussian posterior N(m, S).

    Parameters:
        m: Posterior mean vector of shape (M,).
        S: Posterior covariance matrix of shape (M, M).
        n_samples: Number of posterior samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        W: NumPy array of shape (n_samples, M),
           with one posterior weight sample per row.
    """

    # Convert inputs into NumPy float arrays.
    m = np.asarray(m, dtype=float)
    S = np.asarray(S, dtype=float)

    # Validate the posterior mean and covariance dimensions.
    if m.ndim != 1:
        raise ValueError("m must be a one-dimensional vector.")

    M = len(m)

    if S.shape != (M, M):
        raise ValueError("S must have shape (M, M).")

    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    # Initialize the reproducible random number generator.
    rng = np.random.default_rng(seed)

    # Compute the lower Cholesky factor:
    # S = L @ L.T
    L = np.linalg.cholesky(S)

    # Generate independent standard normal samples.
    # Shape: (n_samples, M)
    Z = rng.standard_normal((n_samples, M))

    # Transform the standard normal samples into samples
    # from the Gaussian posterior N(m, S).
    # W = m + Z @ L.T
    W = m + Z @ L.T

    return W

