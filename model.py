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

# Step 21 - waic
def waic(X, y, W, beta):
    """
    Estimate WAIC using Monte Carlo posterior samples.

    Parameters:
        X: Design matrix of shape (N, M).
        y: Target vector of shape (N,).
        W: Posterior weight samples of shape (S, M).
           Each row contains one sampled weight vector.
        beta: Observation noise precision.

    Returns:
        Dictionary containing Python floats:
            lppd: Log pointwise predictive density.
            p_waic: Effective number of parameters.
            waic: Widely Applicable Information Criterion.
            elpd: Expected log predictive density estimate.
    """

    # Convert inputs into NumPy float arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    W = np.asarray(W, dtype=float)

    # Validate input dimensions.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional design matrix.")

    if W.ndim != 2:
        raise ValueError("W must be a two-dimensional array of weight samples.")

    N, M = X.shape
    S = W.shape[0]

    if N == 0 or M == 0:
        raise ValueError("X must contain observations and features.")

    if y.shape[0] != N:
        raise ValueError("X and y must contain the same number of observations.")

    if W.shape[1] != M:
        raise ValueError("W must contain one weight per design-matrix column.")

    if S < 2:
        raise ValueError("At least two posterior samples are required for sample variance.")

    if beta <= 0:
        raise ValueError("Beta must be strictly positive.")

    # Step 1: Compute predictions for every posterior weight sample.
    # W has shape (S, M), and X.T has shape (M, N).
    # Predictions therefore have shape (S, N).
    predictions = W @ X.T

    # Step 2: Compute pointwise log likelihoods.
    # Gaussian observation model:
    # y_n | w_s ~ N(x_n.T @ w_s, 1/beta)
    #
    # log likelihood:
    # ell_sn = -0.5*log(2*pi/beta)
    #          -0.5*beta*(y_n - prediction_sn)^2

    residuals = y[None, :] - predictions

    log_likelihood = (
        -0.5 * math.log(2.0 * math.pi / beta)
        - 0.5 * beta * residuals**2
    )

    # Step 3: Compute the log pointwise predictive density (LPPD).
    # Use the log-mean-exp trick for numerical stability:
    # log(mean(exp(ell))) = max(ell) + log(mean(exp(ell - max(ell))))

    max_per_point = np.max(log_likelihood, axis=0)

    log_mean_exp = (
        max_per_point
        + np.log(
            np.mean(
                np.exp(log_likelihood - max_per_point[None, :]),
                axis=0
            )
        )
    )

    # Sum over all observations.
    lppd = float(np.sum(log_mean_exp))

    # Step 4: Compute the WAIC complexity penalty.
    # Use sample variance across posterior samples (ddof=1).
    pointwise_variance = np.var(
        log_likelihood,
        axis=0,
        ddof=1
    )

    p_waic = float(np.sum(pointwise_variance))

    # Step 5: Compute WAIC.
    # WAIC = -2 * (LPPD - p_WAIC)
    waic_value = float(-2.0 * lppd + 2.0 * p_waic)

    # Step 6: Compute the ELPD estimate.
    # ELPD = LPPD - p_WAIC
    elpd = float(lppd - p_waic)

    # Return all required quantities as Python floats.
    return {
        "lppd": float(lppd),
        "p_waic": float(p_waic),
        "waic": float(waic_value),
        "elpd": float(elpd)
    }

# Step 22 - select_degree
def select_degree(x, y, degrees, center=5.0, scale=5.0,
                  n_samples=400, seed=0):
    """
    Evaluate five model-selection criteria and select the
    best polynomial degree for each criterion.

    Parameters:
        x: Input data.
        y: Target values.
        degrees: Candidate polynomial degrees.
        center: Centering value for polynomial features.
        scale: Scaling value for polynomial features.
        n_samples: Number of posterior samples for WAIC.
        seed: Random seed for posterior sampling.

    Returns:
        Dictionary mapping each criterion to its selected
        polynomial degree:
            evidence: Highest log marginal likelihood.
            bic: Lowest BIC.
            aic: Lowest AIC.
            loo: Highest LOO predictive log density.
            waic: Lowest WAIC.
    """

    # Convert inputs into NumPy float arrays.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Convert candidate degrees into a list.
    degrees = list(degrees)

    # Validate input dimensions.
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array.")

    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of observations.")

    if len(x) == 0:
        raise ValueError("The dataset cannot be empty.")

    if len(degrees) == 0:
        raise ValueError("At least one candidate degree is required.")

    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least 2.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Input data must contain finite values.")

    # Store the scores for each candidate degree.
    evidence_scores = []
    bic_scores = []
    aic_scores = []
    loo_scores = []
    waic_scores = []

    # Evaluate every candidate polynomial degree.
    for d in degrees:

        # Construct the polynomial design matrix.
        X = poly_design(
            x,
            d,
            center=center,
            scale=scale
        )

        # Estimate prior precision (alpha) and noise
        # precision (beta) using empirical Bayes.
        a, b = fit_alpha_beta(X, y)

        # -------------------------------------------------
        # 1. Bayesian model evidence (higher is better)
        # -------------------------------------------------
        evidence = blr_log_evidence(X, y, a, b)

        # -------------------------------------------------
        # 2. BIC and AIC (lower is better)
        # -------------------------------------------------
        ic = information_criteria(X, y)

        bic = ic["bic"]
        aic = ic["aic"]

        # -------------------------------------------------
        # 3. Leave-one-out cross-validation (higher is better)
        # -------------------------------------------------
        loo = loo_log_predictive(
            X,
            y,
            a,
            b
        )

        # -------------------------------------------------
        # 4. WAIC (lower is better)
        # -------------------------------------------------

        # Compute the Gaussian posterior over regression weights.
        m, S = blr_posterior(
            X,
            y,
            a,
            b
        )

        # Draw weight vectors from the posterior distribution.
        W = sample_posterior(
            m,
            S,
            n_samples,
            seed
        )

        # Estimate WAIC using the posterior samples.
        waic_result = waic(
            X,
            y,
            W,
            b
        )

        waic_score = waic_result["waic"]

        # Store the scores for this polynomial degree.
        evidence_scores.append(float(evidence))
        bic_scores.append(float(bic))
        aic_scores.append(float(aic))
        loo_scores.append(float(loo))
        waic_scores.append(float(waic_score))

    # -------------------------------------------------
    # Select the best polynomial degree for each criterion.
    # np.argmax and np.argmin return the first occurrence
    # when multiple values are tied.
    # -------------------------------------------------

    selected = {
        "evidence": int(degrees[int(np.argmax(evidence_scores))]),
        "bic": int(degrees[int(np.argmin(bic_scores))]),
        "aic": int(degrees[int(np.argmin(aic_scores))]),
        "loo": int(degrees[int(np.argmax(loo_scores))]),
        "waic": int(degrees[int(np.argmin(waic_scores))])
    }

    return selected

# Step 23 - agreement_study
def agreement_study(ns, noises, degrees, n_rep, seed=0, true_degree=2):
    """
    Measure how frequently each model-selection criterion
    correctly identifies the true polynomial degree.

    Parameters:
        ns: Iterable of sample sizes.
        noises: Iterable of noise levels.
        degrees: Candidate polynomial degrees.
        n_rep: Number of repetitions per (noise, n) pair.
        seed: Base random seed.
        true_degree: Actual polynomial degree.

    Returns:
        A list of dictionaries containing hit rates for
        evidence, bic, aic, loo, and waic.
    """

    # Convert inputs into lists.
    ns = list(ns)
    noises = list(noises)
    degrees = list(degrees)

    if n_rep <= 0:
        raise ValueError("n_rep must be positive.")

    if not ns or not noises or not degrees:
        raise ValueError("ns, noises, and degrees cannot be empty.")

    if true_degree not in degrees:
        raise ValueError("true_degree must be included in degrees.")

    # Criteria to evaluate.
    criteria = ["evidence", "bic", "aic", "loo", "waic"]

    results = []

    # Loop over noise levels (outer loop).
    for noise in noises:

        # Loop over sample sizes (inner loop).
        for n in ns:

            # Initialize correct-selection counters.
            hits = {criterion: 0 for criterion in criteria}

            # Run repeated experiments.
            for r in range(n_rep):

                # CORRECTED SEED FORMULA:
                # Use a different reproducible dataset for each repetition.
                x, y = make_poly_data(
                    n,
                    seed=seed + 1000 * r + n,
                    noise=noise
                )

                # Evaluate all five criteria on the same dataset.
                selected = select_degree(
                    x,
                    y,
                    degrees,
                    seed=r
                )

                # Count successful recovery of the true degree.
                for criterion in criteria:
                    if selected[criterion] == true_degree:
                        hits[criterion] += 1

            # Calculate hit rates and construct the result dictionary.
            row = {
                "noise": float(noise),
                "n": int(n)
            }

            for criterion in criteria:
                row[criterion] = round(
                    float(hits[criterion] / n_rep),
                    2
                )

            results.append(row)

    return results

# Step 24 - bma_predict
def bma_predict(x, y, xs, degrees, center=5.0, scale=5.0):
    """
    Compute Bayesian model-averaged predictive distributions.

    For each polynomial degree:
        1. Build training and test design matrices.
        2. Estimate alpha and beta using empirical Bayes.
        3. Compute the Gaussian posterior over weights.
        4. Compute the posterior predictive distribution.
        5. Compute the model's log marginal likelihood.

    Parameters:
        x: Training input values.
        y: Training target values.
        xs: Test input values.
        degrees: Candidate polynomial degrees.
        center: Centering value for polynomial features.
        scale: Scaling value for polynomial features.

    Returns:
        weights: Posterior model probabilities as a NumPy array.
        comps: List of (mean, var) predictive distribution pairs,
               ordered according to degrees.
    """

    # Convert inputs into NumPy float arrays.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    xs = np.asarray(xs, dtype=float)

    # Convert candidate degrees into a list.
    degrees = list(degrees)

    # Validate inputs.
    if x.ndim != 1 or xs.ndim != 1:
        raise ValueError("x and xs must be one-dimensional arrays.")

    if len(x) != len(y):
        raise ValueError("x and y must have the same number of observations.")

    if len(x) == 0 or len(xs) == 0:
        raise ValueError("Training and test datasets cannot be empty.")

    if len(degrees) == 0:
        raise ValueError("At least one polynomial degree is required.")

    log_evidences = []
    comps = []

    # Evaluate every candidate polynomial degree.
    for d in degrees:

        # Construct the training and test design matrices.
        X = poly_design(x, d, center=center, scale=scale)
        Xs = poly_design(xs, d, center=center, scale=scale)

        # Estimate prior and noise precisions using empirical Bayes.
        alpha, beta = fit_alpha_beta(X, y)

        # Compute the Gaussian posterior over regression weights.
        m, S = blr_posterior(X, y, alpha, beta)

        # Compute the posterior predictive mean and variance.
        mean, var = blr_predict(Xs, m, S, beta)

        # Compute the log marginal likelihood for this degree.
        log_ev = blr_log_evidence(X, y, alpha, beta)

        # Store the evidence and predictive distribution.
        log_evidences.append(float(log_ev))
        comps.append((mean, var))

    # Convert log evidences into posterior model probabilities.
    # A uniform prior over polynomial degrees is assumed.
    weights = model_posterior(log_evidences)

    return weights, comps


def bma_log_density(ys, weights, comps):
    """
    Compute the log predictive density of a Bayesian
    model-averaged mixture distribution.

    The predictive density is:
        p(ys | D) = sum_m weights[m] *
                    N(ys | mean_m, var_m)

    Parameters:
        ys: Test target values.
        weights: Posterior probabilities of the models.
        comps: List of (mean, var) predictive distribution pairs.

    Returns:
        NumPy float array of shape (len(ys),) containing
        the log predictive density for each test target.
    """

    # Convert inputs into NumPy float arrays.
    ys = np.asarray(ys, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float)

    # Convert components into a list.
    comps = list(comps)

    # Validate inputs.
    if ys.size == 0:
        raise ValueError("ys cannot be empty.")

    if weights.ndim != 1:
        raise ValueError("weights must be a one-dimensional array.")

    if len(weights) == 0 or len(weights) != len(comps):
        raise ValueError("weights and comps must have the same non-zero length.")

    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Model weights must be finite and non-negative.")

    log_terms = []

    # Compute the log predictive density for each model.
    for weight, (mean, var) in zip(weights, comps):

        mean = np.asarray(mean, dtype=float).reshape(-1)
        var = np.asarray(var, dtype=float).reshape(-1)

        if mean.shape != ys.shape or var.shape != ys.shape:
            raise ValueError(
                "Each predictive mean and variance must match the shape of ys."
            )

        # Replace exactly zero model weights with 1e-300
        # before taking the logarithm, as specified.
        log_weight = math.log(max(float(weight), 1e-300))

        # Log of the model-weighted Gaussian predictive density.
        log_density = (
            log_weight
            + gaussian_logpdf(ys, mean, var)
        )

        log_terms.append(log_density)

    # Stack model-specific log densities.
    # Shape: (number of models, number of test observations).
    log_terms = np.vstack(log_terms)

    # Stable log-sum-exp across models.
    # Subtract the maximum before exponentiation to avoid underflow.
    max_log = np.max(log_terms, axis=0)

    log_mixture_density = (
        max_log
        + np.log(
            np.sum(
                np.exp(log_terms - max_log[None, :]),
                axis=0
            )
        )
    )

    return np.asarray(log_mixture_density, dtype=float)

# Step 25 - bma_vs_map
def bma_vs_map(n, noise, degrees, n_rep, seed=0, n_test=200,
               center=5.0, scale=5.0):
    """
    Compare Bayesian Model Averaging (BMA), MAP model selection,
    and AIC plug-in prediction using held-out predictive density.

    Parameters:
        n: Number of training observations.
        noise: Standard deviation of Gaussian noise.
        degrees: Candidate polynomial degrees.
        n_rep: Number of simulation repetitions.
        seed: Base random seed.
        n_test: Number of held-out test observations.
        center: Centering value for polynomial features.
        scale: Scaling value for polynomial features.

    Returns:
        Dictionary containing:
            bma: Mean BMA predictive log density.
            map: Mean MAP predictive log density.
            aic_plugin: Mean AIC plug-in predictive log density.
            bma_median: Median BMA predictive log density.
            map_median: Median MAP predictive log density.
            aic_plugin_median: Median AIC plug-in predictive log density.
            bma_beats_map: Fraction of repetitions where BMA
                           scores at least as well as MAP.
    """

    # Convert candidate degrees into a list.
    degrees = list(degrees)

    # Validate the inputs.
    if n <= 0:
        raise ValueError("Training sample size must be positive.")

    if n_test <= 0:
        raise ValueError("Test sample size must be positive.")

    if n_rep <= 0:
        raise ValueError("Number of repetitions must be positive.")

    if len(degrees) == 0:
        raise ValueError("At least one candidate degree is required.")

    if noise < 0:
        raise ValueError("Noise standard deviation cannot be negative.")

    # Store the predictive scores from every repetition.
    bma_scores = []
    map_scores = []
    aic_plugin_scores = []

    # Repeat the experiment using different random seeds.
    for r in range(n_rep):

        # -------------------------------------------------
        # 1. Generate training data.
        # -------------------------------------------------
        x, y = make_poly_data(
            n,
            seed=seed + r,
            noise=noise
        )

        # -------------------------------------------------
        # 2. Generate independent held-out test data.
        # -------------------------------------------------
        rs = np.random.RandomState(10000 + seed + r)

        xs = rs.uniform(0, 10, n_test)

        ys = (
            (xs - 4) ** 2
            + noise * rs.normal(size=n_test)
        )

        # -------------------------------------------------
        # 3. Bayesian Model Averaging (BMA).
        # -------------------------------------------------

        # Obtain posterior model probabilities and predictive
        # distributions for every candidate degree.
        weights, comps = bma_predict(
            x,
            y,
            xs,
            degrees,
            center=center,
            scale=scale
        )

        # Compute the mixture log predictive density.
        bma_log_scores = bma_log_density(
            ys,
            weights,
            comps
        )

        # Mean log predictive density across test observations.
        bma_score = float(np.mean(bma_log_scores))

        # -------------------------------------------------
        # 4. MAP model prediction.
        # -------------------------------------------------

        # Select the model with the highest posterior probability.
        # np.argmax selects the first model in case of a tie.
        map_idx = int(np.argmax(weights))

        # Obtain the predictive mean and variance of that model.
        map_mean, map_var = comps[map_idx]

        # Score the selected model on the held-out test data.
        map_log_scores = gaussian_logpdf(
            ys,
            map_mean,
            map_var
        )

        map_score = float(np.mean(map_log_scores))

        # -------------------------------------------------
        # 5. AIC plug-in prediction.
        # -------------------------------------------------

        # Calculate the AIC score for every candidate degree.
        aic_scores = []

        design_matrices = []

        for d in degrees:

            X = poly_design(
                x,
                d,
                center=center,
                scale=scale
            )

            design_matrices.append(X)

            ic = information_criteria(X, y)

            aic_scores.append(ic["aic"])

        # Select the degree with the lowest AIC.
        aic_idx = int(np.argmin(aic_scores))

        # Retrieve the training design matrix for the selected degree.
        X = design_matrices[aic_idx]

        # Construct the test design matrix.
        Xs = poly_design(
            xs,
            degrees[aic_idx],
            center=center,
            scale=scale
        )

        # Fit the least-squares weights using training data.
        w_hat = np.linalg.lstsq(
            X,
            y,
            rcond=None
        )[0]

        # Calculate the training residual sum of squares.
        residual = y - X @ w_hat

        rss = float(residual @ residual)

        # Maximum-likelihood estimate of the noise variance:
        # sigma_hat^2 = RSS / n
        sigma2 = max(rss / n, 1e-300)

        # Predict the held-out observations using the
        # least-squares weights.
        aic_mean = Xs @ w_hat

        # Score the AIC plug-in model using its MLE variance.
        aic_log_scores = gaussian_logpdf(
            ys,
            aic_mean,
            sigma2
        )

        aic_plugin_score = float(np.mean(aic_log_scores))

        # -------------------------------------------------
        # 6. Store the results for this repetition.
        # -------------------------------------------------

        bma_scores.append(bma_score)
        map_scores.append(map_score)
        aic_plugin_scores.append(aic_plugin_score)

    # Convert score lists into NumPy arrays for aggregation.
    bma_scores = np.asarray(bma_scores, dtype=float)
    map_scores = np.asarray(map_scores, dtype=float)
    aic_plugin_scores = np.asarray(aic_plugin_scores, dtype=float)

    # Calculate mean predictive log densities.
    bma_mean = float(np.mean(bma_scores))
    map_mean = float(np.mean(map_scores))
    aic_plugin_mean = float(np.mean(aic_plugin_scores))

    # Calculate median predictive log densities.
    bma_median = float(np.median(bma_scores))
    map_median = float(np.median(map_scores))
    aic_plugin_median = float(np.median(aic_plugin_scores))

    # Fraction of repetitions in which BMA performs
    # at least as well as MAP.
    bma_beats_map = float(
        np.mean(bma_scores >= map_scores)
    )

    # Return all required results rounded to the specified precision.
    return {
        "bma": round(bma_mean, 3),
        "map": round(map_mean, 3),
        "aic_plugin": round(aic_plugin_mean, 3),
        "bma_median": round(bma_median, 3),
        "map_median": round(map_median, 3),
        "aic_plugin_median": round(aic_plugin_median, 3),
        "bma_beats_map": round(bma_beats_map, 2)
    }

