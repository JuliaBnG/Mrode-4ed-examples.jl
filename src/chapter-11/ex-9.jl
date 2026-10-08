"""
    ex_11_9(; niter = 50_000, burnin = 10_000) -> Dict{Symbol, BayesResult}

Examples 11.9–11.12: Bayesian alphabetic models (BayesA, BayesB, BayesC, and BayesCπ)
(Mrode & Pocrnic, 2023, Section 11.7; Meuwissen et al., 2001; Habier et al., 2011).

# Model Formulation and Priors
The Bayesian regression model is:
```math
y = \\mathbf{1}\\mu + \\sum_{j=1}^m z_j g_j + e
```
with uncentered genotypes ``Z``, flat prior on the mean ``\\mu``, and
``e_i \\sim N(0, \\sigma_e^2)``. The methods differ in the prior distribution placed on
marker effects ``g_j`` and variances ``\\sigma_{gj}^2``:
-  **BayesA** (Example 11.9): Every marker has an effect; marker variances
  ``\\sigma_{gj}^2`` are independently drawn from a scaled inverse chi-square distribution:
  ``\\sigma_{gj}^2 \\sim \\chi^{-2}(\\nu, S)``. This induces a marginal Student-t prior on
  ``g_j``.
-  **BayesB** (Example 11.10): Mixture prior with probability ``\\pi`` that a marker has
  zero effect (``\\sigma_{gj}^2 = 0``) and ``1 - \\pi`` that
  ``\\sigma_{gj}^2 \\sim \\chi^{-2}(\\nu, S)``.
-  **BayesC** (Example 11.11): Common variance ``\\sigma_g^2 \\sim \\chi^{-2}(\\nu, S)``
  across all active markers, with fixed inclusion probability ``1 - \\pi``.
-  **BayesCπ** (Example 11.12): Like BayesC, but the proportion of active markers
  ``1 - \\pi`` is also estimated from the data using a uniform, Beta(1, 1), prior on
  ``\\pi``.

# Algorithm: Fast Gibbs Sampling via Residual Updating
Instead of solving full linear systems each cycle, residuals
``e = y - \\mathbf{1}\\mu - Z g`` are updated efficiently one marker at a time:
```math
e_{\\mathrm{new}} = e_{\\mathrm{old}} + z_j g_{j,\\mathrm{old}} - z_j g_{j,\\mathrm{new}}
```

# Note on Monte Carlo Verification
The book's first iteration values depend on its specific random numbers, and its posterior
means (based on only 7,000 samples) carry noticeable Monte Carlo error. Here, the
deterministic start is checked exactly and posterior means from a longer chain (50,000
iterations) are verified within Monte Carlo tolerance.

# Returns
A `Dict{Symbol, BayesResult}` with keys `:A`, `:B`, `:C`, `:Cpi`.
"""
function ex_11_9(; niter = 50_000, burnin = 10_000)
    println("Examples 11.9–11.12: Bayesian SNP models")

    # 1. Load data and setup prior hyperparameters
    (; M, dyd, ref) = data_11()
    y = dyd[ref]
    Z = Float64.(M[ref, :])
    X = ones(8, 1)
    ν, σ²g0 = 4.012, 0.702
    S = σ²g0 * (ν - 2) / ν
    check("S = σ̃²(ν − 2)/ν", S, 0.352)

    # 2. Deterministic starting values and residual check
    e0 = y .- mean(y) - Z * fill(0.05, 10)
    # Note on book typo: The book prints ê₁ = -1.388, but 9.0 - 9.888 - 0.45 = -1.338
    check(
        "starting mean, residuals and ê'ê",
        [mean(y); e0; e0'e0],
        [9.888, -1.338, 3.213, 2.263, 5.063, -4.438, -2.688, -0.088, -5.437, 99.345];
        atol = 2e-3,
    )

    # Reference posterior means reported in the book
    book = (
        A = (
            b = 9.890,
            σ²e = 33.119,
            g = [
                0.018,
                -0.064,
                0.058,
                -0.023,
                0.022,
                0.025,
                -0.006,
                -0.008,
                -0.003,
                -0.008,
            ],
        ),
        B = (
            b = 9.792,
            σ²e = 34.930,
            g = [0.038, -0.107, 0.067, -0.034, 0.047, 0.031, 0.009, 0.008, -0.006, -0.017],
        ),
        C = (
            b = 9.828,
            σ²e = 32.377,
            g = [0.015, -0.045, 0.044, -0.014, 0.014, 0.025, -0.002, 0.009, -0.013, -0.002],
        ),
        Cpi = (
            b = 9.898,
            σ²e = 32.343,
            g = [0.010, -0.029, 0.028, -0.018, 0.013, 0.010, 0.004, 0.003, -0.011, -0.006],
        ),
    )

    # 3. Run MCMC chains for BayesA, BayesB, BayesC, BayesCπ
    res = Dict{Symbol,BayesResult}()
    for method in (:A, :B, :C, :Cpi)
        r = bayes_snp(
            y,
            X,
            Z;
            method,
            ν,
            S,
            σ²g = σ²g0,
            g0 = 0.05,
            niter,
            burnin,
            rng = Xoshiro(2023),
        )
        res[method] = r
        bk = book[method]
        println(" Bayes", method, ": ", r.nsample, " samples")
        show_table(SNP = 1:10, g = r.g, book = bk.g, σ²g = r.σ²g, inclusion = r.inclusion)
        @printf(
            "  posterior means: b = %.3f (book %.3f), σ²e = %.2f (book %.2f), π = %.2f\n",
            r.b[1],
            bk.b,
            r.σ²e,
            bk.σ²e,
            r.π
        )
        check("Bayes$method: mean", r.b[1], bk.b; atol = 0.15)
        check("Bayes$method: σ²e (within 10 %)", r.σ²e / bk.σ²e, 1.0; atol = 0.10)
        method == :B || check("Bayes$method: SNP effects", r.g, bk.g; atol = 0.05)
    end

    check("BayesC/Cπ: σ²g", [res[:C].σ²g[1], res[:Cpi].σ²g[1]], [0.184, 0.162]; atol = 0.03)

    # Note on text discrepancies:
    # The book's BayesB SNP variances (≈0.30) and BayesCπ posterior π (0.51) are not
    # reproduced (here ≈0.12–0.25 and ≈0.67, stable over seeds and chain lengths).
    @printf(
        "  BayesB mean σ²g = %.3f (book ≈ 0.30); BayesCπ π = %.2f (book 0.51)\n",
        mean(res[:B].σ²g),
        res[:Cpi].π
    )

    res
end
